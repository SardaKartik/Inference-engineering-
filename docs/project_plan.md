# Project Plan — Production LLM Inference Stack

This is the master reference for the entire project.
Read this before starting any phase. Each phase builds on the previous one.

---

## The Three Phases at a Glance

```
Phase 1 — Single user, single request
  Goal: find the latency floor for this model on this hardware
  Question: what is the fastest this model can possibly respond?
        ↓
Phase 2 — Single user, multiple requests (conversation)
  Goal: understand how latency degrades as context grows
  Question: how does TTFT change across a 5-turn conversation?
        ↓
Phase 3 — Multiple users, concurrent requests
  Goal: find the GPU ceiling and understand infrastructure scaling
  Question: at what concurrency does 1 GPU break, and what does adding a second one buy?
```

Do not skip ahead. If you do not understand single-request latency, you cannot
diagnose multi-user problems. If you do not have a clean single-user baseline,
your multi-user numbers have no reference point.

---

## What We Are Building and Why

We are not training a model. We are not changing the model.

We are solving the problem of **serving** a 14B parameter model efficiently —
getting tokens in front of users as fast as possible, under real load, in a
way that can be operated and scaled.

This is the work of an **inference engineer / ML platform engineer**. At companies
like Cursor, GitHub Copilot, or any LLM product — there is a team that does
exactly this. They take existing models and build the infrastructure to serve
them to many users with low latency and high reliability.

---

## The Model and Hardware

| Item | Value |
|------|-------|
| Model | Qwen2.5-Coder-14B-Instruct-AWQ |
| Quantization | AWQ (4-bit) |
| Disk size | ~9 GB |
| VRAM (model weights) | ~10 GB |
| GPU | A100 80GB |
| VRAM available for KV cache | ~60 GB |
| Serving engine | vLLM |

**AWQ** means the model is already compressed to 4-bit. This is what makes a
14B model fit in 10 GB of VRAM instead of ~28 GB. The Qwen team did this
compression — we use the result.

---

---

# Phase 1 — Single User, Single Request

## Goal

Find the TTFT and TPOT floor for this model on this hardware.
Understand which parameters actually affect latency and which do nothing.

## The Core Question

> What is the minimum achievable TTFT for a ~450 token prompt on an A100 80GB?

## What You Will Learn

- How long the prefill phase takes (processing the input prompt)
- How fast the decode phase runs (generating output tokens)
- Which vLLM parameters move the numbers and which do not
- How to run a structured tuning loop with one variable at a time

---

## Steps

### Step 1 — Establish the Baseline

Start vLLM with the default parameters from `serve.sh`.
Run the benchmark at 1 user only:

```bash
bash scripts/benchmark.sh 1
```

Record these numbers before changing anything:

| Metric | Baseline Value |
|--------|---------------|
| TTFT P50 | |
| TTFT P95 | |
| TPOT (ms/token) | |
| Avg Latency (ms) | |
| Tokens/sec | |

This is your baseline. Every parameter change will be compared against this.

---

### Step 2 — Run the Tuning Loop

Change **one parameter at a time**. Restart vLLM. Run benchmark at 1 user.
Record the delta. Then decide whether to keep the change.

**Tuning order (try in this sequence):**

**Change 1 — Reduce max_model_len**
```bash
# In serve.sh: change --max-model-len 8192 to --max-model-len 4096
```
Why: smaller context window = less memory vLLM reserves per sequence = faster prefill.
Our prompt is ~450 tokens and output is 512. We do not need 8192.
Expected: TTFT drops 10-30%.

**Change 2 — Explicit dtype**
```bash
# In serve.sh: add --dtype bfloat16
```
Why: A100 is natively optimized for bfloat16. vLLM auto-detects but explicit is safer.
Expected: small improvement or no change. Worth confirming.

**Change 3 — Enable chunked prefill**
```bash
# In serve.sh: add --enable-chunked-prefill
```
Why: breaks the prefill phase into smaller chunks. Can reduce TTFT for long prompts
because the GPU starts generating while still processing later chunks.
Expected: helpful for long prompts (800+ tokens). May not help much at 450 tokens.

**Change 4 — Adjust gpu_memory_utilization**
```bash
# In serve.sh: try --gpu-memory-utilization 0.90
```
Why: more VRAM for KV cache. At single user this barely matters but record it —
the difference becomes visible in Phase 3.

**Tuning record table — fill this in as you go:**

| Change | TTFT P50 | TTFT P95 | TPOT | Delta vs Baseline | Keep? |
|--------|----------|----------|------|-------------------|-------|
| Baseline | | | | — | — |
| max_model_len 4096 | | | | | |
| + dtype bfloat16 | | | | | |
| + chunked prefill | | | | | |
| gpu_memory_utilization 0.90 | | | | | |

---

### Step 3 — Understand the Numbers

After tuning, answer these questions in writing (add your answers below the table):

1. What was your TTFT baseline? After best tuning?
2. Which single change had the biggest impact?
3. What is your TPOT? Is it consistent across runs?
4. Did any change make things worse?

**Expected realistic numbers on A100 80GB with Qwen 14B AWQ:**
```
TTFT:  0.5s — 1.5s (mostly hardware-bound, prefill of 450 tokens)
TPOT:  20ms — 40ms per token
```
If your numbers are in this range, the model is performing correctly.
Do not spend time trying to go below the hardware floor.

### Step 4 — Quality Check

After every parameter change, manually read 2-3 model outputs.
Check that the Python code the model generates still works correctly.
Quantization already compresses quality — further tuning should not
degrade it further, but verify this yourself rather than assuming.

---

## Difficulties to Expect

**Most single-user latency is hardware-bound.**
You will not move TTFT from 3s to 0.5s through parameter tuning alone.
The A100 processes prefill tokens at a fixed rate. The goal is to remove
overhead, not to defy physics.

**The numbers will vary slightly between runs.**
Run each configuration 2-3 times and take the median. Single-run results
have noise — do not conclude one change helped based on a single run.

**vLLM startup takes time.**
Every time you change a parameter in serve.sh and restart, vLLM reloads
the model (~2 minutes). This is not a bug — this is the cold start cost.

---

## Exit Criteria — Phase 1 is Done When

- [ ] Baseline numbers recorded at 1 user
- [ ] At least 3 parameter changes tried and results recorded
- [ ] You know which parameter had the biggest impact on TTFT
- [ ] Best parameter configuration saved in serve.sh
- [ ] You can explain in one sentence why TTFT is what it is
- [ ] Quality check done — model outputs still correct after tuning

---

---

# Phase 2 — Single User, Multiple Requests (Conversation)

## Goal

Understand how TTFT grows as conversation history accumulates.
Understand whether prefix caching helps and by how much.

## The Core Question

> If a single user sends 5 messages in a conversation, how does TTFT change
> from message 1 to message 5?

## Why This Phase Matters

In Phase 1, every request is independent — same prompt, fresh context.
In a real chatbot or coding assistant, each new message appends to the history.

What happens to the prefill cost:
```
Turn 1: 450 tokens to prefill → TTFT: ~0.8s
Turn 2: 450 + 512 (model reply) + 150 (new message) = ~1100 tokens → TTFT: ?
Turn 3: ~1800 tokens → TTFT: ?
Turn 4: ~2500 tokens → TTFT: ?
Turn 5: ~3200 tokens → TTFT: ?
```

TTFT grows because the model must process the entire history on each turn.
This is called **prefill cost growth**. It is a real production problem.
Most engineers do not discover it until users complain about slow later messages.

---

## New Concept — Prefix Caching

vLLM can cache the KV state of a shared prefix (like the system prompt) and
reuse it across turns instead of recomputing it. This reduces the prefill cost
for each turn because the system prompt portion is already computed.

Enable it in serve.sh:
```bash
# Add this flag to serve.sh for Phase 2
--enable-prefix-caching
```

The experiment: run the same 5-turn conversation with and without this flag.
Compare TTFT at each turn. The difference is the value of prefix caching.

---

## Steps

### Step 1 — Build the Conversation Locustfile

A conversation requires each virtual user to maintain state across turns.
The existing `locustfile.py` sends independent requests — it does not simulate
a real conversation.

Create `benchmarks/conversation_locustfile.py`.

Key difference from the original locustfile:

```python
class ConversationUser(HttpUser):

    def on_start(self):
        # Each user gets their own conversation history
        # on_start runs once when the user is created
        self.conversation_history = []
        self.turn_number = 0

    @task
    def send_turn(self):
        self.turn_number += 1

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": USER_MESSAGE
        })

        # Send the full history every time
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT}
            ] + self.conversation_history,
            "max_tokens": 512,
            "stream": True
        }

        # ... make the request, capture TTFT ...

        # After response — append model reply to history
        self.conversation_history.append({
            "role": "assistant",
            "content": collected_reply
        })
```

The key pattern: `self.conversation_history` grows with each turn.
The payload sends the full history every time. This is how real chatbots work.

### Step 2 — Run 5-Turn Conversations Without Prefix Caching

Serve the model without `--enable-prefix-caching` (default).
Run the conversation locustfile at 1 user for 5 turns.

Record TTFT at each turn:

| Turn | Approx Token Count | TTFT (no prefix cache) | TTFT (with prefix cache) |
|------|-------------------|----------------------|------------------------|
| 1 | ~450 | | |
| 2 | ~1100 | | |
| 3 | ~1800 | | |
| 4 | ~2500 | | |
| 5 | ~3200 | | |

### Step 3 — Enable Prefix Caching and Re-run

Add `--enable-prefix-caching` to serve.sh. Restart vLLM. Run the same test.
Fill in the right column above. Calculate the delta.

### Step 4 — Find the Context Limit

With max_model_len at 4096 (from Phase 1 tuning), the context fills around turn 7-8.
What happens when you hit the limit? Does vLLM truncate? Error? Queue?
Run until you hit it and record what happens. This is a real production edge case.

---

## Difficulties to Expect

**Stateful Locust users are non-trivial.**
The default Locust model is stateless between tasks. Maintaining conversation
history as an instance variable works, but requires careful handling. If the
user object is reset, history is lost. Test locally before running on RunPod.

**Token counting is approximate.**
The "approx token count" column above is an estimate. Actual token counts depend
on how the tokenizer counts your specific text. Close enough for planning — the
exact numbers come from running it.

**Context window limit is a hard ceiling.**
With max_model_len=4096, long conversations will hit an error. This is expected
and informative — it teaches you why production systems often use shorter
max_model_len and sliding window strategies.

---

## Exit Criteria — Phase 2 is Done When

- [ ] conversation_locustfile.py written and tested
- [ ] TTFT recorded at turns 1, 3, and 5 without prefix caching
- [ ] TTFT recorded at turns 1, 3, and 5 with prefix caching
- [ ] You can state: "prefix caching saved X ms by turn 5"
- [ ] You know what happens when the context limit is hit
- [ ] You can explain in one sentence why TTFT grows across turns

---

---

# Phase 3 — Multiple Users, Concurrent Requests

## Goal

Find the concurrency ceiling of a single GPU. Then understand what
adding infrastructure (second GPU, Kubernetes) actually buys.

## The Core Question

> At what number of concurrent users does a single A100 80GB become the bottleneck?
> And what does a second GPU actually do to latency and throughput?

## Three Sub-Phases (do in order)

---

### Phase 3a — Multi-User on Single GPU (no infrastructure)

Use the same vLLM server from Phase 1 (best parameters from tuning).
Run the original locustfile.py (single-request, not conversation) at increasing concurrency.

```bash
bash scripts/benchmark.sh 1     # already done in Phase 1
bash scripts/benchmark.sh 5
bash scripts/benchmark.sh 10
bash scripts/benchmark.sh 25
bash scripts/benchmark.sh 50
```

Fill the benchmark table:

| Users | TTFT P50 | TTFT P95 | Avg Latency | Tokens/sec | KV Cache % | Requests Waiting |
|-------|----------|----------|-------------|------------|------------|-----------------|
| 1 | | | | | | |
| 5 | | | | | | |
| 10 | | | | | | |
| 25 | | | | | | |
| 50 | | | | | | |

What to find:
- **The saturation point** — the user count where TTFT first spikes badly.
  When `num_requests_waiting` goes above 0, the system is overloaded.
- **KV cache ceiling** — at what user count does cache usage hit 90%+?
- **GPU utilization** — is the GPU the bottleneck or is memory bandwidth?

---

### Phase 3b — Tensor Parallelism on 2 GPUs (no Kubernetes)

Tensor parallelism splits the model across 2 GPUs. Each GPU holds half the weights.
Both GPUs participate in every forward pass. This reduces latency per request.

Change serve.sh:
```bash
# Add this flag — split model across 2 GPUs
--tensor-parallel-size 2
```

Rent 2x A100 on RunPod. Re-run the same benchmark matrix.

What to measure:
- Does TTFT drop at 1 user? (Each GPU does half the work per step)
- Does the saturation point shift? (2 GPUs = 2x KV cache = more concurrent capacity)
- What is the overhead? (GPU-to-GPU communication costs something)

Comparison table:

| Users | TTFT P50 (1 GPU) | TTFT P50 (2 GPU) | Delta |
|-------|-----------------|-----------------|-------|
| 1 | | | |
| 10 | | | |
| 25 | | | |
| 50 | | | |

---

### Phase 3c — Kubernetes Orchestration

Now the infrastructure layer. Take the Phase 3a single-GPU setup and deploy it
on Kubernetes with auto-scaling and monitoring.

**Why this is the last step — not the first:**
K8s adds operational complexity. If you do K8s before understanding the model's
behavior, you will not be able to tell whether a problem is the model, vLLM,
or K8s. Do K8s after you understand everything else.

**What gets built:**
- Dockerfile using `vllm/vllm-openai` as base image
- K8s manifests: Deployment, Service, ConfigMap, PVC, HPA
- Prometheus + Grafana monitoring stack
- KEDA for queue-depth-based auto-scaling
- Cold start timing: how long from HPA trigger to second pod serving traffic

**Local prep first (free — do before spending GPU credits):**
- Install Minikube
- Write and validate all manifests against a dummy container
- Build the Grafana dashboard using fake data
- Test Prometheus service discovery in Minikube

**Key thing to observe during the GPU session:**
The cold start problem. When HPA fires and a new pod starts, the model takes
2-3 minutes to load. During that time, latency does not improve — it may get
worse because requests queue. Record exactly:
- Time from HPA trigger to second pod `Running`
- Time from `Running` to second pod actually serving traffic
- TTFT during the cold start window vs after

Final comparison table (fill with Phase 1 baseline + Phase 3 results):

| Setup | TTFT P50 | TTFT P95 | Throughput at 50 users | Saturation Point |
|-------|----------|----------|----------------------|-----------------|
| Phase 1 — 1 GPU, no K8s | | | | |
| Phase 3a — 1 GPU, no K8s | | | | |
| Phase 3b — 2 GPU tensor parallel | | | | |
| Phase 3c — K8s 1 replica | | | | |
| Phase 3c — K8s 2 replicas | | | | |

---

## Difficulties to Expect

**Phase 3a — some results will surprise you.**
At low concurrency (5 users) numbers may be similar to 1 user. At 25+ users,
things change fast. The jump from 25 to 50 users often shows non-linear degradation.

**Phase 3b — GPU interconnect matters.**
RunPod's GPU interconnect may be NVLink or PCIe depending on the pod. NVLink
is significantly faster for tensor parallelism. Check which you are getting.
PCIe interconnect adds latency to tensor parallel communication.

**Phase 3c — K8s ops takes longer than expected.**
Budget extra time for Kubernetes. Service discovery, PVC mounting, readiness
probes, KEDA installation — each can have its own failure mode.
Do not start Phase 3c on RunPod until all manifests are validated in Minikube.

**Phase 3c — cold start is not a bug.**
The 2-3 minute delay before a new pod serves traffic is fundamental to LLM
inference. This is not something to fix. It is something to understand and
design around (pre-warming, over-provisioning).

---

## Exit Criteria — Phase 3 is Done When

**Phase 3a:**
- [ ] Benchmark table filled for all 5 concurrency levels
- [ ] Saturation point identified (user count where requests start queuing)
- [ ] KV cache % recorded at each concurrency level

**Phase 3b:**
- [ ] Same benchmark table run on 2-GPU setup
- [ ] TTFT comparison table filled (1 GPU vs 2 GPU)
- [ ] Documented: did tensor parallelism help at single user? At 50 users?

**Phase 3c:**
- [ ] Dockerfile built and pushed to DockerHub
- [ ] All K8s manifests validated in Minikube before GPU session
- [ ] Cold start timing recorded
- [ ] Auto-scaling observed working (HPA trigger → new pod → latency improves)
- [ ] Final comparison table filled (all setups side by side)

---

---

# Costs Across All Phases

| Phase | What | Estimated Cost |
|-------|------|---------------|
| Phase 1 | Single A100, ~2-3 hrs | ~$4-6 |
| Phase 1 | Network Volume (15GB, persistent) | ~$1-2/month |
| Phase 2 | Single A100, ~1-2 hrs | ~$2-4 |
| Phase 3a | Single A100, ~1-2 hrs | ~$2-4 |
| Phase 3b | 2x A100, ~1-2 hrs | ~$8-12 |
| Phase 3c | 2x A100 K8s, ~3-4 hrs | ~$20-30 |
| **Total** | | **~$40-55** |

**RunPod uses per-second billing. Stop the pod the moment you are done.**
Even 30 minutes of idle GPU time costs real money.

---

# What This Project Demonstrates

When this is complete, you will be able to explain from first principles:

1. Why TTFT is the metric that matters most for streaming LLM servers
2. Which vLLM parameters affect latency and which are marketing
3. Why long conversations get slower and what prefix caching does about it
4. At what load a single GPU breaks and what the numbers look like
5. What tensor parallelism actually buys vs what it costs
6. Why LLM inference scaling is fundamentally different from stateless API scaling
   (cold start is the answer — a new pod cannot serve for 2-3 minutes)

These are not theoretical answers. They are backed by your own measurements
on real hardware with real numbers.