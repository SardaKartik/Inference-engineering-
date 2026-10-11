# Benchmarks — What We Measure and Why

This document explains every metric we collect during Phase 1 load testing,
where it comes from, and what it tells us about the system.

---

## Why We Benchmark

Running vLLM and getting responses is not enough. We need numbers to answer
real questions:

- At what point does the system break under load?
- Is the GPU actually being used efficiently or is something else the bottleneck?
- Which vLLM parameter change actually improved performance and by how much?
- What does horizontal scaling in Phase 2 actually buy us compared to Phase 1?

Without benchmarks these questions have no answers. The numbers make the
before/after comparisons meaningful.

---

## The Traffic Profile

Every single benchmark run uses this exact profile. Never change it mid-project
or comparisons between runs become meaningless.

| Setting | Value | Why |
|---------|-------|-----|
| System prompt | ~300 tokens | Realistic coding assistant context |
| User message | ~150 tokens | Realistic code question |
| max_tokens | 512 | Realistic code response length |
| Streaming | Enabled | Required to measure TTFT accurately |

---

## Metrics We Collect

### 1. TTFT — Time To First Token

**What it is:**
Time from when the request is sent to when the very first token streams back.
Measured in milliseconds.

**Why it matters:**
This is what the user actually feels as response speed. For a streaming LLM,
the model starts sending tokens one by one — TTFT is how long the user waits
before anything appears on screen. A model with bad TTFT feels broken even if
the total response time is acceptable.

For a coding assistant this is the single most important latency metric.

**Where it comes from:**
Custom measurement inside the Locust script. We timestamp the moment the
request is sent and the moment the first streaming chunk arrives. The
difference is TTFT. Locust does not measure this by default — we add it
ourselves.

**We collect:** P50 and P95

---

### 2. TPOT — Time Per Output Token

**What it is:**
After the first token arrives (that is TTFT), TPOT measures how long it takes
to generate each subsequent token. It is the average time between consecutive
tokens during the generation phase. Measured in milliseconds per token.

**How it is calculated:**
```
TPOT = (Total response time - TTFT) / (Total output tokens - 1)
```
You take the total time after the first token and divide it by the number of
remaining tokens. This gives the average generation speed per token.

**Why it matters:**
TTFT and TPOT together describe the complete streaming experience:
- TTFT = how long the user waits before anything appears on screen
- TPOT = how fast the text streams after that

A model can have great TTFT but if TPOT is high the response trickles out
slowly and feels sluggish. For a coding assistant where the model generates
hundreds of tokens, bad TPOT means the user sits watching text appear one
word every second — frustrating even if the first token was fast.

Under high concurrency TPOT gets worse because the GPU is sharing compute
across many requests simultaneously. Watching TPOT degrade as you increase
concurrent users shows you directly how continuous batching affects generation
speed per user.

**Where it comes from:**
Custom calculation inside the Locust script. We already measure total response
time, TTFT, and token count — TPOT is derived from those three values.
No extra data collection needed.

**We collect:** Average TPOT across all requests in the test run.

---

### 3. Average Latency (End-to-End)

**What it is:**
Average total time from request sent to last token received, across all
requests in the test run. Measured in milliseconds.

**Why it matters:**
Gives you the typical experience across all users. Useful as a general health
indicator but averages hide outliers — a few very slow requests get buried
behind many fast ones. Always read alongside P95.

**Where it comes from:**
Locust built-in measurement. Reported automatically at the end of every run.

---

### 3. P95 Latency

**What it is:**
95% of all requests in the test completed within this time. Only the slowest
5% took longer. Measured in milliseconds.

**Why it matters:**
This is the tail latency — what your worst-experience users see. In production
systems P95 matters more than average because it tells you how bad things get
for real users under real load. A system with great average latency but terrible
P95 is unreliable.

When you see P95 spike while average stays flat, requests are starting to queue
— some users wait much longer than others.

**Where it comes from:**
Locust built-in measurement. Reported automatically at the end of every run.

---

### 4. Throughput — Tokens Per Second

**What it is:**
Total number of tokens generated per second across all concurrent users
combined during the test. Measured in tokens/sec.

**Why it matters:**
This tells you how efficiently the GPU is doing useful work. A higher
tokens/sec means more output is being generated in the same time — the GPU
is being used well. This is the metric that shows whether continuous batching
is actually working. Under low concurrency tokens/sec is low because the GPU
sits idle between requests. Under optimal concurrency it peaks.

**Where it comes from:**
Calculated inside the Locust script. We count the tokens in each response
and divide by the time taken.

---

### 5. Requests Per Second

**What it is:**
Number of complete requests vLLM finishes per second during the test.

**Why it matters:**
Different from tokens/sec — a short 10-token response and a long 500-token
response both count as one request. Together with tokens/sec it gives the
full throughput picture. If requests/sec is high but tokens/sec is low, users
are getting short responses. If both are high, the system is genuinely
performing well under load.

**Where it comes from:**
Locust built-in measurement. Reported automatically at the end of every run.

---

### 6. GPU Utilization %

**What it is:**
How hard the GPU is computing during the test. Shown as a percentage of
maximum compute capacity. Measured via nvidia-smi.

**Why it matters:**
GPU utilization tells you where the bottleneck actually is.

- Low GPU util + bad latency = the problem is not the GPU. Something else
  is the bottleneck — could be CPU, memory bandwidth, or network.
- High GPU util + bad latency = the GPU is genuinely maxed out. You need
  either better parameters or more GPUs.
- Low GPU util + good latency = the GPU has headroom. You can handle more
  concurrent users before things break.

**Where it comes from:**
Read manually from nvidia-smi during the test using monitor.sh gpu.
Note the approximate GPU util % while the test is running and record it.

---

### 7. KV Cache Utilization %

**What it is:**
What percentage of the KV cache memory is occupied at peak during the test.
Shown as a value between 0 and 1 (0.85 means 85% full).

**Why it matters:**
This is the leading indicator — it tells you the system is about to degrade
before you see it in latency numbers. When KV cache utilization approaches
85-90%, vLLM starts queuing new requests instead of running them immediately.
Latency spikes shortly after. If you watch KV cache utilization you can
predict problems before they fully appear.

This is also the metric most directly controlled by your vLLM parameters:
- Increase gpu_memory_utilization → more KV cache space → lower utilization %
- Decrease max_model_len → less cache per request → lower utilization %
- Decrease max_num_seqs → fewer concurrent requests → lower utilization %

**Where it comes from:**
Read manually from vLLM /metrics endpoint during the test using
monitor.sh vllm. Look for the gpu_cache_usage_perc value.

---

## Full Benchmark Table

Run this for each concurrency level. Fill in numbers during the RunPod session.

| Concurrent Users | TTFT P50 (ms) | TTFT P95 (ms) | TPOT avg (ms) | Avg Latency (ms) | P95 Latency (ms) | Tokens/sec | Requests/sec | GPU Util % | KV Cache % |
|-----------------|--------------|--------------|--------------|-----------------|-----------------|-----------|-------------|-----------|-----------|
| 1 | | | | | | | | | |
| 5 | | | | | | | | | |
| 10 | | | | | | | | | |
| 25 | | | | | | | | | |
| 50 | | | | | | | | | |

---

## Where Each Metric Comes From

| Metric | Source | How to collect |
|--------|--------|---------------|
| TTFT P50 / P95 | Locust custom | Automatic — streaming timestamp in locustfile.py |
| TPOT avg | Locust custom | Automatic — calculated from total time, TTFT, token count |
| Avg Latency | Locust built-in | Automatic — printed at end of run |
| P95 Latency | Locust built-in | Automatic — printed at end of run |
| Tokens/sec | Locust custom | Automatic — calculated in locustfile.py |
| Requests/sec | Locust built-in | Automatic — printed at end of run |
| GPU Util % | nvidia-smi | Manual — read from monitor.sh gpu during test |
| KV Cache % | vLLM /metrics | Manual — read from monitor.sh vllm during test |

---

## Tuning Round — After Baseline

Once the baseline table above is filled, tune one vLLM parameter at a time
and re-run the 25-user test. Record the before and after numbers.

| Parameter Changed | Value Before | Value After | TTFT P95 Before | TTFT P95 After | TPOT Before | TPOT After | Tokens/sec Before | Tokens/sec After |
|------------------|-------------|-------------|----------------|----------------|------------|-----------|------------------|-----------------|
| gpu_memory_utilization | 0.85 | | | | | | | |
| max_num_seqs | 32 | | | | | | | |
| max_model_len | 8192 | | | | | | | |