# Production LLM Inference Stack — Qwen2.5-Coder

A two-phase project to build and operate a production-grade LLM inference stack
from scratch. The goal is to understand inference engineering concepts through
real observation and measurement — not theory.

---

## What This Project Is

Most people use LLMs through an API without understanding what happens underneath.
This project builds that understanding by operating every layer of the stack directly:

- Serving a 14B parameter model on a real GPU
- Observing how memory, batching, and caching behave under load
- Benchmarking latency and throughput at different concurrency levels
- Containerizing the server and deploying it on Kubernetes
- Adding auto-scaling and production observability

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | Qwen2.5-Coder-14B-Instruct (AWQ quantized) |
| Serving engine | vLLM |
| Load testing | Locust |
| Infrastructure | RunPod (A100 80GB) |
| Orchestration | Kubernetes (Phase 2) |
| Monitoring | Prometheus + Grafana (Phase 2) |

---

## Project Structure

```
inference_engineering/
│
├── scripts/                  # Shell scripts for RunPod sessions
│   ├── setup.sh              # Run once — install dependencies + download model
│   ├── serve.sh              # Run every session — start vLLM server
│   ├── monitor.sh            # Watch GPU and vLLM metrics live
│   └── benchmark.sh          # Run Locust load test with user count argument
│
├── benchmarks/               # Load testing
│   ├── locustfile.py         # Defines virtual user behaviour
│   └── results/              # CSV output from benchmark runs (gitignored)
│
├── docs/                     # Reference documentation
│   ├── phase1.md             # Phase 1 working guide
│   ├── phase2.md             # Phase 2 working guide
│   └── benchmarks.md         # Metrics explained — what we measure and why
│
└── README.md
```

---

## Phase 1 — Serving + Benchmarking on Single GPU

**Goal:** Get vLLM serving Qwen2.5-Coder-14B on a real GPU, understand its
internals through live observation, and collect real benchmark numbers.

**Infrastructure:** Single A100 80GB on RunPod (~$8-12 for a full session)

### How to run a Phase 1 session

**First session only — install and download:**
```bash
bash scripts/setup.sh
```

**Every session — start the server:**
```bash
bash scripts/serve.sh
```

**Open a second terminal — watch GPU:**
```bash
bash scripts/monitor.sh gpu
```

**Open a third terminal — watch vLLM internals:**
```bash
bash scripts/monitor.sh vllm
```

**Open a fourth terminal — run load test:**
```bash
bash scripts/benchmark.sh 1    # 1 concurrent user
bash scripts/benchmark.sh 5    # 5 concurrent users
bash scripts/benchmark.sh 10   # 10 concurrent users
bash scripts/benchmark.sh 25   # 25 concurrent users
bash scripts/benchmark.sh 50   # 50 concurrent users
```

Results are saved automatically to `benchmarks/results/` as CSV files.

### Metrics collected

| Metric | What it tells you |
|--------|------------------|
| TTFT P50 / P95 | How long before the first token streams back |
| TPOT | How fast tokens generate after the first one |
| Avg / P95 Latency | Typical and worst-case total response time |
| Tokens/sec | GPU generation throughput |
| Requests/sec | Request completion rate |
| GPU Util % | Whether the GPU is the bottleneck |
| KV Cache % | Memory pressure — leading indicator of degradation |

See `docs/benchmarks.md` for full explanation of every metric.

---

## Phase 2 — Kubernetes Orchestration + Auto-scaling

**Goal:** Take the Phase 1 setup and deploy it on Kubernetes with auto-scaling,
observability, and production-grade configuration.

**Infrastructure:** 2x A100 80GB on RunPod Kubernetes (~$20-30 for a session)

**What gets built:**
- Dockerfile using official vLLM base image with proper health checks
- Kubernetes manifests — Deployment, Service, ConfigMap, PVC, HPA
- Prometheus + Grafana monitoring stack
- HPA auto-scaling triggered by GPU utilization or request queue depth
- Benchmark comparison: Phase 1 single GPU vs Phase 2 K8s 1-replica vs 2-replica

See `docs/phase2.md` for the full working guide.

---

## Key Concepts

**PagedAttention** — vLLM's memory management system. Instead of reserving a
fixed block of memory per request, it allocates memory in small pages as needed.
This is what allows efficient concurrent request handling.

**KV Cache** — Key-Value cache that stores intermediate computations during
generation. Its size directly controls how many concurrent requests can be served.
When it fills, new requests queue.

**Continuous Batching** — vLLM processes multiple requests simultaneously without
waiting for one to finish before starting another. This is what keeps GPU
utilization high under concurrent load.

**Cold Start** — When a new Kubernetes pod starts, it must load the model
(2-3 minutes) before serving any traffic. This is why LLM inference scaling
behaves differently from stateless API scaling.

---

## Cost Estimate

| Phase | What | Est. Cost |
|-------|------|----------|
| Phase 1 | Single A100, ~4-5 hrs | ~$8-12 |
| Phase 1 | Network Volume (persistent storage) | ~$1-2/month |
| Phase 2 — local prep | Minikube on laptop | $0 |
| Phase 2 — GPU session | 2x A100, ~3-4 hrs | ~$20-30 |

**Total across both phases: ~$30-45**

RunPod uses per-second billing — you only pay while the pod is running.
Stop the pod immediately when done to avoid unnecessary charges.

---

## Environment Setup

Copy `.env.example` to `.env` and fill in your HuggingFace token:

```bash
cp .env.example .env
```

Get your token from: HuggingFace → Settings → Access Tokens → Read-only
