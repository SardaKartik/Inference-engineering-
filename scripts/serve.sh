#!/bin/bash
# Shebang line — tells Linux to use bash to run this script.
# Must always be line 1 with nothing above it.

# =============================================================
# serve.sh
# Run this every session after setup.sh has been run once.
# Starts the vLLM server and loads the model into GPU memory.
# Once running, the API is available at http://localhost:8000
# =============================================================

# -------------------------------------------------------------
# Start the vLLM server
# 
# /workspace/models/qwen2.5-coder
#   Path to the model weights on the Network Volume.
#   This is where setup.sh downloaded the model
#
# --quantization awq
#   Tells vLLM the model is AWQ quantized. Required because
#   the model was compressed using AWQ format. Works together
#   with the autoawq package installed in setup.sh.
#
# --gpu-memory-utilization 0.85
#   Uses 85% of the A100 80GB VRAM for the KV cache.
#   The remaining 15% stays as a safety buffer.
#   This is the main tuning knob in Phase 1 Step 4.
#
# --max-model-len 8192
#   Maximum tokens per request (prompt + output combined).
#   Shorter = less KV cache per request = more concurrency.
#   Also a tuning knob in Phase 1 Step 4.
#
# --max-num-seqs 32
#   Maximum requests processed at the same time.
#   Too high = KV cache pressure, latency spikes.
#   Too low = GPU sits idle. Tune this in Phase 1 Step 4.
#
# --served-model-name qwen2.5-coder
#   The name the API registers the model under.
#   Without this, vLLM uses the folder path as the model name.
#   locustfile.py sends {"model": "qwen2.5-coder"} in every
#   request — if the names do not match, vLLM returns
#   model-not-found on every request and all benchmarks fail.
#   This flag makes the name match exactly.
#
# --port 8000
#   Port the API listens on. All requests go to this port.
# -------------------------------------------------------------
vllm serve /workspace/models/qwen2.5-coder \
  --quantization awq \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --served-model-name qwen2.5-coder \
  --port 8000