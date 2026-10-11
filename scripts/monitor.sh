#!/bin/bash
# Shebang line — tells Linux to use bash to run this script.
# Must always be line 1 with nothing above it.

# =============================================================
# monitor.sh
# Run this every session alongside serve.sh.
# Gives you a live window into what the GPU and vLLM are doing
# while requests are being served.
#
# WHY WE NEED THIS:
# When vLLM is running and serving requests, you cannot see
# what is happening inside just by looking at the terminal.
# This script gives you two live views — one into the GPU
# hardware and one into vLLM internals — so you can watch
# memory filling, requests batching, and the system reaching
# its limits in real time. Without this you are guessing.
#
# HOW TO USE:
# Open two separate terminals on RunPod alongside serve.sh.
# Terminal 2 → bash scripts/monitor.sh gpu
# Terminal 3 → bash scripts/monitor.sh vllm
# =============================================================


# -------------------------------------------------------------
# Read the first argument passed to the script.
# $1 means "whatever word you typed after monitor.sh"
# Example: bash monitor.sh gpu  → $1 is "gpu"
#          bash monitor.sh vllm → $1 is "vllm"
# This lets one script handle both modes instead of having
# two separate script files.
# -------------------------------------------------------------

if [ "$1" == "gpu" ]; then
    # ---------------------------------------------------------
    # GPU MODE
    #
    # WHY nvidia-smi:
    # nvidia-smi is a tool that comes built into every NVIDIA
    # GPU driver. It reads hardware-level stats directly from
    # the GPU. vLLM does not report these — only the GPU driver
    # knows the real VRAM usage, temperature, and power draw.
    # This is the ground truth view of what the hardware is
    # doing regardless of what software is running on top.
    #
    # WHAT TO WATCH:
    # VRAM used / total  — watch this climb as requests come in
    #                      and drop when requests finish. This
    #                      is the KV cache filling in real time.
    # GPU utilization %  — how hard the GPU is computing.
    #                      Low % under load means the bottleneck
    #                      is not the GPU (could be CPU or IO).
    #                      High % means the GPU is saturated.
    # Temperature        — should stay under 85C on A100.
    # Power draw         — tells you how much compute is active.
    #
    # watch -n1 means refresh every 1 second so you see
    # changes as they happen.
    # ---------------------------------------------------------
    watch -n1 nvidia-smi

elif [ "$1" == "vllm" ]; then
    # ---------------------------------------------------------
    # vLLM MODE
    #
    # WHY /metrics endpoint:
    # nvidia-smi shows the GPU hardware view but does not know
    # anything about vLLM internals — how many requests are
    # running, how full the KV cache is, how many are waiting.
    # vLLM exposes this internal state on its /metrics endpoint
    # in Prometheus format. We fetch it every 2 seconds and
    # filter to only the three numbers that matter most.
    #
    # curl -s localhost:8000/metrics
    #   Fetches the full metrics page from vLLM silently.
    #   The -s flag means silent — no download progress shown.
    #
    # | grep -E "cache_usage|requests_running|requests_waiting"
    #   The | symbol pipes the output into grep which filters
    #   it down to only lines containing these three keywords.
    #   Without grep you would see hundreds of metrics lines.
    #
    # WHAT TO WATCH:
    # gpu_cache_usage_perc  — % of KV cache currently occupied.
    #                         This is the most important metric.
    #                         When it approaches 85-90% vLLM
    #                         starts queuing new requests instead
    #                         of running them immediately.
    #
    # num_requests_running  — requests being processed right now.
    #                         This shows continuous batching in
    #                         action — multiple requests handled
    #                         at the same time by one GPU.
    #
    # num_requests_waiting  — requests sitting in queue waiting
    #                         for KV cache space to free up.
    #                         When this goes above 0 the system
    #                         is under pressure. When it spikes
    #                         you have found your saturation point
    #                         — the point where more users means
    #                         worse latency for everyone.
    #
    # watch -n2 means refresh every 2 seconds.
    # ---------------------------------------------------------
    watch -n2 'curl -s localhost:8000/metrics | grep -E "cache_usage|requests_running|requests_waiting"'

else
    # ---------------------------------------------------------
    # If no argument or wrong argument is given, show usage
    # so the user knows exactly how to run this script.
    # ---------------------------------------------------------
    echo "Usage:"
    echo "  bash scripts/monitor.sh gpu   → watch GPU stats"
    echo "  bash scripts/monitor.sh vllm  → watch vLLM metrics"
fi