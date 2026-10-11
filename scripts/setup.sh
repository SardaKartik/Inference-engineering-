#!/bin/bash
# This is called a shebang line — it must always be the first
# line of every shell script. It tells Linux which program to
# use to run this file. Without it Linux does not know whether
# to use bash, sh, zsh or something else and the script may
# fail or behave unexpectedly depending on the system default.

# =============================================================
# setup.sh
# Run this ONCE at the start of your first RunPod session.
# From the second session onwards skip this — the model is
# already downloaded to the Network Volume.
# =============================================================

# -------------------------------------------------------------
# step 1 - Install dependencies 
# vllm    : the serving engine that loads the model and exposes
#           it as an API
# autoawq : required to load AWQ quantized models in vLLM.
#           The model name ends with -AWQ which means it is
#           compressed using the AWQ format. Without this
#           package vLLM cannot read the model weights.
# locust  : load testing tool that simulates concurrent users
#           hitting the vLLM API. Installed here so everything
#           is ready in one setup step.
# -------------------------------------------------------------
pip install vllm autoawq locust

# -------------------------------------------------------------
# Step 2 — Check HF_TOKEN then login to HuggingFace
# HuggingFace requires authentication to download models.
# $HF_TOKEN reads the token from the environment variable set
# in RunPod settings — never hardcode the actual token here
# or it will be visible on GitHub.
#
# -z checks if the variable is empty. If HF_TOKEN was not set
# in RunPod environment settings, we stop here with a clear
# error. Without this check, huggingface-cli gets an empty
# token, fails with a confusing auth error, then the download
# fails too — making it hard to know the root cause.
# -------------------------------------------------------------
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Set it in RunPod → Settings → Environment Variables before running this script."
    exit 1
fi

huggingface-cli login --token "$HF_TOKEN"

# -------------------------------------------------------------
# Step 3 — Download the model to the Network Volume
# /workspace/ is where RunPod mounts the Network Volume.
# Saving here means the model persists across sessions —
# next session attach the same volume and skip this script.
# The \ at the end of the line means the command continues
# on the next line — just for readability.
# -------------------------------------------------------------
huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct-AWQ \
  --local-dir /workspace/models/qwen2.5-coder