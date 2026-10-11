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
# step 2 - Login to HuggingFace 
# HuggingFace requires authentication to download models.
# $HF_TOKEN reads the token from the environment variable set
# in RunPod settings — never hardcode the actual token here
# or it will be visible on GitHub.
# -------------------------------------------------------------
huggingface-cli login --token $HF_TOKEN

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