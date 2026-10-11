#!/bin/bash
# Shebang line — tells Linux to use bash to run this script.
# Must always be line 1 with nothing above it.

# =============================================================
# benchmark.sh
# Runs a Locust load test against the vLLM server.
#
# HOW TO USE:
#   bash scripts/benchmark.sh <users> [profile]
#
#   <users>   — number of concurrent users (required)
#   [profile] — prompt profile to use (optional, default: medium)
#               short  : ~50 token prompt,  100 max output tokens
#               medium : ~450 token prompt, 512 max output tokens
#               long   : ~800 token prompt, 1000 max output tokens
#
# EXAMPLES:
#   bash scripts/benchmark.sh 1              # 1 user, medium prompt
#   bash scripts/benchmark.sh 1 short        # 1 user, short prompt
#   bash scripts/benchmark.sh 10 long        # 10 users, long prompt
#   bash scripts/benchmark.sh 25 medium      # 25 users, medium prompt
#
# WHY PROFILES:
#   Different prompt lengths stress the GPU differently.
#   Short prompt = low prefill cost, TTFT dominated by decode.
#   Long prompt  = high prefill cost, TTFT dominated by prefill.
#   Comparing across profiles shows where your latency comes from.
#
# BEFORE RUNNING:
#   serve.sh must already be running in another terminal.
#   Results are saved to benchmarks/results/ as CSV files.
# =============================================================

# -------------------------------------------------------------
# Check 1 — Was an argument provided at all?
# $1 is the first argument passed to the script.
# -z means "is empty" — if nothing was typed after benchmark.sh
# show the usage message and exit with error code 1.
# -------------------------------------------------------------
if [ -z "$1" ]; then
    echo "Error: No user count provided."
    echo ""
    echo "Usage:"
    echo "  bash scripts/benchmark.sh 1"
    echo "  bash scripts/benchmark.sh 5"
    echo "  bash scripts/benchmark.sh 10"
    echo "  bash scripts/benchmark.sh 25"
    echo "  bash scripts/benchmark.sh 50"
    exit 1
fi


# -------------------------------------------------------------
# Check 2 — Is the argument actually a number?
# =~ is a regex match operator.
# ^[0-9]+$ means "only digits, nothing else".
# The ! at the start means "if it does NOT match".
# So if someone types "abc" or "10users" this catches it.
# exit 1 stops the script with an error.
# -------------------------------------------------------------
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: '$1' is not a valid number."
    echo "Please provide a number. Example: bash scripts/benchmark.sh 25"
    exit 1
fi


# -------------------------------------------------------------
# Check 3 — Is it one of our defined test levels?
# We only benchmark at 1, 5, 10, 25, 50 users — the five rows
# in our benchmark table. Any other number is outside the
# defined test matrix.
# This is a WARNING not a hard error — during Phase 1 Step 4
# tuning you might genuinely want to test at an unusual level
# like 15 or 30 to understand behaviour between standard levels.
# Press Enter to continue anyway or Ctrl+C to cancel.
# -------------------------------------------------------------
if [[ "$1" != "1" && "$1" != "5" && "$1" != "10" && "$1" != "25" && "$1" != "50" ]]; then
    echo "Warning: $1 is not a standard test level."
    echo "Standard benchmark levels are: 1, 5, 10, 25, 50"
    echo ""
    echo "Press Enter to continue anyway or Ctrl+C to cancel."
    read
fi


# -------------------------------------------------------------
# Check 4 — Validate the prompt profile argument ($2).
# $2 is optional — if not provided, default to "medium".
# ${2:-medium} means: use $2 if it exists, otherwise use "medium".
# If an invalid profile name is given, show an error and exit.
# -------------------------------------------------------------
PROMPT_PROFILE=${2:-medium}

if [[ "$PROMPT_PROFILE" != "short" && "$PROMPT_PROFILE" != "medium" && "$PROMPT_PROFILE" != "long" ]]; then
    echo "Error: '$PROMPT_PROFILE' is not a valid profile."
    echo "Valid profiles: short, medium, long"
    echo "Example: bash scripts/benchmark.sh 10 long"
    exit 1
fi

# -------------------------------------------------------------
# Export the profile as an environment variable.
# export makes the variable available to child processes —
# in this case, Locust and the locustfile.py it runs.
# locustfile.py reads PROMPT_PROFILE via os.environ.get()
# to decide which prompt and max_tokens to use.
# -------------------------------------------------------------
export PROMPT_PROFILE


# -------------------------------------------------------------
# Run the Locust load test.
#
# -f benchmarks/locustfile.py
#   The file that defines what one user does.
#
# --host http://localhost:8000
#   The vLLM server address. All requests go here.
#
# --users $1
#   Number of concurrent users — taken from your argument.
#
# --spawn-rate 1
#   Adds 1 new user per second until all users are active.
#   Realistic ramp-up — prevents all users hitting at once.
#
# --run-time 60s
#   Runs the test for 60 seconds once all users are active.
#   Enough time to collect stable numbers.
#
# --headless
#   No browser UI — results print directly to terminal.
#
# --csv benchmarks/results/benchmark_$1users_$PROMPT_PROFILE
#   File name includes both user count AND profile so runs
#   do not overwrite each other.
#   Example: benchmark_10users_long_stats.csv
# -------------------------------------------------------------
echo "Starting load test: $1 concurrent users, '$PROMPT_PROFILE' prompt profile"
echo "Results will be saved to benchmarks/results/benchmark_$1users_${PROMPT_PROFILE}*.csv"
echo ""

locust -f benchmarks/locustfile.py \
  --host http://localhost:8000 \
  --users $1 \
  --spawn-rate 1 \
  --run-time 60s \
  --headless \
  --csv benchmarks/results/benchmark_$1users_$PROMPT_PROFILE

