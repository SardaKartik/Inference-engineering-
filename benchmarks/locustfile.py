import os
import time
import json
import logging
from locust import HttpUser, task, between


# =============================================================
# locustfile.py
# Defines what one virtual user does when load testing vLLM.
# Locust clones this user N times to simulate concurrent load.
#
# METRICS COLLECTED:
#   TTFT         — Time To First Token (custom, via streaming)
#   TPOT         — Time Per Output Token (custom, calculated)
#   E2E Latency  — Total request time (Locust built-in)
#   Tokens/sec   — Generation throughput (custom, calculated)
#   Requests/sec — Request completion rate (Locust built-in)
# =============================================================


# =============================================================
# Logging setup
# During debugging use level=logging.DEBUG to see everything.
# During actual benchmark runs change to level=logging.WARNING
# so only errors and warnings appear — clean output.
# Format: timestamp [LEVEL] message
# =============================================================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================
# PROMPT PROFILES
# Three profiles with different input lengths to test how
# prefill cost and TTFT change as prompt size grows.
#
# benchmark.sh passes PROMPT_PROFILE as an environment variable.
# os.environ.get("PROMPT_PROFILE", "medium") reads it here.
# If not set, defaults to medium so the script works when
# run manually without benchmark.sh.
#
# short  : ~50 token prompt  + 100 max output tokens
#          Tiny prefill — isolates decode speed.
#          TTFT here is mostly GPU warmup, not prompt processing.
#
# medium : ~450 token prompt + 512 max output tokens
#          The main baseline. Use for all standard benchmark runs.
#          Realistic coding assistant request.
#
# long   : ~800 token prompt + 1000 max output tokens
#          Heavy prefill — TTFT will visibly climb vs medium.
#          Shows exactly how much prefill cost grows with length.
#          Fills KV cache faster — useful for multi-user runs.
# =============================================================

PROFILES = {

    "short": {
        "system": (
            "You are a Python developer. Answer concisely."
        ),
        "user": (
            "Write a Python function to reverse a string."
        ),
        "max_tokens": 100,
    },

    "medium": {
        "system": (
            "You are an expert Python developer. "
            "Answer clearly and provide complete, working code examples. "
            "Include comments explaining what the code does."
        ),
        "user": (
            "Write a Python function that reads a CSV file, "
            "filters rows where the age column is greater than 25, "
            "and returns the result as a list of dictionaries. "
            "Include error handling for missing files and invalid data."
        ),
        "max_tokens": 512,
    },

    "long": {
        "system": (
            "You are a senior Python engineer with 10 years of experience "
            "building production data pipelines and APIs. "
            "You write clean, well-tested, production-ready code. "
            "You always include type hints, docstrings, error handling, logging, "
            "and explain design decisions. "
            "You follow PEP8 and modern Python best practices. "
            "When asked to write code, provide the complete implementation with no placeholders."
        ),
        "user": (
            "Write a complete Python module for processing customer transaction data. "
            "The module should: "
            "1. Read transaction records from a CSV file with columns: transaction_id, customer_id, "
            "amount, currency, timestamp, status, merchant_name, category. "
            "2. Validate each row — check for missing fields, invalid amounts (negative or zero), "
            "unknown currencies (only USD, EUR, GBP, INR are valid), and malformed timestamps. "
            "3. Filter transactions to only include completed status with amount greater than 100. "
            "4. Group transactions by customer_id and calculate total spend per customer. "
            "5. Convert all amounts to USD using hardcoded rates (EUR=1.08, GBP=1.27, INR=0.012). "
            "6. Return a sorted list of dictionaries with customer_id, transaction_count, total_usd. "
            "Include full error handling, logging at appropriate levels, and type hints throughout."
        ),
        "max_tokens": 1000,
    },

}


# -------------------------------------------------------------
# Read which profile benchmark.sh selected via environment variable.
# Default to medium if running locustfile.py directly without benchmark.sh.
# -------------------------------------------------------------
profile_name = os.environ.get("PROMPT_PROFILE", "medium")

# Guard: if PROMPT_PROFILE is set to an invalid value outside of benchmark.sh
# (e.g. a typo when running Locust directly), fail here with a clear message
# rather than a confusing KeyError traceback.
if profile_name not in PROFILES:
    raise ValueError(
        f"Invalid PROMPT_PROFILE='{profile_name}'. "
        f"Valid values are: {list(PROFILES.keys())}"
    )

profile = PROFILES[profile_name]

SYSTEM_PROMPT = profile["system"]
USER_MESSAGE  = profile["user"]
MAX_TOKENS    = profile["max_tokens"]

MODEL_NAME = "qwen2.5-coder"

# Confirm which profile loaded before any requests fire.
# If you see the wrong profile here, check the PROMPT_PROFILE env variable.
logger.info(f"Prompt profile loaded: '{profile_name}' — max_tokens: {MAX_TOKENS}")


# -------------------------------------------------------------
# VLLMUser — defines what one concurrent user does.
# Extends HttpUser which gives access to self.client for
# making HTTP requests to the vLLM server.
# -------------------------------------------------------------
class VLLMUser(HttpUser):

    # ---------------------------------------------------------
    # wait_time — how long the user waits between requests.
    # between(1, 3) means a random wait of 1 to 3 seconds.
    # This simulates realistic user behaviour — people do not
    # fire requests with zero gap between them.
    # Without this every user would hammer the server as fast
    # as possible which is unrealistic and unfair to measure.
    # ---------------------------------------------------------
    wait_time = between(1, 3)

    @task
    def send_request(self):

        # -----------------------------------------------------
        # Log request start — if you never see this line in
        # the output, the problem is before the request fires.
        # -----------------------------------------------------
        logger.debug("Request started — sending to /v1/chat/completions")

        # -----------------------------------------------------
        # Build the request payload.
        # OpenAI-compatible format that vLLM accepts.
        # stream=True enables token-by-token streaming which is
        # required to measure TTFT accurately. Without streaming
        # vLLM waits until the full response is ready before
        # sending anything back.
        # -----------------------------------------------------
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_MESSAGE}
            ],
            "max_tokens": MAX_TOKENS,
            "stream": True
        }

        # -----------------------------------------------------
        # Record the exact time the request is sent.
        # This is the start point for TTFT measurement.
        # -----------------------------------------------------
        start_time = time.time()

        # -----------------------------------------------------
        # Send the request to vLLM with stream=True.
        # stream=True means the response body is not downloaded
        # all at once — we read it chunk by chunk as tokens arrive.
        # catch_response=True lets us manually mark the request
        # as success or failure based on what we see in the stream.
        # -----------------------------------------------------
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            stream=True,
            catch_response=True
        ) as response:

            # -------------------------------------------------
            # Tracking variables reset for each request.
            # -------------------------------------------------
            ttft             = None   # time to first token
            total_tokens     = 0      # count of tokens received
            first_chunk      = True   # True until first token arrives
            first_line_logged = False  # log the first raw line once only

            # -------------------------------------------------
            # Read the streaming response line by line.
            # Each line from vLLM looks like:
            #   data: {"choices": [{"delta": {"content": "hi"}}]}
            # The last line is always:
            #   data: [DONE]
            # -------------------------------------------------
            for line in response.iter_lines():

                # Decode bytes to string
                if isinstance(line, bytes):
                    line = line.decode("utf-8")

                # Skip blank lines — vLLM sends them between chunks
                if not line:
                    continue

                # -------------------------------------------------
                # Log the first raw line to verify format.
                # If parsing breaks, this shows exactly what vLLM sent
                # so you are not guessing where the problem is.
                # -------------------------------------------------
                if not first_line_logged:
                    logger.debug(f"First raw line from vLLM: {line}")
                    first_line_logged = True

                # Strip "data: " prefix — every SSE line starts with it
                if line.startswith("data: "):
                    line = line[6:]

                # [DONE] marks end of stream
                if line == "[DONE]":
                    logger.debug("Received [DONE] — stream complete")
                    break

                # ---------------------------------------------
                # Parse JSON chunk and extract token content.
                # ---------------------------------------------
                try:
                    chunk   = json.loads(line)
                    content = chunk["choices"][0]["delta"].get("content", "")

                    if content:
                        total_tokens += 1

                        # -------------------------------------
                        # TTFT — capture on the very first token.
                        # Time elapsed from request sent to first
                        # content token received. This is TTFT.
                        # -------------------------------------
                        if first_chunk:
                            ttft        = time.time() - start_time
                            first_chunk = False
                            logger.debug(f"TTFT captured: {ttft:.3f}s — first token: '{content}'")

                except json.JSONDecodeError:
                    # Log the raw line so you know what broke the parser.
                    logger.error(f"JSON parse failed. Raw line: '{line}'")
                    continue

            # -------------------------------------------------
            # Stream finished. Calculate final metrics.
            # -------------------------------------------------
            total_time = time.time() - start_time

            if total_tokens == 0:
                logger.warning("Zero tokens received — check vLLM response format")

            # -------------------------------------------------
            # TPOT — Time Per Output Token
            # Time spent after first token divided by remaining tokens.
            # Formula: (total_time - ttft) / (total_tokens - 1)
            # Guard: ttft is not None (first token was received)
            #        and total_tokens > 1 (avoid division by zero).
            # -------------------------------------------------
            tpot = (total_time - ttft) / (total_tokens - 1) if (ttft is not None and total_tokens > 1) else 0

            # Tokens generated per second — raw GPU throughput.
            tokens_per_second = total_tokens / total_time if total_time > 0 else 0

            # -------------------------------------------------
            # ttft is None when no content tokens were received.
            # Guard the format string — :.3f on None crashes.
            # -------------------------------------------------
            ttft_display = f"{ttft:.3f}s" if ttft is not None else "None"
            logger.debug(
                f"Request complete — "
                f"profile: {profile_name}, "
                f"tokens: {total_tokens}, "
                f"TTFT: {ttft_display}, "
                f"TPOT: {tpot:.4f}s, "
                f"tokens/sec: {tokens_per_second:.1f}, "
                f"total_time: {total_time:.3f}s"
            )

            # -------------------------------------------------
            # Report TTFT as a custom Locust metric.
            # response_time must be in milliseconds.
            # Use "is not None" not "if ttft" — ttft could be
            # 0.0 on a very fast response which is falsy but valid.
            # -------------------------------------------------
            if ttft is not None:
                self.environment.events.request.fire(
                    request_type="TTFT",
                    name="time_to_first_token",
                    response_time=ttft * 1000,
                    response_length=0,
                    exception=None
                )

            # Report TPOT as a custom Locust metric.
            if tpot:
                self.environment.events.request.fire(
                    request_type="TPOT",
                    name="time_per_output_token",
                    response_time=tpot * 1000,
                    response_length=0,
                    exception=None
                )

            # -------------------------------------------------
            # Mark success or failure based on HTTP status code.
            # catch_response=True means Locust waits for us to
            # decide — we must call either success() or failure().
            # Calling success() unconditionally hides real errors:
            # a 503 (server overloaded) would show as 0% error rate.
            # -------------------------------------------------
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")