import time
import json
import logging
from locust import HttpUser, task, between

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
# locustfile.py
# Defines what one virtual user does when load testing vLLM.
# Locust clones this user N times to simulate concurrent load.
#
# METRICS COLLECTED:
#   TTFT  — Time To First Token (custom, via streaming)
#   TPOT  — Time Per Output Token (custom, calculated)
#   E2E Latency — Total request time (Locust built-in)
#   Tokens/sec  — Generation throughput (custom, calculated)
#   Requests/sec — Locust built-in
# =============================================================


# -------------------------------------------------------------
# TRAFFIC PROFILE — Fixed prompt used for every request.
# Never change these mid-project or benchmark comparisons
# between runs become meaningless. Every user sends this exact
# prompt every time so all results are comparable.
#
# Approximate token counts:
#   SYSTEM_PROMPT  ~ 300 tokens
#   USER_MESSAGE   ~ 150 tokens
#   max_tokens     = 512 (output limit)
# -------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert Python developer.
Answer clearly and provide complete, working code examples.
Include comments explaining what the code does."""

USER_MESSAGE = """Write a Python function that reads a CSV file,
filters rows where the age column is greater than 25,
and returns the result as a list of dictionaries.
Include error handling for missing files and invalid data."""

MODEL_NAME = "qwen2.5-coder"


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
        # This is the OpenAI-compatible format that vLLM
        # accepts. stream=True enables token-by-token streaming
        # which is required to measure TTFT accurately.
        # Without streaming vLLM waits until the full response
        # is ready before sending anything back.
        # -----------------------------------------------------
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_MESSAGE}
            ],
            "max_tokens": 512,
            "stream": True
        }

        # -----------------------------------------------------
        # Record the exact time the request is sent.
        # This is the start point for TTFT measurement.
        # -----------------------------------------------------
        start_time = time.time()

        # -----------------------------------------------------
        # Send the request to vLLM with stream=True.
        # stream=True on self.client.post means the response
        # body is not downloaded all at once — we read it
        # chunk by chunk as tokens arrive from the model.
        # catch_response=True lets us manually mark the
        # request as success or failure based on what we see.
        # -----------------------------------------------------
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            stream=True,
            catch_response=True
        ) as response:

            # -------------------------------------------------
            # Tracking variables
            # -------------------------------------------------
            ttft = None           # time to first token
            total_tokens = 0      # count tokens received
            first_chunk = True    # flag — True until first token arrives, then False
            first_line_logged = False  # log the first raw line once to verify format

            # -------------------------------------------------
            # Read the streaming response line by line.
            # Each line from vLLM looks like:
            #   data: {"choices": [{"delta": {"content": "hi"}}]}
            # The last line is always:
            #   data: [DONE]
            # -------------------------------------------------
            for line in response.iter_lines():

                # Network data arrives as raw bytes — decode to readable string
                if isinstance(line, bytes):
                    line = line.decode("utf-8")

                # Skip empty lines
                if not line:
                    continue

                # -------------------------------------------------
                # Log the very first raw line we receive.
                # This verifies the format assumption — if the
                # format is wrong you will see it here immediately
                # without having to guess where parsing broke.
                # Only logged once per request to avoid noise.
                # -------------------------------------------------
                if not first_line_logged:
                    logger.debug(f"First raw line from vLLM: {line}")
                    first_line_logged = True

                # Strip the "data: " prefix from each line
                if line.startswith("data: "):
                    line = line[6:]  # strip exactly 6 chars to get the JSON

                # Stop when we see the [DONE] marker
                if line == "[DONE]":
                    logger.debug("Received [DONE] — stream complete")
                    break

                # ---------------------------------------------
                # Parse the JSON chunk and extract content.
                # Each chunk contains one or more tokens in
                # the delta.content field.
                # ---------------------------------------------
                try:
                    chunk = json.loads(line)
                    content = chunk["choices"][0]["delta"].get("content", "")  # get token text, default "" if missing

                    if content:
                        total_tokens += 1

                        # -------------------------------------
                        # TTFT measurement — first chunk only.
                        # The moment we see the first content
                        # token, record the time elapsed since
                        # the request was sent. This is TTFT.
                        # -------------------------------------
                        if first_chunk:
                            ttft = time.time() - start_time
                            first_chunk = False
                            logger.debug(f"TTFT captured: {ttft:.3f}s — first token content: '{content}'")

                except json.JSONDecodeError:
                    # -----------------------------------------
                    # JSON parse failed — log the raw line so
                    # you can see exactly what broke the parser.
                    # Without this log it silently continues
                    # and you never know what vLLM actually sent.
                    # -----------------------------------------
                    logger.error(f"JSON parse failed. Raw line was: '{line}'")
                    continue

            # -------------------------------------------------
            # All chunks read. Calculate final metrics.
            # -------------------------------------------------
            total_time = time.time() - start_time

            # -------------------------------------------------
            # Guard — if no tokens received something went wrong
            # silently. Log a warning so it is visible.
            # -------------------------------------------------
            if total_tokens == 0:
                logger.warning("Request completed but zero tokens received — check vLLM response format")

            # -------------------------------------------------
            # TPOT — Time Per Output Token
            # Time spent generating after the first token,
            # divided by remaining tokens.
            # Formula: (total_time - ttft) / (total_tokens - 1)
            # Guard against division by zero if only 1 token.
            # -------------------------------------------------
            if ttft and total_tokens > 1:
                tpot = (total_time - ttft) / (total_tokens - 1)
            else:
                tpot = 0

            # -------------------------------------------------
            # Tokens per second — total tokens divided by
            # total time. Measures GPU generation throughput.
            # -------------------------------------------------
            tokens_per_second = total_tokens / total_time if total_time > 0 else 0

            # -------------------------------------------------
            # Log a summary line for every completed request.
            # Use this to verify numbers look reasonable before
            # trusting the CSV output. If TTFT is 0 or tokens
            # is 0, something went wrong in the streaming loop.
            # -------------------------------------------------
            logger.debug(
                f"Request complete — "
                f"tokens: {total_tokens}, "
                f"TTFT: {ttft:.3f}s, "
                f"TPOT: {tpot:.4f}s, "
                f"tokens/sec: {tokens_per_second:.1f}, "
                f"total_time: {total_time:.3f}s"
            )

            # -------------------------------------------------
            # Report TTFT as a custom Locust metric.
            # This makes it appear in Locust output alongside
            # built-in metrics like avg latency and P95.
            # response_time must be in milliseconds.
            # -------------------------------------------------
            if ttft:
                self.environment.events.request.fire(
                    request_type="TTFT",
                    name="time_to_first_token",
                    response_time=ttft * 1000,
                    response_length=0,
                    exception=None
                )

            # -------------------------------------------------
            # Report TPOT as a custom Locust metric.
            # -------------------------------------------------
            if tpot:
                self.environment.events.request.fire(
                    request_type="TPOT",
                    name="time_per_output_token",
                    response_time=tpot * 1000,
                    response_length=0,
                    exception=None
                )

            # -------------------------------------------------
            # Mark the request as successful in Locust.
            # Without this Locust marks it as failed because
            # we used catch_response=True.
            # -------------------------------------------------
            response.success()