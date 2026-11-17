import os
import random
import threading
import time
from types import SimpleNamespace
from typing import Callable, Union

from openai import OpenAI, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

try:
    from . import cost  # imported as utils.llm_client
except ImportError:
    import cost  # imported as llm_client, with eval/utils on sys.path

# LLM client plumbing shared by the baselines:
# - make_client() picks the endpoint for a model name
# - LLMClient wraps a client so that chat.completions.create() is paced by LLM_RPM, waits out rate limits,
#   retries transient errors, and records each call's token usage for the current question (cost.py)
# Methods with their own `utils` module import it as llm_client after appending eval/utils to sys.path
# (appended, so their `utils` keeps its name); a method should use one import style for both modules.


def make_client(model: str) -> OpenAI:
    """
    LLM_BASE_URL (+ LLM_API_KEY) takes every request when set; otherwise Gemini models go to Google,
    names with a "/" (e.g. anthropic/claude-sonnet-4.5) to OpenRouter, and the rest to OpenAI.
    Retries are left to LLMClient, so the SDK's own quick retries are off.
    """
    base_url = os.environ.get("LLM_BASE_URL")
    if base_url:
        return OpenAI(base_url=base_url, api_key=os.environ.get("LLM_API_KEY") or "EMPTY", max_retries=0)
    if "gemini" in model:
        return OpenAI(api_key=os.environ.get("GEMINI_API_KEY"),
                      base_url="https://generativelanguage.googleapis.com/v1beta/openai/", max_retries=0)
    if "/" in model:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"), max_retries=0)
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=0)


def custom_endpoint() -> bool:
    """True when requests go to LLM_BASE_URL, where provider-specific parameters (e.g. OpenRouter's reasoning) don't apply."""
    return bool(os.environ.get("LLM_BASE_URL"))


# Retry settings of call_with_backoff (seconds); the number of rate-limit retries is LLM_MAX_RATE_LIMIT_RETRIES (default 20)
MAX_TRANSIENT_RETRIES = 3     # retries after a connection error, timeout or server error
TRANSIENT_RETRY_DELAY = 10    # wait before transient retry n is n * TRANSIENT_RETRY_DELAY
RATE_LIMIT_BASE_WAIT = 60     # wait after the n-th 429 is min(RATE_LIMIT_BASE_WAIT * 2**n, RATE_LIMIT_MAX_WAIT) + jitter
RATE_LIMIT_MAX_WAIT = 300
RATE_LIMIT_JITTER = 10

# Process-wide request pacing shared by all threads. LLM_RPM caps requests per minute (unset/0 = no cap);
# a 429 pauses every thread, not just the one that hit it.
_request_lock = threading.Lock()
_next_request_time = 0.0


def _acquire_request_slot():
    global _next_request_time
    rpm = float(os.environ.get("LLM_RPM") or 0)
    interval = 60.0 / rpm if rpm > 0 else 0.0
    with _request_lock:
        now = time.monotonic()
        start = max(now, _next_request_time)
        _next_request_time = start + interval
    if start > now:
        time.sleep(start - now)


def _pause_all_requests(seconds):
    global _next_request_time
    with _request_lock:
        _next_request_time = max(_next_request_time, time.monotonic() + seconds)


def split_rpm_across_processes(processes: int) -> None:
    """For methods that run questions in worker processes: each process paces itself, so give each 1/processes of LLM_RPM."""
    rpm = float(os.environ.get("LLM_RPM") or 0)
    if rpm > 0 and processes > 1:
        os.environ["LLM_RPM"] = str(rpm / processes)


def call_with_backoff(make_request: Callable, label: str = ""):
    """Runs make_request() paced by LLM_RPM, waiting out rate limits (429) and retrying transient errors."""
    max_rate_limit_retries = int(os.environ.get("LLM_MAX_RATE_LIMIT_RETRIES", 20))
    rate_limit_retries = 0
    transient_retries = 0
    while True:
        _acquire_request_slot()
        try:
            return make_request()
        except RateLimitError:
            if rate_limit_retries >= max_rate_limit_retries:
                raise
            wait = min(RATE_LIMIT_BASE_WAIT * 2 ** rate_limit_retries, RATE_LIMIT_MAX_WAIT) + random.uniform(0, RATE_LIMIT_JITTER)
            rate_limit_retries += 1
            print(f"[Rate limited] {label} pausing all requests {wait:.0f}s (retry {rate_limit_retries}/{max_rate_limit_retries})")
            _pause_all_requests(wait)
        except (APIConnectionError, APITimeoutError, InternalServerError):
            if transient_retries >= MAX_TRANSIENT_RETRIES:
                raise
            transient_retries += 1
            time.sleep(TRANSIENT_RETRY_DELAY * transient_retries)


class LLMClient:
    """
    Wraps an OpenAI client: chat.completions.create(...) goes through call_with_backoff and records the call's
    usage (from the raw response, whose headers carry a LiteLLM gateway's cost) under `step` for the current
    question. step is a label or a function returning one. Other attributes reach the wrapped client.
    """

    def __init__(self, client: OpenAI, step: Union[str, Callable[[], str]] = "generation"):
        self._client = client
        self._step = step
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def __getattr__(self, name):
        # Reached only for attributes not set in __init__; without _client (e.g. copy or unpickle, which skip
        # __init__) the lookup below would recurse forever
        if name == "_client":
            raise AttributeError(name)
        return getattr(self._client, name)

    def _create(self, **kwargs):
        model = kwargs.get("model", "")
        raw = call_with_backoff(lambda: self._client.chat.completions.with_raw_response.create(**kwargs), label=f"model={model}")
        response = raw.parse()
        step = self._step() if callable(self._step) else self._step
        prompt_text = "\n".join(str(m.get("content", "")) for m in kwargs.get("messages", []))
        completion_text = "\n".join(choice.message.content or "" for choice in response.choices)
        cost.record(cost.usage_record(step, model, "chat", response.usage, raw.headers,
                                      prompt_text=prompt_text, completion_text=completion_text))
        return response
