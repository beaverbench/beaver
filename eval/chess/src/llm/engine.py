import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI, APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, RateLimitError

from utils import cost, llm_client

# Replaces CHESS's langchain engine configs with plain OpenAI-compatible clients. Chat requests use the routing,
# LLM_RPM pacing, rate-limit backoff and usage recording shared by the baselines (eval/utils/llm_client.py);
# embeddings default to OpenRouter (EMBED_BASE_URL / EMBED_API_KEY override it) and are recorded here.

_clients = {}
_clients_lock = threading.Lock()

# Attempts per embedding batch; attempt n waits n * TRANSIENT_RETRY_DELAY seconds after a failure
EMBED_MAX_ATTEMPTS = 5


def _chat_client(model: str) -> OpenAI:
    with _clients_lock:
        if model not in _clients:
            _clients[model] = llm_client.make_client(model)
        return _clients[model]


def _embed_client() -> OpenAI:
    key = ("embed", os.environ.get("EMBED_BASE_URL"))
    with _clients_lock:
        if key not in _clients:
            _clients[key] = OpenAI(base_url=os.environ.get("EMBED_BASE_URL") or "https://openrouter.ai/api/v1",
                                   api_key=os.environ.get("EMBED_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "EMPTY",
                                   max_retries=0)
        return _clients[key]


# Models that reject a non-default temperature (e.g. OpenAI reasoning models) are retried without one,
# and models that reject a JSON-schema response_format are retried without it (the output is still parsed)
_no_temperature_models = set()
_no_response_format_models = set()


def chat(model: str, prompt: str, temperature: float, response_format: Optional[Dict[str, Any]] = None,
         step: str = "") -> str:
    client = llm_client.LLMClient(_chat_client(model), step=step)
    while True:
        kwargs = {} if model in _no_temperature_models else {"temperature": temperature}
        if response_format is not None and model not in _no_response_format_models:
            kwargs["response_format"] = response_format
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except BadRequestError as e:
            message = str(e).lower()
            if "temperature" in message and model not in _no_temperature_models:
                print(f"[{model}] rejects temperature={temperature}; using its default temperature from now on")
                _no_temperature_models.add(model)
                continue
            if "response_format" in kwargs and any(k in message for k in ("response_format", "json_schema", "structured")):
                print(f"[{model}] rejects structured output; parsing plain-text JSON from now on")
                _no_response_format_models.add(model)
                continue
            raise


def embed(texts: List[str], model: str, batch_size: int = 256, step: str = "") -> np.ndarray:
    """Returns L2-normalized embeddings, so dot products are cosine similarities as with OpenAI embeddings."""
    client = _embed_client()
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        for attempt in range(EMBED_MAX_ATTEMPTS):
            try:
                raw = client.embeddings.with_raw_response.create(model=model, input=batch)
                response = raw.parse()
                embeddings += [e.embedding for e in response.data]
                cost.record(cost.usage_record(step, model, "embedding", response.usage, raw.headers,
                                              prompt_text="\n".join(batch)))
                break
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError):
                if attempt == EMBED_MAX_ATTEMPTS - 1:
                    raise
                time.sleep(llm_client.TRANSIENT_RETRY_DELAY * (attempt + 1))
    embeddings = np.array(embeddings, dtype=np.float32).reshape(len(texts), -1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.where(norms == 0, 1, norms)
