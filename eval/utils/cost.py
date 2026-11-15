import argparse
import contextvars
import glob
import json
import os
import re
import sys
import threading
from typing import Any, Dict, List, Optional

# Token and cost accounting shared by the baselines.
# Every LLM or embedding call is recorded as one entry holding the raw token counts (no prices applied); an entry
# may instead total several calls, counted by its num_calls (Claude Code reports usage per question, not per call);
# a baseline writes a question's entries to <instance_dir>/cost.json, and unify.py copies them to
# unified-output/<method>/<run>/cost/<id>.json. summarize() prices the chat tokens with PRICES (edit it to add or
# change a price) and writes cost_summary.json; run this file as a script to redo that for a run, e.g. after
# changing PRICES.

# List prices in USD per million tokens, keyed by normalize_model() names. cache_read / cache_write price
# prompt-cache hits and writes (Anthropic: 0.1x and 1.25x the input price).
PRICES = {
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
}


def normalize_model(model: str) -> str:
    """'azure/anthropic/claude-sonnet-4-5', 'aws/anthropic/bedrock-claude-sonnet-4-5', 'anthropic/claude-sonnet-4.5'
    and 'claude-sonnet-4-5-20250929' -> 'claude-sonnet-4-5'."""
    name = model.lower().rsplit("/", 1)[-1].replace(".", "-")
    name = re.sub(r"^bedrock-", "", name)
    name = re.sub(r"-(global|latest)$", "", name)
    return re.sub(r"-\d{8}$", "", name)


def _to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(getattr(obj, "__dict__", {}))


def _header_float(headers, name: str) -> Optional[float]:
    try:
        value = headers.get(name) if headers is not None else None
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _estimate_tokens(text: Optional[str]) -> int:
    if not text:
        return 0
    import tiktoken
    return len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))


def usage_record(step: str, model: str, kind: str, usage: Any = None, headers=None,
                 prompt_text: Optional[str] = None, completion_text: Optional[str] = None) -> Dict[str, Any]:
    """
    One call's entry. kind is "chat" or "embedding"; usage is the response's usage object (OpenAI format) and
    headers its HTTP headers (a LiteLLM gateway reports its cost there). prompt_tokens counts the whole input,
    cached tokens included (the OpenAI convention). Without usage, tokens are estimated from the texts with
    tiktoken and the entry is marked usage_source="estimated".
    """
    u = _to_dict(usage)
    details = _to_dict(u.get("prompt_tokens_details"))
    # Chat Completions name them prompt/completion_tokens; the Responses API input/output_tokens
    prompt_tokens = u.get("prompt_tokens", u.get("input_tokens"))
    completion_tokens = u.get("completion_tokens", u.get("output_tokens"))
    usage_source = "api"
    if prompt_tokens is None:
        prompt_tokens = _estimate_tokens(prompt_text)
        completion_tokens = _estimate_tokens(completion_text)
        usage_source = "estimated"
    cache_read = u.get("cache_read_input_tokens")
    if cache_read is None:
        cache_read = details.get("cached_tokens")
    cache_write = u.get("cache_creation_input_tokens")
    if cache_write is None:
        cache_write = details.get("cache_creation_tokens") or details.get("cache_write_tokens")
    return {
        "step": step,
        "model": model,
        "kind": kind,
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "cache_read_tokens": int(cache_read or 0),
        "cache_write_tokens": int(cache_write or 0),
        "usage_source": usage_source,
        "gateway_cost": _header_float(headers, "x-litellm-response-cost"),
        "gateway_cost_original": _header_float(headers, "x-litellm-response-cost-original"),
    }


class CostRecorder:
    """Collects one question's entries; thread-safe, since a question may make concurrent calls."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(self, entry: Dict[str, Any]) -> None:
        with self._lock:
            self.calls.append(entry)

    def write(self, path: str, append: bool = False) -> None:
        """append=True adds the calls to an existing cost.json, for methods that process a question over several runs."""
        with self._lock:
            calls = list(self.calls)
        if append and os.path.exists(path):
            with open(path) as f:
                calls = json.load(f)["calls"] + calls
        with open(path, "w") as f:
            json.dump({"calls": calls}, f, indent=2)


# The recorder of the question being processed, for code that cannot pass one down to its LLM calls.
# Threads started with contextvars.copy_context() share it with their parent.
_current_recorder: contextvars.ContextVar = contextvars.ContextVar("cost_recorder", default=None)


def start_recording() -> CostRecorder:
    recorder = CostRecorder()
    _current_recorder.set(recorder)
    return recorder


def record(entry: Dict[str, Any]) -> None:
    recorder = _current_recorder.get()
    if recorder is not None:
        recorder.add(entry)


def call_cost(entry: Dict[str, Any], prices: Dict[str, Dict[str, float]] = PRICES, uncached: bool = False) -> Optional[float]:
    """
    List-price cost of a chat entry in USD, or None if its model has no price. uncached=True prices every
    input token at the plain input price, i.e. the cost without prompt caching.
    """
    price = prices.get(normalize_model(entry["model"]))
    if price is None:
        return None
    if uncached:
        return (entry["prompt_tokens"] * price["input"] + entry["completion_tokens"] * price["output"]) / 1e6
    cache_read, cache_write = entry["cache_read_tokens"], entry["cache_write_tokens"]
    plain_input_tokens = max(entry["prompt_tokens"] - cache_read - cache_write, 0)
    return (plain_input_tokens * price["input"]
            + cache_read * price.get("cache_read", price["input"])
            + cache_write * price.get("cache_write", price["input"])
            + entry["completion_tokens"] * price["output"]) / 1e6


def _empty_totals() -> Dict[str, Any]:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cache_read_tokens": 0,
            "cache_write_tokens": 0, "cost_usd": 0.0, "cost_usd_uncached": 0.0, "unpriced_calls": 0}


def _add(totals: Dict[str, Any], entry: Dict[str, Any], prices: Optional[Dict[str, Dict[str, float]]]) -> Optional[float]:
    calls = entry.get("num_calls", 1)
    totals["calls"] += calls
    for key in ("prompt_tokens", "completion_tokens", "cache_read_tokens", "cache_write_tokens"):
        totals[key] += entry[key]
    if prices is None:
        return None
    cost = call_cost(entry, prices)
    if cost is None:
        totals["unpriced_calls"] += calls
        return None
    totals["cost_usd"] += cost
    totals["cost_usd_uncached"] += call_cost(entry, prices, uncached=True)
    return cost


def _finalize(totals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls without a price make the cost unknown, so cost_usd becomes None rather than an undercount; if some
    calls were priced, their subtotal is kept under *_priced_calls_only.
    """
    if totals["unpriced_calls"]:
        if totals["unpriced_calls"] < totals["calls"]:
            totals["cost_usd_priced_calls_only"] = totals["cost_usd"]
            totals["cost_usd_uncached_priced_calls_only"] = totals["cost_usd_uncached"]
        totals["cost_usd"] = totals["cost_usd_uncached"] = None
    return totals


def summarize(run_dir: str, prices: Dict[str, Dict[str, float]] = PRICES) -> Dict[str, Any]:
    """
    Sums a run's per-question cost files: <run_dir>/<id>/cost.json in a method's output folder, or
    <run_dir>/cost/<id>.json in a unified-output folder. Chat calls are priced at list
    prices twice: cost_usd as billed, with prompt-cache reads and writes (a gateway may cache prompts on its
    own, and writes cost 1.25x), and cost_usd_uncached with every input token at the input price. Embedding
    calls are counted but not priced. If any chat call's model has no price, the cost is reported as None (see
    _finalize). The gateway-reported cost is summed for reference over the calls that have it.
    """
    files = sorted(glob.glob(os.path.join(run_dir, "*", "cost.json")) + glob.glob(os.path.join(run_dir, "cost", "*.json")))
    llm, embedding = _empty_totals(), _empty_totals()
    by_step: Dict[str, Dict[str, Any]] = {}
    by_stage: Dict[str, Dict[str, Any]] = {}
    unpriced_models, estimated_calls = set(), 0
    gateway = {"calls": 0, "cost_usd": 0.0, "cost_original_usd": 0.0}
    for fn in files:
        with open(fn) as f:
            calls = json.load(f)["calls"]
        for entry in calls:
            estimated_calls += entry["usage_source"] == "estimated"
            if entry["kind"] == "embedding":
                _add(embedding, entry, None)
                continue
            if _add(llm, entry, prices) is None:
                unpriced_models.add(entry["model"])
            _add(by_step.setdefault(entry["step"], _empty_totals()), entry, prices)
            if "stage" in entry:
                _add(by_stage.setdefault(str(entry["stage"]), _empty_totals()), entry, prices)
            if entry.get("gateway_cost") is not None:
                gateway["calls"] += 1
                gateway["cost_usd"] += entry["gateway_cost"]
                gateway["cost_original_usd"] += entry.get("gateway_cost_original") or 0.0
    for key in ("cost_usd", "cost_usd_uncached", "unpriced_calls"):
        del embedding[key]
    _finalize(llm)
    for totals in list(by_step.values()) + list(by_stage.values()):
        _finalize(totals)
    num_questions = len(files)
    per_question = lambda total: None if total is None else (total / num_questions if num_questions else 0.0)
    return {
        "num_questions": num_questions,
        "llm": {**llm, "cost_usd_per_question": per_question(llm["cost_usd"]),
                "cost_usd_uncached_per_question": per_question(llm["cost_usd_uncached"])},
        "llm_by_step": by_step,
        **({"llm_by_stage": by_stage} if by_stage else {}),
        "embedding": embedding,
        "unpriced_llm_models": sorted(unpriced_models),
        "estimated_usage_calls": estimated_calls,
        "gateway_reported": gateway,
        "prices_usd_per_million_tokens": prices,
    }


def write_summary(run_dir: str, prices: Dict[str, Dict[str, float]] = PRICES) -> Dict[str, Any]:
    summary = summarize(run_dir, prices)
    with open(os.path.join(run_dir, "cost_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    llm = summary["llm"]
    tokens = (f"{llm['calls']} LLM calls, {llm['prompt_tokens']} prompt + {llm['completion_tokens']} completion tokens "
              f"over {summary['num_questions']} questions")
    if llm["cost_usd"] is None:
        print(f"Cost: not priced; {tokens}")
    else:
        print(f"Cost: ${llm['cost_usd']:.4f} (${llm['cost_usd_per_question']:.4f}/question; "
              f"${llm['cost_usd_uncached']:.4f} without prompt caching); {tokens}")
    if summary["unpriced_llm_models"]:
        print(f"No price for {summary['unpriced_llm_models']}: add them to PRICES in eval/utils/cost.py; "
              f"their tokens are counted but not priced", file=sys.stderr)
    if summary["estimated_usage_calls"]:
        print(f"{summary['estimated_usage_calls']} calls have estimated tokens (no usage returned, or a response cut "
              f"off by a time limit)", file=sys.stderr)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sum a run's per-question cost files into cost_summary.json")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="unified-output/<method>/<run_name>, or a method's run output folder")
    args = parser.parse_args()
    write_summary(args.input_dir)
