import argparse
import asyncio
import contextlib
import dataclasses
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, HookMatcher, ResultMessage,
                              create_sdk_mcp_server, query, tool)
from dotenv import load_dotenv
from tqdm import tqdm

CC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CC_DIR.parent))

from prompt import SYSTEM_APPEND, get_prompt, get_schema
from utils import cost
from utils.ex_acc import get_mysql_credentials
from utils.hints import EvalConfig, read_json, get_evidence, get_retrieved_tables
from utils.readonly_sql import execute_readonly

# Runs Claude Code (through the Claude Agent SDK) on BEAVER as a SQL agent. Its only tools are execute_sql
# (read-only queries on the MySQL database) and submit_sql (records the final query and ends the session);
# Claude Code's built-in tools are off. The CLI loads no settings and runs with CLAUDE_CONFIG_DIR and its cwd
# inside the run folder, so it neither reads nor writes ~/.claude. Each question writes output_dir/<id>/result.sql
# (read by eval/utils/unify.py), trace.jsonl (the prompt and every message of the session) and cost.json; the
# run's totals go to output_dir/cost_summary.json.

MAX_QUERIES = 30              # execute_sql calls per question; later calls ask the agent to submit
SOFT_TIME_LIMIT = 600         # seconds; after it, execute_sql asks the agent to submit
HARD_TIME_LIMIT = 900         # seconds; the session is stopped
MAX_TURNS = MAX_QUERIES + 10  # backstop for an agent that ignores the budget message
NUDGE_MAX_TURNS = 3           # turns of the follow-up session for an agent that stopped without submitting
MAX_ROWS = 50                 # rows of a query result shown to the agent
MAX_CELL_CHARS = 200

BUDGET_MESSAGE = "Query budget exhausted. Call submit_sql now with your best SQL query based on what you have collected."
NUDGE_PROMPT = "You have not submitted an answer. Call submit_sql now with your best SQL query based on what you have collected."
SUBMIT_TOOL = "mcp__beaver__submit_sql"
TOOLS = ["mcp__beaver__execute_sql", SUBMIT_TOOL]


def _text(text: str, is_error: bool = False) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], **({"is_error": True} if is_error else {})}


def format_result(columns: List[str], rows: List[tuple], total: int) -> str:
    def cell(v):
        s = "NULL" if v is None else str(v).replace("\t", " ").replace("\n", " ")
        return s if len(s) <= MAX_CELL_CHARS else s[:MAX_CELL_CHARS] + "..."
    lines = ["\t".join(columns)] + ["\t".join(cell(v) for v in row) for row in rows]
    lines.append(f"[{total} rows" + (f", first {len(rows)} shown]" if total > len(rows) else "]"))
    return "\n".join(lines)


class Session:
    """One question's agent state, shared by its tools and hook."""

    def __init__(self, mysql_creds: Dict[str, Any]):
        self.mysql_creds = mysql_creds
        self.start = time.monotonic()
        self.queries = 0
        self.submitted: Optional[str] = None

    def tools(self):
        @tool("execute_sql", f"Run one read-only MySQL query (a single SELECT or WITH statement) on the database. "
                             f"Returns up to {MAX_ROWS} rows as tab-separated text.", {"sql": str})
        async def execute_sql(args):
            if self.queries >= MAX_QUERIES or time.monotonic() - self.start > SOFT_TIME_LIMIT:
                return _text(BUDGET_MESSAGE)
            self.queries += 1
            try:
                columns, rows, total = await asyncio.to_thread(execute_readonly, self.mysql_creds, args["sql"], max_rows=MAX_ROWS)
            except Exception as e:
                return _text(f"Error: {e}", is_error=True)
            return _text(f"{format_result(columns, rows, total)}\n({self.queries}/{MAX_QUERIES} queries used)")

        @tool("submit_sql", "Submit the final SQL query that answers the question. This ends the task.", {"sql": str})
        async def submit_sql(args):
            sql = args["sql"].strip()
            if not sql:
                return _text("Error: the SQL query is empty.", is_error=True)
            self.submitted = sql
            return _text("Submitted.")

        return [execute_sql, submit_sql]

    async def stop_after_submit(self, input_data, tool_use_id, context):
        return {"continue_": False, "stopReason": "SQL submitted"} if self.submitted is not None else {}


def _usage_entry(model: str, input_tokens: int, cache_read: int, cache_write: int, output_tokens: int,
                 num_calls: int, usage_source: str) -> Dict[str, Any]:
    # Anthropic's input_tokens leave out cache reads and writes; cost.py's prompt_tokens include them
    entry = cost.usage_record("agent", model, "chat", {
        "prompt_tokens": input_tokens + cache_read + cache_write, "completion_tokens": output_tokens,
        "cache_read_input_tokens": cache_read, "cache_creation_input_tokens": cache_write,
    })
    entry.update(usage_source=usage_source, num_calls=num_calls)
    return entry


class UsageTracker:
    """
    Token usage of one question's sessions. A session's ResultMessage totals it per model (background calls
    included), and becomes one cost entry per model. A resumed session's totals also count the session it resumes,
    so its entries hold only the difference. A session stopped by the time limit has no ResultMessage, so its usage
    is summed from the assistant messages instead; their output counts are taken at the start of each response, so
    those entries are marked estimated.
    """

    USAGE_KEYS = ("inputTokens", "cacheReadInputTokens", "cacheCreationInputTokens", "outputTokens")

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self._calls: Dict[str, AssistantMessage] = {}
        self._reported: Dict[str, Dict[str, int]] = {}  # per-model totals of the last ResultMessage

    def see(self, message: AssistantMessage) -> None:
        # A response arrives as one message per content block, all with its message_id
        if message.message_id and message.usage:
            self._calls[message.message_id] = message

    def end_session(self, result: Optional[ResultMessage]) -> None:
        if result is not None and result.model_usage:
            calls = Counter(m.model for m in self._calls.values())
            totals = {model: {k: u.get(k, 0) for k in self.USAGE_KEYS} for model, u in result.model_usage.items()}
            for model, total in totals.items():
                previous = self._reported.get(model, {})
                d = {k: total[k] - previous.get(k, 0) for k in self.USAGE_KEYS}
                if any(d.values()):
                    self.entries.append(_usage_entry(model, d["inputTokens"], d["cacheReadInputTokens"],
                                                     d["cacheCreationInputTokens"], d["outputTokens"], calls.get(model, 1), "api"))
            self._reported = totals
        else:
            for m in self._calls.values():
                u = m.usage
                self.entries.append(_usage_entry(m.model, u.get("input_tokens", 0), u.get("cache_read_input_tokens", 0),
                                                 u.get("cache_creation_input_tokens", 0), u.get("output_tokens", 0), 1, "estimated"))
        self._calls = {}


def _record(trace, message) -> None:
    data = dataclasses.asdict(message) if dataclasses.is_dataclass(message) else {"repr": repr(message)}
    trace.write(json.dumps({"type": type(message).__name__, **data}, default=str) + "\n")
    trace.flush()


async def run_session(prompt: str, options: ClaudeAgentOptions, trace, usage: UsageTracker) -> Optional[ResultMessage]:
    result = None
    try:
        async with contextlib.aclosing(query(prompt=prompt, options=options)) as messages:
            async for message in messages:
                _record(trace, message)
                if isinstance(message, AssistantMessage):
                    usage.see(message)
                elif isinstance(message, ResultMessage):
                    result = message
    finally:
        usage.end_session(result)
    return result


async def run_question(q: Dict[str, Any], prompt: str, mysql_creds: Dict[str, Any], model: str,
                       output_dir: Path, workdir: Path, env: Dict[str, str]) -> str:
    instance_dir = output_dir / q["id"]
    instance_dir.mkdir(parents=True, exist_ok=True)
    session = Session(mysql_creds)
    usage = UsageTracker()
    stderr_lines: List[str] = []
    options = ClaudeAgentOptions(
        model=model,
        tools=[],
        mcp_servers={"beaver": create_sdk_mcp_server("beaver", tools=session.tools())},
        allowed_tools=TOOLS,
        strict_mcp_config=True,
        setting_sources=[],
        skills=[],
        system_prompt={"type": "preset", "preset": "claude_code", "append": SYSTEM_APPEND},
        permission_mode="dontAsk",
        cwd=str(workdir),
        env=env,
        max_turns=MAX_TURNS,
        hooks={"PostToolUse": [HookMatcher(matcher=SUBMIT_TOOL, hooks=[session.stop_after_submit])]},
        stderr=stderr_lines.append,
    )

    async def agent(trace):
        result = await run_session(prompt, options, trace, usage)
        # The agent stopped without submitting: ask once more (API errors are left to a rerun)
        if (session.submitted is None and result is not None and result.api_error_status is None
                and result.subtype in ("success", "error_max_turns")):
            trace.write(json.dumps({"type": "Prompt", "prompt": NUDGE_PROMPT}) + "\n")
            nudge_options = dataclasses.replace(options, resume=result.session_id, max_turns=NUDGE_MAX_TURNS)
            await run_session(NUDGE_PROMPT, nudge_options, trace, usage)

    status = "NO_SQL"
    with open(instance_dir / "trace.jsonl", "w") as trace:
        trace.write(json.dumps({"type": "Prompt", "system_append": SYSTEM_APPEND, "prompt": prompt}) + "\n")
        try:
            await asyncio.wait_for(agent(trace), HARD_TIME_LIMIT)
        except asyncio.TimeoutError:
            trace.write(json.dumps({"type": "Timeout", "seconds": HARD_TIME_LIMIT}) + "\n")
            status = "TIMEOUT"
        except Exception as e:
            trace.write(json.dumps({"type": "Error", "error": f"{type(e).__name__}: {e}"}) + "\n")
            print(f"Error processing {q['id']}: {type(e).__name__}: {e}")
            status = "ERROR"

    if session.submitted is not None:
        status = "SUBMITTED"
    with open(instance_dir / "result.sql", "w") as f:
        f.write(session.submitted or "")
    with open(instance_dir / "cost.json", "w") as f:
        json.dump({"calls": usage.entries}, f, indent=2)
    if stderr_lines:
        with open(instance_dir / "cli_stderr.log", "w") as f:
            f.write("\n".join(stderr_lines))
    return status


async def run_all(todo, mysql_creds, model: str, output_dir: Path, num_workers: int) -> Dict[str, int]:
    workdir = output_dir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    env = {
        "CLAUDE_CONFIG_DIR": str(output_dir / ".claude"),
        # Claude Code's background calls use the haiku model; a gateway may serve only --model
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        # The workdir sits inside this repo (output/ is only gitignored): leave its git status (branch, commits,
        # git user) out of the system prompt, and keep auto-memory off
        "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    }
    semaphore = asyncio.Semaphore(num_workers)

    async def one(q, prompt):
        async with semaphore:
            return await run_question(q, prompt, mysql_creds, model, output_dir, workdir, env)

    statuses: Dict[str, int] = {}
    for future in tqdm(asyncio.as_completed([one(q, prompt) for q, prompt in todo]), total=len(todo)):
        status = await future
        statuses[status] = statuses.get(status, 0) + 1
    return statuses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Claude model, e.g. azure/anthropic/claude-sonnet-4-5 on a gateway")
    parser.add_argument("--dataset", type=str, default="dw")
    parser.add_argument("--q_fn", type=str, default="dev")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=str(CC_DIR.parent.parent / "data"))
    parser.add_argument("--num_workers", type=int, default=8, help="Questions processed in parallel")
    parser.add_argument("--gold_tables", action="store_true")
    parser.add_argument("--join_keys", action="store_true")
    parser.add_argument("--mapping", action="store_true")
    parser.add_argument("--knowledge", action="store_true")
    parser.add_argument("--decomp", action="store_true")
    args = parser.parse_args()

    load_dotenv(CC_DIR.parent.parent / ".env")
    if not (os.environ.get("ANTHROPIC_API_KEY")
            or (os.environ.get("ANTHROPIC_BASE_URL") and os.environ.get("ANTHROPIC_AUTH_TOKEN"))):
        raise SystemExit("Set ANTHROPIC_API_KEY, or ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN for a gateway, in .env")
    mysql_creds = get_mysql_credentials(args.dataset)
    if not mysql_creds:
        raise SystemExit("MYSQL_HOST / MYSQL_USER / MYSQL_PASSWORD are not set in .env")

    eval_config = EvalConfig(
        gold_tables=args.gold_tables, join_keys=args.join_keys, mapping=args.mapping,
        knowledge=args.knowledge, decomp=args.decomp,
    )
    print(eval_config)

    data_dir = Path(args.data_dir)
    qs = read_json(data_dir / args.dataset / f"{args.q_fn}.json")
    dev_tables = read_json(data_dir / args.dataset / "dev_tables.json")
    structures = read_json(data_dir / "template_structure.json")
    retrieved_tables = None if eval_config.gold_tables else get_retrieved_tables(args.dataset, str(data_dir))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    todo = []
    for q in qs:
        if eval_config.gold_tables:
            tables = q["tables"]
        else:
            if q["id"] not in retrieved_tables:
                continue
            tables = retrieved_tables[q["id"]]
        # Skip already processed questions (empty results from failed runs are retried)
        result_file = output_dir / q["id"] / "result.sql"
        if result_file.exists() and result_file.stat().st_size > 0:
            continue
        schema = get_schema(tables, dev_tables, q["join_keys"] if eval_config.join_keys else None)
        todo.append((q, get_prompt(q, schema, get_evidence(q, eval_config, structures))))

    if not todo:
        print("All questions already processed.")
        cost.write_summary(str(output_dir))
        return
    print(f"Running {len(todo)} questions with {args.num_workers} workers...")

    statuses = asyncio.run(run_all(todo, mysql_creds, args.model, output_dir, args.num_workers))
    print(f"Status counts: {statuses}")
    cost.write_summary(str(output_dir))
    failed = len(todo) - statuses.get("SUBMITTED", 0)
    if failed:
        print(f"{failed}/{len(todo)} questions produced no SQL; rerun with the same --timestamp to retry them.")


if __name__ == "__main__":
    main()
