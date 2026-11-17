import argparse
import contextvars
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

CHESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHESS_DIR / "src"))
sys.path.insert(0, str(CHESS_DIR.parent))

from config import get_config
from preprocess import index_dir
from database_utils.db_values import load_db_lsh
from database_utils.execution import ExecutionStatus
from runner.database_manager import DatabaseCatalog, DatabaseManager
from runner.logger import Logger
from runner.task import Task
from workflow.system_state import SystemState
from workflow.agents.information_retriever.information_retriever import InformationRetriever
from workflow.agents.schema_selector.schema_selector import SchemaSelector
from workflow.agents.candidate_generator.candidate_generator import CandidateGenerator
from utils import cost
from utils.ex_acc import get_mysql_credentials
from utils.hints import EvalConfig, read_json, get_evidence, get_retrieved_tables

# Runs CHESS (IR -> SS -> CG) on BEAVER; replaces CHESS's main.py and run_manager.py. Each question writes
# output_dir/<id>/result.sql (read by eval/utils/unify.py), history.json, generation.log and cost.json (token
# usage of every LLM and embedding call); the run's totals go to output_dir/cost_summary.json.

AGENT_CLASSES = {
    "information_retriever": InformationRetriever,
    "schema_selector": SchemaSelector,
    "candidate_generator": CandidateGenerator,
}


def pick_final_sql(state: SystemState):
    """
    CHESS run_manager.pick_final_sql: the latest query that executes with a non-empty result, else the
    generated one (no gold SQL is involved).
    """
    final_sql, final_status, generated_at_step = "", None, None
    for sql_id, sql_meta_infos in state.SQL_meta_infos.items():
        if not sql_meta_infos:
            continue
        sql_meta_info = sql_meta_infos[0]
        if final_sql and sql_meta_info.SQL == final_sql:
            continue
        status = sql_meta_info.execution_status
        if not final_sql or status == ExecutionStatus.SYNTACTICALLY_CORRECT:
            final_sql, final_status, generated_at_step = sql_meta_info.SQL, status, sql_id
    return final_sql, final_status, generated_at_step


def run_question(item, catalog: DatabaseCatalog, config, output_dir: Path) -> str:
    q, tables, join_keys, evidence = item
    instance_dir = output_dir / q["id"]
    DatabaseManager.open(catalog, q["db"], tables, join_keys, seed=q["id"])
    Logger.open(q["id"], instance_dir)
    (instance_dir / "generation.log").unlink(missing_ok=True)
    # A rerun of the question replaces its cost.json, so it holds the calls of the last attempt
    recorder = cost.start_recording()

    task = Task(question_id=q["id"], db_id=q["db"], question=q["question"], evidence=evidence)
    state = SystemState(task=task, tentative_schema=DatabaseManager().get_db_schema(), execution_history=[])
    try:
        # Agents are built per question: tools keep per-run results on themselves
        for agent_name, agent_config in config["team_agents"].items():
            AGENT_CLASSES[agent_name](agent_config)(state)
    finally:
        recorder.write(instance_dir / "cost.json")

    final_sql, final_status, generated_at_step = pick_final_sql(state)
    state.execution_history.append({"final_sql": {
        "SQL": final_sql,
        "EXECUTION_STATUS": final_status.value if final_status else None,
        "GENERATED_AT_STEP": generated_at_step,
    }})
    Logger().dump_history_to_file(state.execution_history)
    with open(instance_dir / "result.sql", "w") as f:
        f.write(final_sql.strip())
    return final_status.value if final_status else "NO_SQL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="LLM for every CHESS step")
    parser.add_argument("--embedding_model", type=str, default="qwen/qwen3-embedding-8b",
                        help="Embedding model for value retrieval (OpenRouter name unless EMBED_BASE_URL is set)")
    parser.add_argument("--dataset", type=str, default="dw")
    parser.add_argument("--q_fn", type=str, default="dev")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=str(CHESS_DIR.parent.parent / "data"))
    parser.add_argument("--num_workers", type=int, default=8, help="Questions processed in parallel")
    parser.add_argument("--log_level", type=str, default="warning")
    parser.add_argument("--gold_tables", action="store_true")
    parser.add_argument("--join_keys", action="store_true")
    parser.add_argument("--mapping", action="store_true")
    parser.add_argument("--knowledge", action="store_true")
    parser.add_argument("--decomp", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='%(levelname)s: %(message)s')
    load_dotenv(CHESS_DIR.parent.parent / ".env")

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

    db_id = next(iter(dev_tables.values()))["db"]
    lsh_dir = index_dir(db_id)
    if not (lsh_dir / "lsh.pkl").exists():
        raise SystemExit(f"Value index not found at {lsh_dir}; run: python preprocess.py --dataset {args.dataset}")
    print(f"Loading value index from {lsh_dir}")
    mysql_creds = get_mysql_credentials(args.dataset)
    if not mysql_creds:
        raise SystemExit("MYSQL_HOST / MYSQL_USER / MYSQL_PASSWORD are not set in .env")
    catalog = DatabaseCatalog(dev_tables, load_db_lsh(lsh_dir), mysql_creds)
    config = get_config(args.model, args.embedding_model)

    output_dir = Path(args.output_dir)
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
        join_keys = q["join_keys"] if eval_config.join_keys else None
        todo.append((q, tables, join_keys, get_evidence(q, eval_config, structures)))

    if not todo:
        print("All questions already processed.")
        cost.write_summary(str(output_dir))
        return
    print(f"Running {len(todo)} questions with {args.num_workers} workers...")

    statuses = {}
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # A fresh context per question keeps its DatabaseManager and Logger to itself
        futures = {executor.submit(contextvars.Context().run, run_question, item, catalog, config, output_dir): item[0]["id"]
                   for item in todo}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                status = future.result()
            except Exception as e:
                print(f"Error processing {futures[future]}: {type(e).__name__}: {e}")
                status = "ERROR"
            statuses[status] = statuses.get(status, 0) + 1

    print(f"Final SQL status counts: {statuses}")
    cost.write_summary(str(output_dir))
    failed = statuses.get("NO_SQL", 0) + statuses.get("ERROR", 0)
    if failed:
        print(f"{failed}/{len(todo)} questions produced no SQL; rerun with the same --timestamp to retry them.")


if __name__ == "__main__":
    main()
