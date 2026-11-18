import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv

from utils.llm import GPTChat, EVAL_MODEL
from utils.evaluate_extraction_subtasks import (
    eval_extraction_output, EXTRACTION_PROMPT
)
from utils.utils import read_json, write_json

load_dotenv('../.env')

PROMPT_TEMPLATE = """
You are a strict SQL decomposition evaluator. Your job is to score how well the GENERATED_SQL reflects the intended QUERY DECOMPOSITION described by the REFERENCE_SUBQUESTION_SQLS.

You are NOT judging performance, style, formatting, or minor syntax differences. You ARE judging whether the generated SQL contains (explicitly or implicitly) the same sub-questions/steps as the reference decomposition and correctly composes them into the final result.

Inputs:
1) QUESTION (the natural-language question the SQL answers):
<<<
{question}
>>>

2) GENERATED_SQL:
<<<
{generated_sql}
>>>

3) REFERENCE_SUBQUESTIONS (a list; the natural-language description of each decomposition step):
<<<
{reference_subquestions}
>>>

4) REFERENCE_SUBQUESTION_SQLS (a list; each item is the SQL of the step at the same position in REFERENCE_SUBQUESTIONS; some items may be intermediate-only or a combined final query):
<<<
{reference_subquestion_sqls}
>>>

The reference steps may stop before the final step that turns them into the answer to the QUESTION (e.g. an outer aggregation, filter, ranking or join over the step results). Operations in GENERATED_SQL that build on the reference steps to answer the QUESTION are the expected final composition, not a divergence. Use the QUESTION only to tell which operations are that final composition; do not score whether GENERATED_SQL answers the QUESTION correctly beyond the reference steps.

Task:
- Infer the core "decomposition steps" expressed by the REFERENCE_SUBQUESTIONS and REFERENCE_SUBQUESTION_SQLS.
- Check whether the GENERATED_SQL implements those steps, either as:
  (a) separate CTEs/subqueries, OR
  (b) merged into fewer steps but still logically equivalent (same filters, groupings, joins, and computed measures), OR
  (c) computed implicitly in a single query (allowed if equivalent).
- Focus on SEMANTIC equivalence of decomposition components, not surface form.

How to compare (important):
For each reference step, extract these semantic elements:
A. Tables involved (and whether they are required by the reference step)
B. Join structure / join keys (e.g., i.uuid = iic.instance_uuid)
C. Filters / predicates (e.g., deleted=0, host='prime-35', availability_zone='flare3', key_name non-empty)
D. Group-by keys and aggregation intent (e.g., count caches per display_name, count distinct uuid, variance(memory_mb))
E. Derived computations (e.g., MAX(memory_mb)-MIN(memory_mb), VARIANCE(memory_mb), LIMIT 10 top-k, MAX(instance_count) by zone)
F. Composition logic across steps (e.g., top-k step feeding into stats step; joining step outputs on display_name)

Then decide whether GENERATED_SQL includes each step’s intent:
- Fully covered: all critical elements A-F present (allow renamed aliases/CTE names).
- Partially covered: some critical elements missing/changed.
- Not covered: the step’s intent is absent or contradicted.

Criticality rules:
- Missing a required FILTER in the reference is a major error (unless the reference includes extra tables/joins that are clearly unnecessary and do not change semantics).
- Missing a required JOIN that affects row cardinality or eligibility is a major error.
- Differences in alias names, whitespace, casing, or minor reordering are NOT errors.
- Extra constraints added to a reference step that would change that step's result are errors.
- Extra tables/joins inside a reference step are acceptable ONLY if they do not change that step's result (i.e., provably redundant) - otherwise treat as divergence.
- The rules above apply to the operations inside each reference step. How the final composition combines the steps (join type or keys between step results, ROLLUP/subtotals, extra output columns, ordering, possible row duplication) is NOT an error unless it drops or contradicts a reference step's filter, grouping, join or computed measure. Do not judge whether the composition is the best or a fully correct way to answer the QUESTION.

Scoring rubric (0–5):
5 = Perfect decomposition fidelity.
    - All reference steps’ intents are covered (even if merged), and the final composition matches.
4 = Mostly faithful.
    - Covers nearly all step intents; at most one minor omission or a non-critical mismatch that likely does not change the final answer.
3 = Partially faithful.
    - Captures the general approach but misses or alters at least one important step element (e.g., a key filter, top-k, distinctness, required join, or key aggregation).
2 = Weak fidelity.
    - Only a small subset of decomposition is reflected; major components are missing or substantially changed.
1 = Minimal fidelity.
    - Barely aligns; only superficial overlap (some same tables/columns) without matching decomposition intent.
0 = No fidelity.
    - Unrelated SQL or entirely fails to reflect the reference decomposition.

Output format (MUST follow exactly):
Return a JSON object with:
- "score": integer from 0 to 5
- "step_coverage": array where each entry has:
    "reference_step_index", "covered" (one of "full"|"partial"|"none"), "notes"
- "overall_rationale": 3-6 sentences explaining the score, citing the biggest mismatches.
Do NOT include any extra keys. Do NOT include markdown. Do NOT include code blocks.

Now evaluate.
"""

def decomposition_prompt(gold_entry, generated_sql):
    return PROMPT_TEMPLATE.format(
        question=gold_entry["question"],
        generated_sql=generated_sql,
        reference_subquestions=json.dumps(gold_entry.get("sub_questions", []), indent=2),
        reference_subquestion_sqls=json.dumps(gold_entry.get("sub_sqls", []), indent=2),
    )


def decomp_score(decomp_output):
    """The judge's 0-5 score as an int, or None if missing or not a valid 0-5 rating."""
    try:
        score = int((decomp_output or {}).get("score"))
    except (TypeError, ValueError):
        return None
    return score if 0 <= score <= 5 else None


SUMMARY_METRICS = [
    "table_retrieval_f1",
    "column_mapping_recall",
    "join_key_f1",
    "domain_knowledge_f1",
    "domain_knowledge_f1_lenient",
    "domain_knowledge_precision",
    "domain_knowledge_recall",
    "query_decomposition_score",
]

def evaluate_single_entry(sql_fn: Path, gold_data, chat_client: GPTChat, output_dir: Path, decomp_only=False):
    result = {
        "sql_fn": str(sql_fn),
        "status": "error",
        "has_decomp": False,
        "generated_sql": "",
        "scores": {},
        "extraction_output": None,
        "decomp_output": None,
    }

    try:
        q_id = sql_fn.name.replace('.sql', '')

        gold_entry = gold_data[q_id]
        reference_sqls = gold_entry.get("sub_sqls", [])
        
        # Reuse the cached LLM outputs, but recompute the scores, so scoring changes apply without new LLM calls.
        # A file written by a --decomp_only run has no extraction: a full run keeps its judge output and extracts.
        output_file = output_dir / f"{q_id}.json"
        cached_decomp = None
        if output_file.exists():
            try:
                cached = read_json(output_file)
                if cached.get("status") == "success" and (decomp_only or cached.get("extraction_output") is not None):
                    cached["scores"] = {
                        **({} if decomp_only else eval_extraction_output(cached["extraction_output"], gold_entry)),
                        "query_decomposition_score": decomp_score(cached.get("decomp_output")),
                    }
                    write_json(cached, output_file)
                    return cached
                if cached.get("status") == "success":
                    cached_decomp = cached.get("decomp_output")
            except Exception as e:
                print(f"File {output_file} corrupted? Re-evaluating. Error: {e}")

        # Read Generated SQL
        with open(sql_fn, "r") as f:
            generated_sql = f.read().strip()

        decomp_output = None
        if not reference_sqls or len(reference_sqls) == 0:
            result["has_decomp"] = False
        else:
            result["has_decomp"] = True
            decomp_output = cached_decomp or chat_client.get_json_response(decomposition_prompt(gold_entry, generated_sql))
            if decomp_score(decomp_output) is None:
                # Not cached, so a rerun retries it instead of scoring an LLM failure
                raise RuntimeError(f"decomposition judge gave no valid 0-5 score: {str(decomp_output)[:200]}")
            result["decomp_output"] = decomp_output

        extraction_output = None
        if not decomp_only:
            extraction_output = chat_client.get_json_response(EXTRACTION_PROMPT.format(sql=generated_sql))
            if extraction_output is None:
                raise RuntimeError("extraction returned no JSON")

        result["generated_sql"] = generated_sql
        result["extraction_output"] = extraction_output
        result["scores"] = {
            **({} if decomp_only else eval_extraction_output(extraction_output, gold_entry)),
            "query_decomposition_score": decomp_score(decomp_output),
        }
        result["status"] = "success"
        print(f"Processed {q_id}: {json.dumps(result['scores'])}")

        write_json(result, output_file)
    except Exception as e:
        print(f"Error processing {sql_fn}: {e}")
        import traceback
        traceback.print_exc()
        result["status"] = f"exception: {str(e)}"

    return result

def run_evaluation_task(sql_fn, gold_data, model, output_dir, decomp_only=False):
    try:
        chat_client = GPTChat(model=model)
        return evaluate_single_entry(sql_fn, gold_data, chat_client, output_dir, decomp_only)
    except Exception as e:
        return {"status": f"init_exception: {e}"}

def main():
    parser = argparse.ArgumentParser(description="Subtask evaluation")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str, default=EVAL_MODEL, help="LLM model to use")
    parser.add_argument("--num_workers", type=int, default=40, help="Number of parallel workers")
    parser.add_argument("--gold_file", type=str, default=None,
                        help="Gold annotations (default: ../data/<dataset>/dev_sampled.json, as the baselines use)")
    parser.add_argument("--output_subdir", type=str, default="subtask_eval_v2",
                        help="Per-question results folder in input_dir; existing results there are reused, so the "
                             "old subtask_eval/ (old prompt and scoring) is left untouched")
    parser.add_argument("--ids_file", type=str, default=None,
                        help="Only evaluate these question ids (one per line); the summary gets a _subset suffix")
    parser.add_argument("--decomp_only", action="store_true",
                        help="Only score query decomposition (skip extraction), e.g. for a second judge model in its "
                             "own --output_subdir")
    args = parser.parse_args()

    gold_file = args.gold_file or f'../data/{args.dataset}/dev_sampled.json'
    gold_data = read_json(gold_file)
    gold_data = {q['id']: q for q in gold_data}
    
    generated_sql_dir = Path(args.input_dir) / 'generated'
    generated_sql_fns = sorted(generated_sql_dir.glob('*.sql'))
    if args.ids_file:
        ids = set(open(args.ids_file).read().split())
        generated_sql_fns = [fn for fn in generated_sql_fns if fn.stem in ids]
    print(f"Found {len(generated_sql_fns)} generated SQLs to evaluate")
    
    output_dir = Path(args.input_dir) / args.output_subdir
    output_dir.mkdir(exist_ok=True)

    results = []
    
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(generated_sql_fns))
    except ImportError:
        pbar = None
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(run_evaluation_task, generated_sql_fn, gold_data, args.model, output_dir, args.decomp_only)
            for generated_sql_fn in generated_sql_fns
        ]
        for f in futures:
            res = f.result()
            results.append(res)
            if pbar: pbar.update(1)

    if pbar: pbar.close()
    
    valid_results = [r for r in results if r["status"] == "success"]
    valid_results_with_decomp = [r for r in results if r["status"] == "success" and r["has_decomp"]]

    # Each metric is averaged only over records where it is not None (non-empty gold set for that subtask)
    avg_scores, counts = {}, {}
    for metric in SUMMARY_METRICS:
        values = [r["scores"][metric] for r in valid_results if r["scores"].get(metric) is not None]
        avg_scores[f"avg_{metric}"] = sum(values) / len(values) if values else None
        counts[f"n_{metric}"] = len(values)
    if avg_scores["avg_query_decomposition_score"] is not None:
        # the original decomp score is 0-5, divide by 5 to normalize
        avg_scores["avg_query_decomposition_score"] /= 5

    summary = {
        "total_questions": len(results),
        "total_valid_questions": len(valid_results),
        "total_valid_questions_with_query_decomposition": len(valid_results_with_decomp),
        **avg_scores,
        **counts,
    }

    summary_path = Path(args.input_dir) / f"summary_{args.output_subdir}{'_subset' if args.ids_file else ''}.json"
    write_json(summary, summary_path)

    print(f"\nEvaluation Complete.")
    print(f"Summary saved to: {summary_path}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
