import os
import time
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, APIConnectionError, APITimeoutError, InternalServerError

from prompt import get_ete_prompts, EvalConfig

# Sampling from the official XiYanSQL-QwenCoder vLLM example
TEMPERATURE = 0.1
MAX_TOKENS = 1024
MAX_RETRIES = 3


def process_output(output):
    # The prompt ends with an open ```sql fence, so the SQL runs up to the closing ```
    if "```sql" in output:
        output = output.split("```sql")[-1]
    if "```" in output:
        output = output.split("```")[0]
    return output.strip()


def run_prompts(prompts, instance_ids, output_dir: str, model: str, num_workers: int):
    todo = []
    for prompt, instance_id in zip(prompts, instance_ids):
        sql_file = os.path.join(output_dir, instance_id, "predicted_0.sql")
        # Skip already processed instances (empty files from failed requests are retried)
        if not os.path.exists(sql_file) or os.path.getsize(sql_file) == 0:
            todo.append((prompt, instance_id))

    if not todo:
        print("All prompts already processed.")
        return

    print(f"Running {len(todo)} prompts...")

    base_url = os.environ.get("LLM_BASE_URL")
    if not base_url:
        raise SystemExit("LLM_BASE_URL is not set; point it at the vLLM server, e.g. http://localhost:8000/v1")
    client = OpenAI(base_url=base_url, api_key=os.environ.get("LLM_API_KEY") or "EMPTY", max_retries=0)

    def process_prompt(item):
        prompt, instance_id = item
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                content = resp.choices[0].message.content or ""
                return instance_id, content, f"Prompt:\n{prompt[0]['content']}\n\nResponse:\n{content}"
            except (APIConnectionError, APITimeoutError, InternalServerError) as e:
                print(f"Transient error on {instance_id} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                time.sleep(5 * 2 ** attempt)
            except Exception as e:
                # e.g. 400 when the prompt exceeds --max-model-len; not worth retrying
                print(f"Error processing {instance_id}: {e}")
                return instance_id, "", f"Prompt:\n{prompt[0]['content']}\n\nError: {e}"
        return instance_id, "", f"Prompt:\n{prompt[0]['content']}\n\nError: retries exhausted"

    def save(result):
        instance_id, content, gen_log = result
        instance_dir = os.path.join(output_dir, instance_id)
        os.makedirs(instance_dir, exist_ok=True)
        with open(os.path.join(instance_dir, "predicted_0.sql"), "w") as f:
            f.write(process_output(content))
        with open(os.path.join(instance_dir, "generation.log"), "w") as f:
            f.write(gen_log)
        return content != ""

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Save as results arrive so an interrupted run can resume
        futures = [executor.submit(process_prompt, item) for item in todo]
        ok = sum(save(f.result()) for f in tqdm(as_completed(futures), total=len(futures)))

    if ok < len(todo):
        print(f"{len(todo) - ok}/{len(todo)} prompts failed; rerun with the same --timestamp to retry them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xiyansql-32b-2504",
                        help="Model name served by vLLM (its --served-model-name)")
    parser.add_argument("--dataset", type=str, default="dw")
    parser.add_argument("--q_fn", type=str, default="dev")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="../../data")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--gold_tables", action="store_true")
    parser.add_argument("--join_keys", action="store_true")
    parser.add_argument("--mapping", action="store_true")
    parser.add_argument("--knowledge", action="store_true")
    parser.add_argument("--decomp", action="store_true")
    args = parser.parse_args()

    eval_config = EvalConfig(
        gold_tables=args.gold_tables, join_keys=args.join_keys, mapping=args.mapping,
        knowledge=args.knowledge, decomp=args.decomp,
    )
    print(eval_config)

    prompts, instance_ids = get_ete_prompts(args.dataset, args.q_fn, eval_config, args.data_dir)

    print(f"Output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    run_prompts(prompts, instance_ids, args.output_dir, args.model, args.num_workers)
