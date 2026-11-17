import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from prompt import get_ete_prompts
from utils import EvalConfig

import argparse
import sys

# eval/utils is appended, not prepended, so this folder's utils.py still wins the `utils` name
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
import cost
import llm_client

MODEL_MAP = {
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5.2": "gpt-5.2",
    "o4-mini": "o4-mini",
    "gemini": "gemini-2.0-flash",
    "qwen": "qwen/qwen3-next-80b-a3b-instruct",
    "claude": "anthropic/claude-sonnet-4.5",
    "minimax": "minimax/minimax-m2.1",
}


def process_output(output):
    # remove everything of left of <ans> if <ans> is in output
    if "<ans>" in output:
        output = output.split("<ans>")[-1]
    # remove everything of right of </ans> if </ans> is in output
    if "</ans>" in output:
        output = output.split("</ans>")[0]
    # remove everything of left of ```sql if ```sql is in output
    if "```sql" in output:
        output = output.split("```sql")[-1]
    # remove everything of right of ``` if ``` is in output
    if "```" in output:
        output = output.split("```")[0]
    return output.strip()




def run_prompts(prompts, indices, instance_ids, output_dir: str, model: str, num_workers: int = 20):
    prompts_to_process = []
    indices_to_process = []
    instance_ids_to_process = []
    
    for prompt, idx, instance_id in zip(prompts, indices, instance_ids):
        instance_dir = os.path.join(output_dir, instance_id)
        sql_file = os.path.join(instance_dir, "predicted_0.sql")
        
        # Check if already processed
        if not os.path.exists(sql_file) or os.path.getsize(sql_file) == 0:
            prompts_to_process.append(prompt)
            indices_to_process.append(idx)
            instance_ids_to_process.append(instance_id)
            
    if not prompts_to_process:
        print("All prompts already processed.")
        return

    print(f"Running {len(prompts_to_process)} prompts...")
    
    model_name = MODEL_MAP.get(model, model)
    # LLM_BASE_URL, else Gemini / OpenRouter ("/" in the name) / OpenAI; requests are paced and their usage recorded
    client = llm_client.LLMClient(llm_client.make_client(model_name), step="generate")

    def process_prompt(item):
        prompt, idx, instance_id = item
        if prompt is None:
            return

        # Token usage of this question's call (eval/utils/cost.py); a rerun replaces its cost.json
        recorder = cost.start_recording()
        try:
            # Reasoning is turned off for Qwen and Claude on OpenRouter; a custom endpoint gets a plain request
            kwargs = {}
            if not llm_client.custom_endpoint() and ("qwen" in model_name or "claude" in model_name):
                kwargs["extra_body"] = {"reasoning": {"effort": "none"}}
            resp = client.chat.completions.create(
                model=model_name,
                messages=prompt,
                **kwargs,
            )
            content = resp.choices[0].message.content
            # To match the structure of other baselines, we could log the prompt/response
            # Here we provide a simple generic generation log.
            gen_log = f"Prompt:\n{prompt}\n\nResponse:\n{content}"
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            content, gen_log = "", f"Error: {e}"

        # Saved as each question finishes, so an interrupted run can resume
        instance_dir = os.path.join(output_dir, instance_id)
        os.makedirs(instance_dir, exist_ok=True)
        with open(os.path.join(instance_dir, "predicted_0.sql"), "w") as f:
            f.write(process_output(content or ""))
        with open(os.path.join(instance_dir, "generation.log"), "w") as f:
            f.write(gen_log)
        recorder.write(os.path.join(instance_dir, "cost.json"))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        items = list(zip(prompts_to_process, indices_to_process, instance_ids_to_process))
        list(tqdm(executor.map(process_prompt, items), total=len(items)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--dataset", type=str, default="dw")
    parser.add_argument("--q_fn", type=str, default="dev")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="../../data")
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--gold_tables", action="store_true")
    parser.add_argument("--join_keys", action="store_true")
    parser.add_argument("--mapping", action="store_true")
    parser.add_argument("--knowledge", action="store_true")
    parser.add_argument("--decomp", action="store_true")
    args = parser.parse_args()

    model = args.model
    dataset = args.dataset
    q_fn = args.q_fn

    eval_config = EvalConfig(
        gold_tables=args.gold_tables, join_keys=args.join_keys, mapping=args.mapping, knowledge=args.knowledge, decomp=args.decomp
    )
    print(eval_config)

    prompts, instance_ids = get_ete_prompts(dataset, q_fn, eval_config, args.data_dir)
    
    output_dir = args.output_dir

    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    selected_indices = list(range(len(prompts)))
    selected_prompts = [prompts[i] for i in selected_indices]

    run_prompts(selected_prompts, selected_indices, instance_ids, output_dir, model, args.num_workers)
    cost.write_summary(output_dir)
