# import debugpy; debugpy.connect(("127.0.0.1", 5690))
import argparse
import os
import json
import openai
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from multiprocessing import Pool, set_start_method

from llm.chatgpt import init_chatgpt, ask_llm, cost, llm_client
from utils.enums import LLM
from utils.post_process import process_duplication, get_sqls


def process_batch(batch, submit_folder, db_ids, args, i, 
    openai_api_key, openai_group_id, model):
    """Function to process each batch in parallel."""

    # Setup logging for this batch
    instance_id = batch['instance_id'][0]
    instance_dir = os.path.join(submit_folder, instance_id)
    os.makedirs(instance_dir, exist_ok=True)
    log_file = os.path.join(instance_dir, "generate.log")
    
    logger = logging.getLogger(instance_id)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    logger.info("="*80)
    logger.info(f"Processing question: {instance_id}")
    logger.info("="*80)

    prompt = batch["prompt"][0]
    logger.info(f"[Input Prompt]\n{prompt}\n[End Input Prompt]")

    # init openai api
    init_chatgpt(args.openai_api_key, args.openai_group_id, args.model)

    if args.post_mode == 'consistency-from-generated-pass@n':  # load the saved sql from the output of pass@n
        cur_db_ids = db_ids[i * args.batch_size: (i+1) * args.batch_size]
        results = []
        for db_id, instance_id in zip(cur_db_ids, batch["instance_id"]):
            result = {
                'db_id': db_id,
                'p_sqls': [],
                'instance_id': instance_id
            }
            for n in range(args.n):
                sql_file_path = os.path.join(submit_folder, f"{instance_id}@{n}.sql")
                with open(sql_file_path, "r") as sql_file:
                    sql_content = sql_file.read().strip()
                    result['p_sqls'].append(sql_content)
                    logger.info(f"[Loaded SQL Candidate {n}]\n{sql_content}\n[End SQL Candidate {n}]")

            final_sql = get_sqls(result, args.n, args.db_dir, instance_id)
            logger.info(f"[Final SQL]\n{final_sql}\n[End Final SQL]")
            with open(os.path.join(submit_folder, f"{instance_id}.sql"), "w") as submit_file:
                submit_file.write(final_sql)
        return

    # Token usage of the LLM call (eval/utils/cost.py); a rerun replaces cost.json
    recorder = cost.start_recording()
    try:
        logger.info("[Requesting LLM Response]")
        res = ask_llm(args.model, batch["prompt"], args.temperature, args.n, args.max_tokens)
        logger.info(f"[Received LLM Response] - {len(res.get('response', [[]])[0])} SQL candidates")
    except openai.error.InvalidRequestError:
        logger.error(f"The {i}-th question has too much tokens! Return \"SELECT\" instead")
        print(f"The {i}-th question has too much tokens! Return \"SELECT\" instead")
        res = {"response": [["SELECT" for _ in range(args.n)]]}  # hard-code for batch_size=1
    finally:
        recorder.write(os.path.join(instance_dir, "cost.json"))

    if args.n == 1: 
        assert len(res["response"]) == args.batch_size == 1
        sql = res["response"][0][0]
        instance_id = batch["instance_id"][0]
        logger.info(f"[Raw SQL Response]\n{sql}\n[End Raw SQL Response]")
        sql = " ".join(sql.replace("\n", " ").split())
        sql = process_duplication(sql)
        logger.info(f"[Processed SQL]\n{sql}\n[End Processed SQL]")
        with open(os.path.join(submit_folder, f"{instance_id}@0.sql"), "w") as submit_file:
            submit_file.write(sql)
    else:
        results = []
        cur_db_ids = db_ids[i * args.batch_size: (i+1) * args.batch_size]
        for sqls, db_id, instance_id in zip(res["response"], cur_db_ids, batch["instance_id"]):  # dummy loop, only excute once
            logger.info(f"[Generated {len(sqls)} SQL Candidates]")
            processed_sqls = []
            for candidate_idx, sql in enumerate(sqls):
                logger.info(f"[Raw SQL Candidate {candidate_idx}]\n{sql}\n[End Raw SQL Candidate {candidate_idx}]")
                sql = " ".join(sql.replace("\n", " ").split())
                sql = process_duplication(sql)
                logger.info(f"[Processed SQL Candidate {candidate_idx}]\n{sql}\n[End Processed SQL Candidate {candidate_idx}]")

                processed_sqls.append(sql)
            result = {
                'db_id': db_id,
                'p_sqls': processed_sqls,
                'instance_id': instance_id
            }

            if args.post_mode == 'pass@n':
                for n in range(args.n):
                    file_name = f"{instance_id}@{n}.sql"
                    file_path = os.path.join(submit_folder, file_name)
                    with open(file_path, "w") as submit_file:
                        submit_file.write(processed_sqls[n])
                    logger.info(f"[Saved SQL {n} to: {file_path}]")
            elif args.post_mode == 'consistency@n':  # exec_result based consistency
                final_sql = get_sqls(result, args.n, args.db_dir, instance_id)
                logger.info(f"[Final SQL from consistency voting]\n{final_sql}\n[End Final SQL]")
                with open(os.path.join(submit_folder, f"{instance_id}.sql"), "w") as submit_file:
                    submit_file.write(final_sql)
            else:
                raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str)
    parser.add_argument("--results_dir", type=str, default=None, help="Folder for this run's results (default: <question>/RESULTS_MODEL-<model>-SQL)")
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--openai_group_id", type=str, default=None)
    parser.add_argument("--model", type=str, default=LLM.GPT_35_TURBO)

    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--db_dir", type=str, default="../../resource/databases/spider2-localdb")
    parser.add_argument('--max_tokens', type=int, default=1000)
    parser.add_argument('--post_mode', type=str, choices=['pass@n', 'consistency@n', 'consistency-from-generated-pass@n', None], default=None)
    parser.add_argument("--is_sql_debug", action="store_true", default=False)
    parser.add_argument("--processes", type=int, default=16)  # New argument for specifying the number of processes
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()


    if args.is_sql_debug:
        QUESTION_FILE = "debug_questions.json"
    else:
        QUESTION_FILE = "questions.json"

    # check args
    assert args.model in LLM.BATCH_FORWARD or \
           args.model not in LLM.BATCH_FORWARD and args.batch_size == 1, \
        f"{args.model} doesn't support batch_size > 1"

    # load ids that already predicted
    DEBUG_PREFIX = "SQL_DEBUG_" if args.is_sql_debug else ""
    submit_folder = args.results_dir or os.path.join(args.question, f'{DEBUG_PREFIX}RESULTS_MODEL-{args.model}-SQL')
    os.makedirs(submit_folder, exist_ok=True)

    if args.override:
        pred_ids = set()
    else:
        pred_ids = [file.split(".")[0].split("@")[0] for file in os.listdir(submit_folder) if file.endswith(".sql")]
        pred_ids = set(pred_ids)
    questions_json = json.load(open(os.path.join(args.question, QUESTION_FILE), "r"))
    questions = [{"prompt": item["prompt"], "instance_id": item.get("id", item.get("instance_id"))} for item in questions_json["questions"] \
        if item.get("id", item.get("instance_id")) not in pred_ids]
    db_ids = [item.get("db", item.get("db_id")) for item in questions_json["questions"] \
        if item.get("id", item.get("instance_id")) not in pred_ids]

    question_loader = DataLoader(questions, batch_size=args.batch_size, shuffle=False, drop_last=False)

    set_start_method('spawn', force=True)  # Ensures that the correct start method is used for multiprocessing

    token_cnt = 0
    # Each worker process paces its own requests
    llm_client.split_rpm_across_processes(args.processes)
    with Pool(processes=args.processes) as pool:
        with tqdm(total=len(question_loader)) as pbar:
            for _ in pool.starmap(process_batch, [
                (
                    batch, submit_folder, db_ids, args, i, 
                    args.openai_api_key, args.openai_group_id, args.model
                ) for i, batch in enumerate(question_loader)
            ]):
                pbar.update(1)

    cost.write_summary(submit_folder)
