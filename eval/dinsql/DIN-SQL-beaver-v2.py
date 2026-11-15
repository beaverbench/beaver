# Modified DIN-SQL for Beaver dataset with organized per-question outputs
# Creates subfolders for each question with logs, SQL, and results

import pandas as pd
import time
import openai
import os
import sys
from tqdm import tqdm
import json
from multiprocessing import Pool, set_start_method
from pathlib import Path
import logging
import numpy as np

# Import the original DIN-SQL functions
from importlib.machinery import SourceFileLoader
dinsql = SourceFileLoader("dinsql", "DIN-SQL.py").load_module()

# Use prompts and helper functions from original DIN-SQL
schema_linking_prompt = dinsql.schema_linking_prompt
classification_prompt = dinsql.classification_prompt
easy_prompt = dinsql.easy_prompt
medium_prompt = dinsql.medium_prompt
hard_prompt = dinsql.hard_prompt

hard_prompt_maker = dinsql.hard_prompt_maker
medium_prompt_maker = dinsql.medium_prompt_maker
easy_prompt_maker = dinsql.easy_prompt_maker
classification_prompt_maker = dinsql.classification_prompt_maker
schema_linking_prompt_maker = dinsql.schema_linking_prompt_maker
find_fields_MYSQL_like = dinsql.find_fields_MYSQL_like
find_primary_keys_MYSQL_like = dinsql.find_primary_keys_MYSQL_like
find_foreign_keys_MYSQL_like = dinsql.find_foreign_keys_MYSQL_like
debuger = dinsql.debuger
GPT4o_generation = dinsql.GPT4o_generation
GPT4o_debug = dinsql.GPT4o_debug
cost = dinsql.cost  # eval/utils/cost.py, made importable by DIN-SQL.py


def creating_schema_preprocessed(DATASET_JSON):
    """Load preprocessed schema directly (already in DINSQL format)"""
    with open(DATASET_JSON, 'r') as f:
        schema_list = json.load(f)
    
    schema = []
    f_keys = []
    p_keys = []
    
    for db_schema in schema_list:
        db_id = db_schema['db']
        tables = db_schema['table_names_original']
        col_names = db_schema['column_names']
        col_types = db_schema['column_types']
        foreign_keys = db_schema['foreign_keys']
        primary_keys = db_schema['primary_keys']
        
        for col, col_type in zip(col_names, col_types):
            table_idx, col_name = col
            if table_idx == -1:
                for table in tables:
                    schema.append([db_id, table, '*', 'text'])
            else:
                schema.append([db_id, tables[table_idx], col_name, col_type])
        
        for pk_idx in primary_keys:
            table_idx, col_name = col_names[pk_idx]
            p_keys.append([db_id, tables[table_idx], col_name])
        
        for fk_pair in foreign_keys:
            first_idx, second_idx = fk_pair
            first_table_idx, first_col = col_names[first_idx]
            second_table_idx, second_col = col_names[second_idx]
            f_keys.append([db_id, tables[first_table_idx], tables[second_table_idx], first_col, second_col])
    
    spider_schema = pd.DataFrame(schema, columns=['Database name', ' Table Name', ' Field Name', ' Type'])
    spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    
    return spider_schema, spider_primary, spider_foreign


def process_single_question(index, row, args, output_base_dir, spider_schema, spider_primary, spider_foreign):
    """Process a single question and save outputs to organized subfolder"""
    question_id = row['id']
    question_dir = Path(output_base_dir) / question_id
    question_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging for this question
    log_file = question_dir / 'generation.log'
    logger = logging.getLogger(question_id)
    logger.setLevel(logging.INFO)
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # Token usage of this question's LLM calls; a rerun replaces its cost.json
    recorder = cost.start_recording()

    logger.info("="*80)
    logger.info(f"Processing question: {question_id}")
    logger.info(f"Question text: {row['question']}")
    logger.info(f"Database: {row['db']}")
    logger.info("="*80)
    
    try:
        # Step 1: Schema Linking
        logger.info("\n[STEP 1: Schema Linking]")
        schema_links = None
        join_keys = row.get('join_keys')
        gold_tables = row.get('tables')
        if join_keys:
             logger.info(f"[Using Join Keys]\n{join_keys}")

        while schema_links is None:
            try:
                schema_linking_input = schema_linking_prompt_maker(row['question'], row['db'], spider_schema, spider_foreign, join_keys=join_keys, table_filter=gold_tables)
                logger.info(f"[Input Prompt]\n{schema_linking_input}\n[End Input Prompt]")
                res = GPT4o_generation(schema_linking_input, model=args.model)
                assert len(res['response']) == 1
                schema_links = res['response'][0]
                logger.info(f"[LLM Response]\n{schema_links}\n[End LLM Response]")
            except Exception as e:
                logger.error(f"Schema linking error: {e}")
                time.sleep(3)
        
        try:
            schema_links = schema_links.split("Schema_links: ")[1]
            logger.info(f"[Extracted Schema Links]\n{schema_links}\n[End Extracted Schema Links]")
        except:
            logger.warning("Slicing error for schema_linking module")
            schema_links = "[]"
        
        # Step 2: Classification
        logger.info("\n[STEP 2: Query Classification]")
        classification = None
        while classification is None:
            try:
                classification_input = classification_prompt_maker(row['question'], row['db'], schema_links[1:], spider_schema, spider_foreign, join_keys=join_keys, table_filter=gold_tables)
                logger.info(f"[Input Prompt]\n{classification_input}\n[End Input Prompt]")
                res = GPT4o_generation(classification_input, model=args.model)
                assert len(res['response']) == 1
                classification = res['response'][0]
                logger.info(f"[LLM Response]\n{classification}\n[End LLM Response]")
            except Exception as e:
                logger.error(f"Classification error: {e}")
                time.sleep(3)
        
        try:
            predicted_class = classification.split("Label: ")[1]
            logger.info(f"[Predicted Query Type]\n{predicted_class}\n[End Predicted Query Type]")
        except:
            logger.warning("Slicing error for classification module")
            predicted_class = '"NESTED"'
        
        # Step 3: SQL Generation
        logger.info("\n[STEP 3: SQL Generation]")
        try:
            sub_questions = classification.split('questions = ["')[1].split('"]')[0]
            flag = 'NESTED'
            logger.info(f"[Query Type: NESTED]")
            logger.info(f"[Sub-questions]\n{sub_questions}\n[End Sub-questions]")
        except Exception as e:
            logger.info(f'[Query Type: NON-NESTED] (Error: {e})')
            flag = 'NON-NESTED'
        
        SQL_list = None
        while SQL_list is None:
            try:
                if flag == 'NESTED':
                    sql_input = hard_prompt_maker(row, schema_links, sub_questions, spider_schema, spider_foreign, args, table_filter=gold_tables)
                    logger.info(f"[Input Prompt for Hard Query]\n{sql_input}\n[End Input Prompt]")
                    res = GPT4o_generation(sql_input, n=args.n, temperature=args.temperature, model=args.model)
                    SQL_list = res['response']
                else:
                    sql_input = medium_prompt_maker(row, schema_links, spider_schema, spider_foreign, args, table_filter=gold_tables)
                    logger.info(f"[Input Prompt for Medium Query]\n{sql_input}\n[End Input Prompt]")
                    res = GPT4o_generation(sql_input, n=args.n, temperature=args.temperature, model=args.model)
                    SQL_list = res['response']
                logger.info(f"[Generated {len(SQL_list)} SQL Candidates]")
                for i, sql in enumerate(SQL_list):
                    logger.info(f"[SQL Candidate {i}]\n{sql}\n[End SQL Candidate {i}]")
            except Exception as e:
                logger.error(f"SQL generation error: {e}")
                time.sleep(3)
        
        # Step 4: SQL Debugging and saving
        logger.info("\n[STEP 4: SQL Debugging and Validation]")
        for idx, sql in enumerate(SQL_list):
            try:
                sql = sql.split("SQL:")[1]
                sql = sql.replace('```sql', '').replace('```', '').strip()
                logger.info(f"[Raw SQL Candidate {idx}]\n{sql}\n[End Raw SQL Candidate {idx}]")
            except:
                logger.error(f"SQL slicing error for index {idx}")
                sql = "SELECT"
            
            debugged_SQL = None
            while debugged_SQL is None:
                try:
                    debug_input = debuger(row['question'], row['db'], sql, spider_schema, spider_primary, spider_foreign, join_keys=join_keys, table_filter=gold_tables)
                    logger.info(f"[Debug Input for SQL {idx}]\n{debug_input}\n[End Debug Input for SQL {idx}]")
                    res = GPT4o_debug(debug_input, model=args.model)
                    assert len(res['response']) == 1
                    debugged_SQL = res['response'][0]
                    logger.info(f"[Debug Output for SQL {idx}]\n{debugged_SQL}\n[End Debug Output for SQL {idx}]")
                except Exception as e:
                    logger.error(f"SQL debugging error for SQL {idx}: {e}")
                    time.sleep(3)
            
            debugged_SQL = debugged_SQL.replace("\n", " ").replace('```sql', '').replace('```', '').strip()
            
            # Save SQL file
            sql_file = question_dir / f'predicted_{idx}.sql'
            with open(sql_file, 'w') as f:
                f.write(debugged_SQL)
            logger.info(f"[Final SQL {idx}]\n{debugged_SQL}\n[Saved to: {sql_file}]")
        
        logger.info("="*80)
        logger.info(f"Completed processing for {question_id}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Fatal error processing {question_id}: {e}", exc_info=True)
    finally:
        recorder.write(str(question_dir / 'cost.json'))
        # Remove handler to avoid duplicates
        logger.removeHandler(handler)
        handler.close()
    
    return index


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process dataset and output file.")
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument('--processes', type=int, default=5)
    parser.add_argument('--override', action='store_true')
    parser.add_argument("--use_special_function", action="store_true", default=False)
    parser.add_argument("--use_plan", action="store_true", default=False)
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")

    args = parser.parse_args()

    DATASET_SCHEMA = f"./preprocessed_data/{args.dev}/tables_preprocessed.json"
    DATASET = f"./preprocessed_data/{args.dev}/{args.dev}_preprocessed.json"

    openai.api_key = os.environ.get("OPENAI_API_KEY")

    set_start_method('spawn')
    s_time = time.time()
    
    # Load schema and questions
    spider_schema, spider_primary, spider_foreign = creating_schema_preprocessed(DATASET_SCHEMA)
    val_df = pd.read_json(DATASET)
    # by default, pd will put nan for missing keys, we want it to be None
    val_df = val_df.replace({np.nan: None})
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f'./output/{args.dev}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which questions to process
    if args.override:
        pred_ids = set()
    else:
        # Check which questions already have outputs
        pred_ids = set()
        output_path = Path(output_dir)
        if output_path.exists():
            for qdir in output_path.iterdir():
                if qdir.is_dir() and (qdir / 'predicted_0.sql').exists():
                    pred_ids.add(qdir.name)
    
    # Each worker process paces its own requests
    dinsql.llm_client.split_rpm_across_processes(args.processes)
    with Pool(processes=args.processes) as pool:
        with tqdm(total=val_df.shape[0]) as pbar:
            inputs = [(index, row, args, output_dir, spider_schema, spider_primary, spider_foreign) 
                     for index, row in val_df.iterrows() if row['id'] not in pred_ids]
            print(f"Number of data samples to predict: {len(inputs)}")
            for _ in pool.starmap(process_single_question, inputs):
                pbar.update(1)

    print(f"Total time: {time.time() - s_time} seconds")
    print(f"Output directory: {output_dir}")
    cost.write_summary(output_dir)
