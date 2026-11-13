import os
import pandas as pd
import json
import logging
import math
import re
import sqlglot
from sqlglot.expressions import Table, Column, CTE
import numpy as np
import sqlparse

def extract_all_blocks(main_content, code_format):
    sql_blocks = []
    start = 0
    
    while True:

        sql_query_start = main_content.find(f"```{code_format}", start)
        if sql_query_start == -1:
            break
        

        sql_query_end = main_content.find("```", sql_query_start + len(f"```{code_format}"))
        if sql_query_end == -1:
            break 

        sql_block = main_content[sql_query_start + len(f"```{code_format}"):sql_query_end].strip()
        sql_blocks.append(sql_block)

        start = sql_query_end + len("```")
    
    return sql_blocks

def hard_cut(str_e, length=0):
    if length:
        if len(str_e) > length:
            str_e = str_e[:int(length)]+"\n"
    return str_e

def get_values_from_table(csv_data_str):
    return '\n'.join(csv_data_str.split('\n')[1:])

def search_file(directory, target_file):
    result = []
    for root, dirs, files in os.walk(directory):
        if target_file in files:
            result.append(os.path.join(root, target_file))
    return result

def get_longest(sql_list):
    sql_list_len = [len(i) for i in sql_list]
    sql_list_len_index = sql_list_len.index(max(sql_list_len))
    return sql_list[sql_list_len_index]

def get_shortest(sql_list):
    sql_list_len = [len(i) for i in sql_list]
    sql_list_len_index = sql_list_len.index(min(sql_list_len))
    return sql_list[sql_list_len_index]

import threading
def initialize_logger(log_path, logger_name=None):
    if logger_name is None:
        logger_name = threading.current_thread().name
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path, mode='w')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    # Close any existing handlers before replacing them to avoid leaking file descriptors
    for handler in logger.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)
    logger.addHandler(file_handler)
    return logger

def close_logger(logger):
    """Close and detach all handlers of a logger to release file descriptors."""
    if logger is None:
        return
    for handler in logger.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)

def extract_between(file_path, start_str, end_str):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        if not content:
            pass
    
    results = []
    start_index = 0

    while True:
        start_index = content.find(start_str, start_index)
        if start_index == -1:
            break
        start_index += len(start_str)
        end_index = content.find(end_str, start_index)
        if end_index == -1:
            break
        results.append(content[start_index:end_index])
        start_index = end_index + len(end_str)
    
    return results


def compare_pandas_table(pred, gold, condition_cols=[], ignore_order=False, tolerance=0.001):
    """_summary_

    Args:
        pred (Dataframe): _description_
        gold (Dataframe): _description_
        condition_cols (list, optional): _description_. Defaults to [].
        ignore_order (bool, optional): _description_. Defaults to True.

    """
    # print('condition_cols', condition_cols)

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        if ignore_order_:
            v1, v2 = (sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                    sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))))
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True
    
    if condition_cols != []:
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold
    pred_cols = pred

    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()
    score = 1
    for _, gold in enumerate(t_gold_list):
        if not any(vectors_match(gold, pred, ignore_order_=ignore_order) for pred in t_pred_list):
            score = 0
        else:
            for j, pred in enumerate(t_pred_list):
                if vectors_match(gold, pred, ignore_order_=ignore_order):
                    break

    return score

def clear_description(table_info):
    return re.sub(r"Description:[^\n]*", "", table_info)

def get_table_info(test_path, sql_data, api, clear_des=False, full_tb_info=None):
    if full_tb_info:
        return full_tb_info[sql_data]
    else:
        table_info_txt = ["prompts.txt"]      
        table_info = ''
        for txt in table_info_txt:
            txt_path = search_file(os.path.join(test_path, sql_data), txt)
            for path in txt_path:
                with open(path) as f:
                    table_info += f.read()
        if clear_des:
            if len(table_info) > 200000:
                table_info = clear_description(table_info)
        return table_info

def get_api_name(sql_data):
    if sql_data.startswith("sf"):
        return "snowflake"
    elif sql_data.startswith("local"):
        return "sqlite"
    elif sql_data.startswith("bq") or sql_data.startswith("ga"):
        return "bigquery"
    elif any(sql_data.startswith(prefix) for prefix in ["beaver", "dw", "sp", "nova", "neutron"]):
        return "mysql"
    else:
        raise NotImplementedError("Invalid file name.")

def remove_digits(s):
    return re.sub(r'\d', '', s)

def is_file(filepath, suffix):
    return os.path.isfile(filepath) and filepath.lower().endswith(suffix)

def matching_at_same_position(s1, s2):
    min_length = min(len(s1), len(s2))
    matches = [s1[i] for i in range(min_length) if s1[i] == s2[i]]
    return "".join(matches)

def get_dictionary(db_path, task):
    json_path = os.path.join(db_path, f"spider2-{task}.jsonl")
    task_dict = {}
    with open(json_path) as f:
        for line in f:
            line_js = json.loads(line)
            if task == "snow":
                task_dict[line_js['id']] = line_js['instruction']
                # if not line_js['id'].startswith("sf"):
                #     line_js['id'] = "sf_"+line_js['id']
                # task_dict[line_js['id']] = line_js['question']
            elif task == "lite":
                task_dict[line_js['id']] = line_js['question']

    dictionaries = [entry for entry in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, entry))]
    return dictionaries, task_dict

def get_db_id(db_path, ex_id):
    task = "lite"
    assert ex_id.startswith("local")
    json_path = os.path.join(db_path, f"spider2-{task}.jsonl")
    with open(json_path) as f:
        for line in f:
            line_js = json.loads(line)
            if line_js['id'] == ex_id:
                return line_js["db"]


def get_sqlite_path(db_path="", sql_data=None, db_id=None, task=None):
    if db_id:
        if task == "lite":
            local_db_pth = "../../spider2-lite/resource/databases/spider2-localdb"
            return os.path.join(local_db_pth, db_id+".sqlite")
        elif task == "BIRD":
            local_db_pth = "../../data/BIRD/dev_databases"
            return os.path.join(local_db_pth, db_id, db_id+".sqlite")
        elif task == "spider":
            local_db_pth = "../../data/spider/test_database"
            return os.path.join(local_db_pth, db_id, db_id+".sqlite")
    if not db_path:
        return ""
    sql_data_path = os.path.join(db_path, sql_data)
    sqlite_path = None
    for sqlite in os.listdir(sql_data_path):
        if sqlite.endswith(".sqlite"):
            sqlite_path = os.path.join(sql_data_path, sqlite)
    return sqlite_path

def split_sql(sql: str):
    return [stmt.strip() for stmt in sqlparse.split(sql) if stmt.strip()]

def clear_sample_rows(text, byte_limit=1000):
    pattern = re.compile(r"(Sample rows:\s*)(.*?)(\n-{10,}\n)", re.DOTALL)

    def trim_block(match):
        prefix = match.group(1)
        content = match.group(2).strip()
        suffix = match.group(3)

        try:
            data = json.loads(content)
            if isinstance(data, list):
                for row in data:
                    for k, v in row.items():
                        if isinstance(v, str):
                            v_bytes = v.encode("utf-8")
                            if len(v_bytes) > byte_limit:
                                row[k] = v_bytes[:1000].decode("utf-8", errors="ignore")
            trimmed_json = json.dumps(data, ensure_ascii=False, indent=2)
            return prefix + trimmed_json + "\n" + suffix
        except Exception as e:
            return prefix + content[:byte_limit*10] + suffix

    return pattern.sub(trim_block, text)

def get_tb_info(text):
    tb = []
    for i in text.split("-"*50):
        i = i.strip()
        if i.startswith("Table full name:"):
            tb.append(i)
    return tb

def get_external(text):
    if "External knowledge that might be helpful: " in text:
        return text[text.find("External knowledge that might be helpful: "):text.find("The table structure information is")]
    else:
        return ""

def compute_precision_recall(predicted: set, ground_truth: set):
    if not predicted:
        precision = 0.0
    else:
        precision = len(predicted & ground_truth) / len(predicted)

    if not ground_truth:
        recall = 0.0
    else:
        recall = len(predicted & ground_truth) / len(ground_truth)

    return precision, recall

def digit_entropy_ratio(s: str) -> float:
    if not s:
        return 0.0
    s = s.replace(" ", "")
    digit_count = sum(c.isdigit() for c in s)
    return 1.0 - digit_count / len(s)

def extract_column_names(sql: str) -> list[str]:
    column_names = []
    for line in sql.splitlines():
        line = line.strip()
        if not line or line.upper().startswith("CREATE") or line.startswith(")") or line.startswith("PARTITION"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            column_name = parts[0].strip('`",')
            column_names.append(column_name)
    return column_names

def is_valid_result(df_csv, do_column_exploration):
    df_csv = df_csv.fillna("")
    df_csv_str = df_csv.astype(str)
    nested_val = [(item) for i, row in enumerate(df_csv.values.tolist()) for j, item in enumerate(row) if isinstance(item, str) and '\n' in item in item]
    # print(df_csv_str)
    if nested_val:
        return False

    if ((df_csv_str == "0") | (df_csv_str == "")).all().any() and not do_column_exploration:
        return False

    return True

def filter_bijection_like_dict(d):
    keys = set(d.keys())
    new_d = {}

    for k, vs in d.items():
        filtered_values = [v for v in vs if v in keys]
        if filtered_values:
            new_d[k] = filtered_values

    return new_d

def is_csv_empty(path):
    try:
        df = pd.read_csv(path)
        return df.empty
    except pd.errors.EmptyDataError:
        return True

def extract_real_table_names(sql: str, dialect: str = "bigquery"):
    expr = sqlglot.parse_one(sql, read=dialect)

    cte_names = {cte.alias for cte in expr.find_all(CTE)}
    cte_names_upper = {name.upper() for name in cte_names}

    full_table_names = {
        table.sql(dialect=dialect)
        for table in expr.find_all(Table)
        if table.name.upper() not in cte_names_upper
    }

    return full_table_names, {column.name for column in expr.find_all(Column)}

def clear_name(table_names, do_remove_digits=True):
    if isinstance(table_names, str):
        cleared = table_names.split(" ")[0].replace('"', '').replace("'", '').replace("`", '').replace("*", '').upper()
        if do_remove_digits:
            return remove_digits(cleared)
        else:
            return cleared
    if do_remove_digits:
        return {remove_digits(raw_name.split(" ")[0].replace('"', '').replace("'", '').replace("`", '').replace("*", '').upper()) for raw_name in table_names}
    return {raw_name.split(" ")[0].replace('"', '').replace("'", '').replace("`", '').replace("*", '').upper() for raw_name in table_names}

def remove_declare_lines(sql_script: str) -> str:
    lines = sql_script.splitlines()
    cleaned_lines = [line for line in lines if not line.strip().upper().startswith("DECLARE")]
    return "\n".join(cleaned_lines)

def clear_byte(rows):
    for i in rows:
        for k, v in i.items():
            if isinstance(v, str) and "bytearray(b" in v:
                i[k] = "bytearray(b'...')"
    return rows

def clear_tb(tb):
    return tb.replace("\"", "").replace("`", "").upper()

def extract_code_blocks(text: str, tag: str):
    pattern = rf"```{tag}\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return [match.strip() for match in matches]

def get_metrics(ce_json):
    ce_recall_tbs = []
    ce_precision_tbs = []
    ce_recall_cols = []
    ce_precision_cols = []
    for ce in ce_json:
        ce_recall_tb = ce["recall_tb"] if ce["recall_tb"] is not None else 1
        ce_precision_tb = ce["precision_tb"] if ce["precision_tb"] is not None else 1
        ce_recall_col = ce["recall_col"] if ce["recall_col"] is not None else 1 
        ce_precision_col = ce["precision_col"] if ce["precision_col"] is not None else 1

        ce_recall_tbs.append(ce_recall_tb)
        ce_precision_tbs.append(ce_precision_tb)
        ce_recall_cols.append(ce_recall_col)
        ce_precision_cols.append(ce_precision_col)

    print(f"""
P(recall_tb == 1):    {np.mean(np.array(ce_recall_tbs) == 1):.4f}
P(precision_tb == 1): {np.mean(np.array(ce_precision_tbs) == 1):.4f}
P(recall_col == 1):   {np.mean(np.array(ce_recall_cols) == 1):.4f}
P(precision_col == 1):{np.mean(np.array(ce_precision_cols) == 1):.4f}
AVG recall_tb:        {np.mean(ce_recall_tbs):.4f}
AVG precision_tb:     {np.mean(ce_precision_tbs):.4f}
AVG recall_col:       {np.mean(ce_recall_cols):.4f}
AVG precision_col:    {np.mean(ce_precision_cols):.4f}
    """)

def is_same_schema(repre_js_pth, js_pth):
    with open(repre_js_pth) as f:
        repre = json.load(f)
    with open(js_pth) as f:
        js = json.load(f)
    if set(repre["column_names"]) == set(js["column_names"]):
        return True
    print("The difference between two sets:")
    print(set(repre["column_names"]) - set(js["column_names"]))
    print(set(js["column_names"]) - set(repre["column_names"]))
    return False
