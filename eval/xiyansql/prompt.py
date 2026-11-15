import math
import os
import re
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# Evidence (【参考信息】) is the shared hint text minus join keys, which go in 【Foreign keys】 below;
# 【Foreign keys】 is native M-Schema, so the model needs no pointer to it.
from utils.hints import EvalConfig, read_json, get_evidence, get_retrieved_tables

# Official XiYanSQL-QwenCoder prompt (https://github.com/XGenerationLab/XiYanSQL-QwenCoder), kept verbatim
NL2SQL_TEMPLATE_CN = """你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{dialect}知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```sql"""

DIALECT = "MySQL"


# --- M-Schema (https://github.com/XGenerationLab/M-Schema) ---
# Re-implements MSchema.to_mschema() with its defaults (example_num=3, show_type_detail=False) over
# dev_tables.json instead of a live DB, so every baseline sees the same table metadata. dev_tables.json
# carries no primary keys or column comments, so neither is shown; join keys become 【Foreign keys】.

def _is_email(s):
    return re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', s) is not None


def _examples_to_str(examples):
    # Port of M-Schema utils.examples_to_str: any email/URL value drops all examples for the column
    for v in examples:
        if _is_email(str(v)) or 'http://' in str(v) or 'https://' in str(v):
            return []
    return [str(v) for v in examples if len(str(v)) > 0]


def _field_line(name, field_type, examples, example_num=3):
    raw_type = field_type.split("(")[0].upper()
    line = f"({name}:{raw_type}"

    # M-Schema samples DISTINCT values; example_columns can repeat values and uses NaN for NULL.
    # MySQL SET values arrive as lists; join them the way MySQL returns them ('A,B').
    examples = [v for v in examples if v is not None and not (isinstance(v, float) and math.isnan(v))]
    examples = [",".join(map(str, v)) if isinstance(v, list) else v for v in examples]
    examples = list(dict.fromkeys(_examples_to_str(examples)))[:example_num]
    if raw_type in ['DATE', 'TIME', 'DATETIME', 'TIMESTAMP']:
        examples = examples[:1]
    elif examples and max(len(s) for s in examples) > 20:
        examples = [] if max(len(s) for s in examples) > 50 else examples[:1]
    if examples:
        line += f", Examples: [{', '.join(examples)}]"
    return line + ")"


def table_mschema(table_name, corpus_tables):
    t = corpus_tables[table_name]
    fields = [
        _field_line(name, ty, ex)
        for name, ty, ex in zip(t["column_names"], t["column_types"], t["example_columns"])
    ]
    return "\n".join([f"# Table: {table_name}", "[", ",\n".join(fields), "]"])


def to_mschema(db_id, tables, corpus_tables, join_keys=None):
    out = [f"【DB_ID】 {db_id}", "【Schema】"]
    out += [table_mschema(t, corpus_tables) for t in tables]
    if join_keys:
        out.append("【Foreign keys】")
        out += [f"{left}={right}" for left, right in join_keys]
    return "\n".join(out)


def get_ete_prompts(dataset: str, q_fn: str, eval_config: EvalConfig, data_dir: str = "../../data"):
    structures = read_json(f"{data_dir}/template_structure.json")
    qs = read_json(f"{data_dir}/{dataset}/{q_fn}.json")
    dev_tables = read_json(f"{data_dir}/{dataset}/dev_tables.json")

    if not eval_config.gold_tables:
        retrieved_tables = get_retrieved_tables(dataset, data_dir)

    prompts = []
    instance_ids = []

    for q in tqdm(qs):
        q_id = q['id']

        if eval_config.gold_tables:
            tables = q["tables"]
        else:
            if q_id not in retrieved_tables:
                continue
            tables = retrieved_tables[q_id]

        join_keys = q["join_keys"] if eval_config.join_keys else None
        prompt = NL2SQL_TEMPLATE_CN.format(
            dialect=DIALECT,
            question=q["question"],
            db_schema=to_mschema(q["db"], tables, dev_tables, join_keys),
            evidence=get_evidence(q, eval_config, structures),
        )

        prompts.append([{"role": "user", "content": prompt}])
        instance_ids.append(q_id)

    print(f"#prompts: {len(prompts)}")
    return prompts, instance_ids
