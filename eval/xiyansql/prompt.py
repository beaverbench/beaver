import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

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


@dataclass
class EvalConfig:
    gold_tables: bool
    join_keys: bool
    mapping: bool
    knowledge: bool
    decomp: bool


def read_json(fn):
    with open(fn) as f:
        return json.load(f)


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


# --- Evidence (【参考信息】) ---
# Same hint content as the fewshot baseline, minus join keys, which go in 【Foreign keys】 above.
# Instruction sentences are worded as in the ReFoRCE preprocessing (knowledge and decomposition ones are
# shared with DIN-SQL and DAIL-SQL). The join-key
# sentence is left out: 【Foreign keys】 is native M-Schema, so the model needs no pointer to it.

MAPPING_INSTRUCTION = "You should use the provided mapping to determine which columns and tables should be used in the SQL statement."
KNOWLEDGE_INSTRUCTION = "You should use the domain knowledge to help determine which tables and columns to use in the SQL statement as well as constructing the SQL statement."
DECOMP_INSTRUCTION = "You must answer each subquery individually and then combine them to form the complete SQL statement. Each subquery you generate must be explicitly used in the final query you generate; do not simplify the subqueries you generate for implementation in the final query."

def get_mapping(mapping):
    desc = []
    for sq, cols in mapping.items():
        cols_desc = ", ".join(f"column {c.split('.')[1]} in table {c.split('.')[0].lower()}" for c in cols)
        desc.append(f'"{sq}" in the user question refers to {cols_desc}')
    return "\n".join(desc)


def get_decomp(q, structures):
    desc = [f"Subquery {i + 1}: {subq}" for i, subq in enumerate(q["sub_questions"])]
    if not desc:
        return ""
    desc.append(DECOMP_INSTRUCTION)
    structure_name = q.get("detailed_category")
    if structure_name and structure_name != 'real' and structure_name in structures:
        structure = structures[structure_name]
        desc.append(f"\nStructure of the SQL statement with subqueries denoted:\n{structure['structure']}")
        desc.append(f"\n{structure['subquery_decomposition']}")
    return "\n".join(desc)


def get_evidence(q, eval_config: EvalConfig, structures):
    evidence = []
    if eval_config.mapping:
        evidence.append(f"Mapping:\n{get_mapping(q['column_mapping'])}\n{MAPPING_INSTRUCTION}")
    if eval_config.knowledge and "\n".join(q['domain_knowledge']) != "":
        evidence.append("Domain knowledge:\n" +"\n".join(q['domain_knowledge']) + f"\n{KNOWLEDGE_INSTRUCTION}")
    if eval_config.decomp:
        decomp = get_decomp(q, structures)
        if decomp:
            evidence.append(f"Query decomposition:\n{decomp}")
    return "\n\n".join(evidence)


def get_retrieved_tables(dataset: str, data_dir: str):
    retrieved_tables_fn = Path(f"{data_dir}/{dataset}/retrieval/retrieved_tables.json")
    reranked_tables_fn = Path(f"{data_dir}/{dataset}/retrieval/reranked_tables.json")

    assert retrieved_tables_fn.exists(), f"{retrieved_tables_fn} not found; run table retrieval first"

    if reranked_tables_fn.exists():
        print(f'Loading reranked tables from {reranked_tables_fn}')
        return read_json(reranked_tables_fn)

    print(f'Loading retrieved tables from {retrieved_tables_fn}')
    return read_json(retrieved_tables_fn)


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
