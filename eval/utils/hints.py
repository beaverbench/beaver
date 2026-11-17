import json
from dataclasses import dataclass
from pathlib import Path

# Hint settings and evidence text shared by the XiYan-SQL, CHESS and Claude Code baselines; Few-shot, ReFoRCE,
# DIN-SQL and DAIL-SQL import single helpers (get_mapping, get_retrieved_tables, dk_sentences). XiYan-SQL, CHESS
# and Claude Code pass join keys in the schema, so the evidence holds only mapping, domain knowledge and decomposition.
# Instruction sentences are worded as in the ReFoRCE preprocessing (knowledge and decomposition ones are
# shared with DIN-SQL and DAIL-SQL).


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


MAPPING_INSTRUCTION = "You should use the provided mapping to determine which columns and tables should be used in the SQL statement."
KNOWLEDGE_INSTRUCTION = "You should use the domain knowledge to help determine which tables and columns to use in the SQL statement as well as constructing the SQL statement."
DECOMP_INSTRUCTION = "You must answer each subquery individually and then combine them to form the complete SQL statement. Each subquery you generate must be explicitly used in the final query you generate; do not simplify the subqueries you generate for implementation in the final query."

def get_mapping(mapping):
    desc = []
    for sq, cols in mapping.items():
        cols_desc = ", ".join(f"column {c.split('.')[1]} in table {c.split('.')[0].lower()}" for c in cols)
        desc.append(f'"{sq}" in the user question refers to {cols_desc}')
    return "\n".join(desc)


def dk_sentences(domain_knowledge):
    """Hint sentences from domain_knowledge (list of dicts with 'fact', or legacy list of strings)."""
    return [d["fact"] if isinstance(d, dict) else d for d in (domain_knowledge or [])]


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
    knowledge = "\n".join(dk_sentences(q['domain_knowledge']))
    if eval_config.knowledge and knowledge:
        evidence.append(f"Domain knowledge:\n{knowledge}\n{KNOWLEDGE_INSTRUCTION}")
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
