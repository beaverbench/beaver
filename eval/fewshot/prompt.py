import os
import sys
from tqdm import tqdm

from utils import read_json, format_tables, EvalConfig, format_join, system, user, assistant

# Shared with XiYan-SQL and CHESS; eval/utils is appended, not prepended, so this folder's utils.py keeps the `utils` name
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from hints import get_mapping, get_retrieved_tables, dk_sentences

def get_join_keys(join_keys):
    desc = []
    for join_key in join_keys:
        # key1, key2 = join_key
        # t1, col1 = key1.split(".")
        # t2, col2 = key2.split(".")
        # t1, t2 = f"{t1.lower()}", f"{t2.lower()}"
        # desc.append(
        #     f"Column {col1} in table {t1} joins with column {col2} in table {t2}"
        # )
        desc.append(format_join(join_key))
    return "\n".join(desc)


def get_knowledge(domain_knowledge):
    return "\n".join(dk_sentences(domain_knowledge))


def get_decomp(decomps):
    decomp_str = []
    for subq_idx, subq in enumerate(decomps):
        decomp_str.append(f"Subquery {subq_idx + 1}: {subq}")
    return "\n".join(decomp_str)


def get_user_prompt(
    q,
    tables,
    corpus_tables,
    eval_config: EvalConfig
):
    desc = [
        format_tables(tables, corpus_tables, eval_config.instances),
        f"User question: {q['question']}",
    ]

    if eval_config.mapping:
        desc.append(f"Mapping:\n{get_mapping(q['column_mapping'])}")

    if eval_config.join_keys:
        desc.append(f"Join keys:\n{get_join_keys(q['join_keys'])}")

    if eval_config.knowledge and get_knowledge(q['domain_knowledge']) != '':
        desc.append(
            f"Domain knowledge:\n{get_knowledge(q['domain_knowledge'])}"
        )

    if eval_config.decomp and get_decomp(q["sub_questions"]) != "":
        desc.append(f"Query decomposition:\n{get_decomp(q['sub_questions'])}")

    return user("\n\n".join(desc))

def get_ete_prompts(dataset: str, q_fn: str, eval_config: EvalConfig, data_dir: str = "../../data"):
    structures = read_json(f"{data_dir}/template_structure.json")
    qs = read_json(f"{data_dir}/{dataset}/{q_fn}.json")
    example = read_json(f"{data_dir}/{dataset}/example.json")
    dev_tables = read_json(f"{data_dir}/{dataset}/dev_tables.json")

    if not eval_config.gold_tables:
        retrieved_tables = get_retrieved_tables(dataset, data_dir)

    example_prompt = [
        get_user_prompt(example, example["tables"], dev_tables, eval_config),
        assistant(f"SQL: <ans>{example['sql']}</ans>"),
    ]

    prompts = []
    instance_ids = []

    for q in tqdm(qs):
        q_id = q['id']
        q_knowledge = eval_config.knowledge and get_knowledge(q['domain_knowledge']) != ''
        q_decomp = eval_config.decomp and get_decomp(q["sub_questions"]) != ""

        if eval_config.gold_tables:
            tables = q["tables"]
        else:
            if q_id not in retrieved_tables:
                continue
            tables = retrieved_tables[q_id]

        db_type = "MySQL"

        instruction = ["You are given a list of tables", "a user question"]
        if eval_config.join_keys:
            instruction.append("join keys among the provided tables")
        if eval_config.mapping:
            instruction.append("a mapping from information mentioned in the user question to columns in the provided tables")
        if q_knowledge:
            instruction.append("domain knowledge"),
        if q_decomp:
            instruction.append("decomposition of the user question")
        instruction[-1] = f"and {instruction[-1]}"
        instruction = ", ".join(instruction) + ", "

        instruction += f"your task is output a {db_type} SQL statement that can be used to answer the user question based on the provided information. You need to ensure that syntax and functions used in your SQL statement are appropriate for {db_type} database. If you are unable to determine the SQL statement, output None. "

        if eval_config.mapping:
            instruction += f"You should use the provided mapping to determine which columns and tables should be used in the SQL statement. "
        if eval_config.join_keys:
            instruction += f"You should use the provided join keys to determine how to connect the tables in the SQL statement. "
        if q_knowledge:
            instruction += f"You should use the provided domain knowledge to determine which tables, columns, and literals should be used in the SQL statement. "
        if q_decomp:
            instruction += f"You must answer each subquery individually and then combine them to form the complete SQL statement. Each subquery you generate must be explicitly used in the final SQL statement, without being simplified. "

            instruction += f"Below is the structure of the SQL statement with subqueries denoted. Each provided subquery is used in the final SQL statement in such a structure."

            structure_name = q.get("detailed_category")
            if structure_name and structure_name != 'real' and structure_name in structures:
                structure = structures[structure_name]
                instruction += f"\n\n{structure['structure']}"
                # The following explanation describes which numbered subqueries you are given correspond to which query in the query structure you were provided:
                instruction += f"\n\n{structure['subquery_decomposition']} "

        instruction += f"The SQL statement need to be wrapped in <ans></ans> tags."

        prompt = [system(instruction)] + example_prompt
        prompt += [get_user_prompt(q, tables, dev_tables, eval_config)]

        prompts.append(prompt)
        instance_ids.append(q_id)

    print(f"#prompts: {len(prompts)}")
    return prompts, instance_ids
