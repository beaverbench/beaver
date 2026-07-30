#!/usr/bin/env python3
"""
Convert Beaver dataset format to ReFoRCE format.
This script transforms the Beaver dev_dw_new/combined_fixed.json and dev_tables_new.json files
into the format expected by ReFoRCE with 3 preprocessing options.
"""

import json
import argparse
import random
from pathlib import Path
import re


def format_instances_as_examples(instances, column_names):
    """
    Convert instances to example strings for each column.
    
    Args:
        instances: List of lists, where each inner list contains values for one column
        column_names: List of column names
    
    Returns:
        Dict mapping column names to example value strings
    """
    if not instances or not column_names:
        return {}
    
    examples = {}
    for i, col_name in enumerate(column_names):
        if i < len(instances):
            # Get values for this column, limit to first 3 unique values
            col_values = instances[i]
            if col_values:
                # Convert values to strings and limit
                value_strs = [str(v) for v in col_values[:3]]
                examples[col_name] = value_strs
    
    return examples


def get_all_tables_from_db(beaver_tables, db_id):
    """
    Get all table references for a given database ID.
    
    Args:
        beaver_tables: Dict of all tables
        db_id: Database identifier
    
    Returns:
        List of table reference keys (e.g., "dw#sep#TABLE_NAME")
    """
    all_tables = []
    for table_ref, table_info in beaver_tables.items():
        if table_info.get('db') == db_id:
            all_tables.append(table_ref)
    return all_tables


def create_table_schema(table_info, examples_dict):
    """
    Create a CREATE TABLE statement with examples.
    
    Args:
        table_info: Table metadata from beaver_tables
        examples_dict: Dict mapping column names to example values
    
    Returns:
        CREATE TABLE statement as string
    """
    # Use clean table name without database prefix
    table_name = table_info['table_name']
    create_stmt = f"CREATE TABLE {table_name} (\n"
    
    cols = []
    for col_name, col_type in zip(table_info['column_names'], 
                                 table_info['column_types']):
        # Add examples if available
        if col_name in examples_dict and examples_dict[col_name]:
            example_str = ", ".join([f"'{v}'" for v in examples_dict[col_name]])
            cols.append(f"    {col_name} {col_type} -- example: [{example_str}]")
        else:
            cols.append(f"    {col_name} {col_type}")
    
    create_stmt += ",\n".join(cols)
    create_stmt += "\n);"
    
    return create_stmt


def add_mapping_info(mapping):
    """
    Format mapping information as comments.
    Clean table references by removing database prefix.
    
    Args:
        mapping: Dict mapping question phrases to column references
    
    Returns:
        Formatted mapping string
    """
    if not mapping:
        return ""
    
    lines = ["\n-- Schema Mapping (question concepts to columns):"]
    for concept, columns in mapping.items():
        # Clean column references by removing "dw#sep#" prefix
        cleaned_columns = []
        for col in columns:
            if '#sep#' in col:
                # Remove "{db_id}#sep#" or similar prefixes
                col = col.split('#sep#', 1)[1]
            cleaned_columns.append(col)
        column_list = ", ".join(cleaned_columns)
        lines.append(f"-- '{concept}' -> {column_list}")
    
    lines.append("\n")
    lines.append("You should use the provided mapping to determine which columns and tables should be used in the SQL statement.")
    
    return "\n".join(lines)


def add_join_keys_info(join_keys):
    """
    Format join keys information as comments.
    Clean table references by removing database prefix.
    
    Args:
        join_keys: List of [col1, col2] pairs showing how tables join
    
    Returns:
        Formatted join keys string
    """
    if not join_keys:
        return ""
    
    lines = ["\n-- Join Keys (how tables connect):"]
    for join_pair in join_keys:
        if len(join_pair) == 2:
            # Clean join key references by removing "{db_id}#sep#" prefix
            col1 = join_pair[0].split('#sep#', 1)[1] if '#sep#' in join_pair[0] else join_pair[0]
            col2 = join_pair[1].split('#sep#', 1)[1] if '#sep#' in join_pair[1] else join_pair[1]
            lines.append(f"-- {col1} = {col2}")
    
    lines.append("\n")
    lines.append("You should use the provided join keys to determine how to connect the tables in the SQL statement.")
    
    return "\n".join(lines)





def extract_question_and_sql(text):
    """Extract question and SQL from text using regex patterns."""
    sql_match = re.search(r"<sqlans>\s*(.*?)\s*</sqlans>", text, re.DOTALL)
    nl_match = re.search(r"<nlans>\s*(.*?)\s*</nlans>", text, re.DOTALL)

    sql = sql_match.group(1).strip() if sql_match else ""
    # Remove db prefix if present
    if '#sep#' in sql:
        sql = sql.replace('dw#sep#', '').replace('sp#sep#', '') # Handle known prefixes or use regex
    # Better approach might be passed in db_id but for now let's make it generic or handle both
    # Actually let's change signature to accept db_id
    # But since this function is not used in main loop (process_sql is used), I'll just make it robust
    sql = re.sub(r'\w+#sep#', '', sql)  
    sql = sql.replace('\n', ' ')      
    sql = sql.replace('"', '')
    sql = ' '.join(sql.split())   
    question = nl_match.group(1).strip() if nl_match else ""

    return question, sql

def process_sql(sql, db_id="dw"):
    sql = sql.strip()
    # Case insensitive removal of db prefix
    prefix = f"{db_id}#sep#"
    pattern = re.compile(re.escape(prefix), re.IGNORECASE)
    sql = pattern.sub('', sql)
    
    sql = sql.replace('\n', ' ')      
    sql = sql.replace('"', '')
    sql = ' '.join(sql.split())   
    return sql

def read_json(fn):
    with open(fn) as f:
        return json.load(f)

def get_retrieved_tables(dataset: str, data_dir="../../data"):
    retrieved_tables_fn = Path(f"{data_dir}/{dataset}/retrieval/retrieved_tables.json")
    reranked_tables_fn = Path(f"{data_dir}/{dataset}/retrieval/reranked_tables.json")

    assert retrieved_tables_fn.exists()
    
    if reranked_tables_fn.exists():
        print(f'Loading reranked tables from {reranked_tables_fn}')
        return read_json(reranked_tables_fn)
    
    print(f'Loading retrieved tables from {retrieved_tables_fn}')
    return read_json(retrieved_tables_fn)

def convert_beaver_to_reforce(dataset, beaver_questions_path, beaver_tables_path, output_path, sampled_questions_path,
                              preprocessing_option=2, join_keys_path=None, sample_size=None, seed=42, db_id="dw"):
    """
    Convert Beaver dataset to ReFoRCE format with preprocessing options.
    
    Args:
        beaver_questions_path: Path to Beaver dev_dw_new/combined_fixed.json
        beaver_tables_path: Path to Beaver dev_tables_new.json
        output_path: Path to save the converted data
        preprocessing_option: Integer 1-3 for preprocessing level:
            1 - Base info with top k tables
            2 - Base info with gold tables + mapping + join keys
            3 - Base info with gold tables + mapping + join keys + external knowledge + subqueries
        join_keys_path: Optional path to global join keys file (for option 2)
        sample_size: Optional number of queries to randomly sample (e.g., 20)
        seed: Random seed for reproducibility (default: 42)
    """
    # Load Beaver data
    print(f"Loading Beaver questions from: {beaver_questions_path}")
    with open(beaver_questions_path, 'r') as f:
        beaver_questions = json.load(f)
    
    print(f"Loading Beaver tables from: {beaver_tables_path}")
    with open(beaver_tables_path, 'r') as f:
        beaver_tables = json.load(f)

    # Lowercase tables for specific databases to match MySQL casing
    if db_id in ['sp', 'neutron', 'nova']:
        print(f"Lowercasing table/column names for {db_id}...")
        new_beaver_tables = {}
        for k, v in beaver_tables.items():
            new_k = k.lower()
            v['table_name'] = v['table_name'].lower()
            v['column_names'] = [c.lower() for c in v['column_names']]
            new_beaver_tables[new_k] = v
        beaver_tables = new_beaver_tables
    
    # Random sampling if requested
    if sample_size is not None and sample_size < len(beaver_questions):
        random.seed(seed)
        beaver_questions = random.sample(beaver_questions, sample_size)
        print(f"Randomly sampled {sample_size} queries from dataset (seed={seed})")

        # save sampled questions to file
        with open(sampled_questions_path, 'w') as f:
            json.dump(beaver_questions, f, indent=4)
    
    # Load global join keys if provided
    global_join_keys = []
    if preprocessing_option == 2 and join_keys_path:
        print(f"Loading global join keys from: {join_keys_path}")
        with open(join_keys_path, 'r') as f:
            global_join_keys = json.load(f)

    # Load templates if preprocessing option is 3
    templates = []
    if preprocessing_option == 3:
        templates_path = "../../data/template_structure.json"
        print(f"Loading templates from: {templates_path}")
        with open(templates_path, 'r') as f:
            templates = json.load(f)
    
    print(f"\nPreprocessing option: {preprocessing_option}")
    option_desc = {
        1: "Base info with TOP-K tables only",
        2: "Base info with GOLD tables + MAPPING + JOIN KEYS",
        3: "Base info with GOLD tables + MAPPING + JOIN KEYS + DOMAIN KNOWLEDGE + SUBQUERY GOLD QUESTIONS + SUBQUERY GOLD QUERIES"
    }
    print(f"Mode: {option_desc.get(preprocessing_option, 'Unknown')}")
    
    # Convert to ReFoRCE format
    reforce_data = []
    
    # Read only first few for testing when requested
    print(f"\nProcessing {len(beaver_questions)} questions...")
    
    # Helper to find table in beaver_tables case-insensitively
    def get_table_key(t_name):
        t_name_lower = t_name.lower()
        for k in beaver_tables:
            if k.lower() == t_name_lower:
                return k
        return None

    # Fix casing for SP database
    if db_id in ['sp', 'neutron', 'nova']:
        print("Fixing casing for SP database...")
        for item in beaver_questions:
            # Fix SQL casing
            if 'sql' in item:
                current_sql = item['sql']
                for k in beaver_tables:
                    # Case insensitive replace of table names
                    # Use regex with word boundaries to avoid replacing substrings
                    pattern = re.compile(r'\b' + re.escape(k) + r'\b', re.IGNORECASE)
                    current_sql = pattern.sub(k, current_sql)
                item['sql'] = current_sql
            
            # Fix gold_tables list casing
            if "tables" in item:
                gold_tables = item["tables"]
                fixed_gold_tables = []
                for t in gold_tables:
                    k = get_table_key(t)
                    if k:
                        fixed_gold_tables.append(k)
                    else:
                        fixed_gold_tables.append(t.lower()) # Fallback
                item["tables"] = fixed_gold_tables
            
            # Fix mapping casing
            if "column_mapping" in item:
                new_mapping = {}
                for m_k, m_v in item["column_mapping"].items():
                    new_m_v = [col.lower() for col in m_v]
                    new_mapping[m_k] = new_m_v
                item["column_mapping"] = new_mapping

            # Fix join_keys casing
            if 'join_keys' in item:
                new_join_keys = []
                for jk in item['join_keys']:
                    new_join_keys.append([c.lower() for c in jk])
                item['join_keys'] = new_join_keys

    retrieved_tables = None
    if preprocessing_option == 1:
        retrieved_tables = get_retrieved_tables(dataset)

    for idx, item in enumerate(beaver_questions):
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(beaver_questions)}")
        
        # Determine which tables to include based on preprocessing option
        table_refs = []
        if preprocessing_option == 1:
            gold_tables = retrieved_tables[item['id']]
        else:
            gold_tables = item.get('tables', [])
        
        # Normalize gold tables for lookup
        gold_tables_lookup = []
        for t in gold_tables:
            tk = get_table_key(t)
            if tk:
                gold_tables_lookup.append(tk)
            else:
                print(f"  Warning: Table {t} not found in beaver_tables for item {item['id']}")
        
        table_refs = list(set(gold_tables_lookup)) # Use the found keys for table_refs
        
        # Generate db_desc from selected tables
        db_desc_parts = []
        
        for table_ref in table_refs:
            if table_ref in beaver_tables:
                table_info = beaver_tables[table_ref]
                
                # Get examples from instances if available
                instances = table_info.get('example_columns', [])
                examples_dict = format_instances_as_examples(
                    instances, 
                    table_info['column_names']
                )
                
                # Create table schema
                create_stmt = create_table_schema(table_info, examples_dict)
                db_desc_parts.append(create_stmt)
            else:
                # This case should ideally be caught by the get_table_key lookup earlier
                print(f"  Warning: Table {table_ref} not found in beaver_tables during schema generation for item {item['id']}")
        
        db_desc = "\n\n".join(db_desc_parts)
        
        # Add mapping info for options 2 and 3
        if preprocessing_option >= 2 and 'column_mapping' in item:
            db_desc += add_mapping_info(item['column_mapping'])
        
        # Add join keys info for options 2 and 3
        if preprocessing_option >= 2:
            # Use question-specific join keys if available
            join_keys = item.get('join_keys', [])
            print(join_keys)
            if join_keys:
                db_desc += add_join_keys_info(join_keys)
            # Optionally also add global join keys
            elif global_join_keys:
                db_desc += "\n\n-- Global Join Keys (database-wide):"
                db_desc += add_join_keys_info(global_join_keys)

        # Add other evidences for option 3
        if preprocessing_option >= 3:
            domain_knowledge = item.get('domain_knowledge', [])
            sub_questions = item.get('sub_questions', [])
            sub_sqls = item.get('sub_sqls', [])

            if domain_knowledge:
                db_desc += "\n\n-- Domain Knowledge (database-wide):\n"
                db_desc += "\n".join(domain_knowledge)
                db_desc += "\n"
                db_desc += "You should use the domain knowledge to help determine which tables and columns to use in the SQL statement as well as constructing the SQL statement."

            if sub_questions:
                db_desc += "\n\n-- Subquery Gold Questions (database-wide):\n"
                db_desc += "\n".join(sub_questions)
                db_desc += "\n"
                db_desc += "You must answer each subquery individually and then combine them to form the complete SQL statement. Each subquery you generate must be explicitly used in the final query you generate; do not simplify the subqueries you generate for implementation in the final query."
            
            detailed_category = item.get('detailed_category')
            if detailed_category and detailed_category != 'real' and detailed_category in templates:
                db_desc += "\n\n Here is an explanation of which numbered subqueries you are given correspond to which query in the query structure you were provided:\n"
                db_desc += f"{templates[detailed_category]['structure']}\n\n"
                db_desc += templates[detailed_category]['subquery_decomposition']
            
        
        # Create ReFoRCE entry
        reforce_entry = {
            "id": item['id'],
            "db": item['db'],
            "db_desc": db_desc
        }

        question = item['question']
        sql = process_sql(item['sql'], db_id=db_id)
        # question, sql = extract_question_and_sql(item['resp'])
        reforce_entry['question'] = question
        if db_id == 'dw':
            pass
            # sql = sql.upper()
        reforce_entry['gold_sql'] = sql
        
        reforce_data.append(reforce_entry)
    
    # Save converted data
    print(f"\nSaving {len(reforce_data)} entries to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(reforce_data, f, indent=2)
    
    print(f"✓ Conversion complete! Saved to {output_path}")
    print(f"  Total questions: {len(reforce_data)}")
    print(f"  Sample instance_id: {reforce_data[0]['id']}")
    print(f"  Preprocessing option: {preprocessing_option} - {option_desc.get(preprocessing_option)}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Beaver dataset to ReFoRCE format with preprocessing options',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Preprocessing Options:
  1 - Base info with TOP K tables only (default, most focused)
  2 - Base info with GOLD tables + MAPPING + JOIN KEYS (full context)
  3 - Base info with GOLD tables + MAPPING + JOIN KEYS + EXTERNAL KNOWLEDGE + SUBQUERY GOLD QUESTIONS + SUBQUERY GOLD QUERIES

Examples:
  # Option 1: Base info with TOP K tables only (default)
  python convert_beaver_to_reforce.py --beaver_questions beaver/dev_dw_new/combined_fixed.json \\
    --beaver_tables beaver/dev_tables_new.json --output output1.json
  
  # Option 2: Full context with join keys, 20 samples
  python convert_beaver_to_reforce.py --beaver_questions beaver/dev_dw_new/combined_fixed.json \\
    --beaver_tables beaver/dev_tables_new.json --output output2.json \\
    --option 2 --join_keys beaver/dw_join_keys.json --sample_size 20

  # Option 3: Full context with join keys with internal_evidence, external_evidence, subquery_gold_questions, and subquery_gold_queries 20 samples
  python convert_beaver_to_reforce.py --beaver_questions beaver/dev_dw_new/combined_fixed.json \\
    --beaver_tables beaver/dev_tables_new.json --output output3.json \\
    --option 3 --join_keys beaver/dw_join_keys.json --sample_size 20
  
        """
    )
    parser.add_argument('--dataset', type=str, default="dw",
                       help='dataset (default: dw)')
    parser.add_argument('--beaver_questions', type=str, required=True,
                       help='Path to Beaver questions file (dev_dw_new/combined_fixed.json)')
    parser.add_argument('--beaver_tables', type=str, required=True,
                       help='Path to Beaver tables file (dev_tables_new.json)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for converted ReFoRCE format file')
    parser.add_argument('--option', type=int, default=1, choices=[1, 2, 3],
                       help='Preprocessing option (1-3, default: 1)')
    parser.add_argument('--join_keys', type=str, default=None,
                       help='Path to global join keys file (used with option 2)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of queries to randomly sample (e.g., 20 for testing)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling reproducibility (default: 42)')
    parser.add_argument('--db_id', type=str, default="dw",
                       help='Database ID (default: dw)')
    parser.add_argument('--sampled_questions_path', type=str, default="../../../beaver/beaver_reforce_sample50.json",
                       help='Path to save sampled questions')
    
    args = parser.parse_args()

    if 'real' in args.db_id or 'easy' in args.db_id:
        args.db_id = args.db_id.replace('_real', '')
    
    convert_beaver_to_reforce(
        args.dataset,
        args.beaver_questions,
        args.beaver_tables,
        args.output,
        args.sampled_questions_path,
        preprocessing_option=args.option,
        join_keys_path=args.join_keys,
        sample_size=args.sample_size,
        seed=args.seed,
        db_id=args.db_id
    )


if __name__ == '__main__':
    main()
