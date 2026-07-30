"""
V2 Preprocessing script to convert Beaver dataset format to DINSQL format
Supports 3 options for different levels of information:
1. Option 1: Base information with gold tables only
2. Option 2: Base information with gold tables + mapping + join keys
3. Option 3: Base information with full context (mapping, join keys, external info, subqueries)

Uses dev_tables_new.json which includes rows and instances for examples
"""
import json
import os
import os.path as osp
import sys
import re
from pathlib import Path


def convert_beaver_tables_to_dinsql_format(beaver_tables_path, output_path, gold_tables_filter=None, split='dw'):
    """
    Convert Beaver's dev_tables_new.json format to DINSQL's expected format
    
    Args:
        beaver_tables_path: Path to dev_tables_new.json
        output_path: Path to save the converted schema
        gold_tables_filter: Optional set of table keys to filter (for option 2-4)
    
    Beaver format: dict with keys like "db_name#sep#table_name", each containing:
        - db_id, table_name_original, column_names_original, column_types, 
          primary_key, foreign_key, rows, instances
    
    DINSQL format: list of dicts, each containing:
        - db_id, table_names_original, column_names_original, column_types, 
          foreign_keys, primary_keys
    """
    with open(beaver_tables_path, 'r') as f:
        beaver_tables = json.load(f)
    
    # Apply filter if provided (for gold tables only)
    if gold_tables_filter:
        # Case insensitive filtering
        # Create lowercased filter set
        gold_filter_lower = {t.lower() for t in gold_tables_filter}
        
        # Filter tables whose lowercase key is in the filter
        # Note: we preserved the original key case from beaver_tables
        filtered_tables = {}
        for k, v in beaver_tables.items():
            # Handle potential db#sep# prefix in key if needed, or assume key is just table name or matches filter format
            # In beaver_tables, keys are usually "db#sep#table" or just "table" depending on file
            # In questions gold_tables, they are often "table" or "db#sep#table"
            # Let's clean the key for comparison
             k_clean = k.split('#sep#')[1] if '#sep#' in k else k
             if k_clean.lower() in gold_filter_lower:
                 filtered_tables[k] = v
        
        beaver_tables = filtered_tables
        print(f"  Filtered to {len(beaver_tables)} gold tables")
    
    # Group tables by database
    db_schemas = {}
    
    for key, table_info in beaver_tables.items():
        db_id = table_info['db']
        
        if db_id not in db_schemas:
            db_schemas[db_id] = {
                'db_id': db_id,
                'db': db_id,
                'table_names': [],
                'table_names_original': [],
                'column_names': [],
                'column_names_original': [],
                'column_types': [],
                'foreign_keys': [],
                'primary_keys': []
            }
        
        schema = db_schemas[db_id]
        table_name = table_info['table_name']
        if split in ["neutron", "nova"]:
            table_name = table_name.lower()
        table_idx = len(schema['table_names_original'])
        
        # Add table name
        schema['table_names'].append(table_name)
        schema['table_names_original'].append(table_name)
        
        # Add wildcard column for the table
        schema['column_names'].append([-1, '*'])
        schema['column_names_original'].append([-1, '*'])
        schema['column_types'].append('text')
        
        # Add columns
        for col_name, col_type in zip(table_info['column_names'], table_info['column_types']):
            if split in ["neutron", "nova"]:
                col_name = col_name.lower()
            schema['column_names'].append([table_idx, col_name])
            schema['column_names_original'].append([table_idx, col_name])
            schema['column_types'].append(col_type)
        
        # Add primary key (if exists)
        if table_info.get('primary_key'):
            pk_list = table_info['primary_key']
            if not isinstance(pk_list, list):
                pk_list = [pk_list]
            
            for pk_col in pk_list:
                # Find the column index in the full column list
                for idx, (tbl_idx, col_name) in enumerate(schema['column_names']):
                    if tbl_idx == table_idx and col_name == pk_col:
                        schema['primary_keys'].append(idx)
                        break
        
        # Add foreign keys (if exist)
        if table_info.get('foreign_key'):
            for fk_info in table_info['foreign_key']:
                if isinstance(fk_info, dict):
                    source_col = fk_info.get('column_name')
                    target_table_full = fk_info.get('referenced_table_name', '')
                    target_col = fk_info.get('referenced_column_name')
                    
                    # Extract target table name (remove db_id prefix if present)
                    if '#sep#' in target_table_full:
                        target_table = target_table_full.split('#sep#')[1]
                    else:
                        target_table = target_table_full
                    
                     # Case insensitive lookup for target table
                    target_table_idx = None
                    try:
                        # Try exact match first
                        target_table_idx = schema['table_names_original'].index(target_table)
                    except ValueError:
                        # Try case insensitive match
                        for i, name in enumerate(schema['table_names_original']):
                            if name.lower() == target_table.lower():
                                target_table_idx = i
                                break
                    
                    source_idx = None
                    for idx, (tbl_idx, col_name) in enumerate(schema['column_names']):
                        if tbl_idx == table_idx and col_name == source_col:
                            source_idx = idx
                            break
                    
                    target_idx = None
                    if target_table_idx is not None:
                        for idx, (tbl_idx, col_name) in enumerate(schema['column_names']):
                            if tbl_idx == target_table_idx and col_name == target_col:
                                target_idx = idx
                                break
                    
                    if source_idx is not None and target_idx is not None:
                        schema['foreign_keys'].append([source_idx, target_idx])
    
    # Convert to list format
    dinsql_format = list(db_schemas.values())
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dinsql_format, f, indent=2)
    
    print(f"  Converted {len(beaver_tables)} tables into {len(dinsql_format)} databases")
    print(f"  Saved to: {output_path}")
    
    return dinsql_format


def format_mapping_for_prompt(mapping):
    """Convert mapping dict to readable string for prompt"""
    if not mapping:
        return ""
    
    lines = []
    for concept, columns in mapping.items():
        columns_str = ", ".join(columns)
        lines.append(f"  - {concept}: {columns_str}")
    
    return "\n".join(lines)


def format_join_keys_for_prompt(join_keys):
    """Convert join_keys list to readable string for prompt"""
    if not join_keys:
        return ""
    
    lines = []
    for idx, join_pair in enumerate(join_keys, 1):
        if len(join_pair) == 2:
            lines.append(f"  {idx}. {join_pair[0]} = {join_pair[1]}")
    
    return "\n".join(lines)

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

def convert_beaver_questions_to_dinsql_format(dataset, beaver_questions_path, output_path, option=1, templates=None):
    """
    Convert Beaver's dev_dw.json format to DINSQL's expected format
    
    Args:
        beaver_questions_path: Path to dev_dw.json
        output_path: Path to save the converted questions
        option: Preprocessing option (1, 2, or 3)
            1: Base information with top k tables
            2: Base information with gold tables + mapping + join keys
            3: Base information with full context
        templates: Optional templates dict for option 3
    
    Beaver format: list of dicts with:
        - question, db_id, sql, oracle_sql, gold_tables, mapping, join_keys
    
    DINSQL format: list of dicts with:
        - instance_id, question, db_id, gold_sql
        - For option 3+: additional mapping information
        - For option 4: additional join keys information
        - For option 5: additional external knowledge, subquery questions, and template hints
    """
    with open(beaver_questions_path, 'r') as f:
        beaver_questions = json.load(f)
    
    retrieved_tables = None
    if option == 1:
        retrieved_tables = get_retrieved_tables(dataset)
    
    dinsql_format = []
    for question_info in beaver_questions:
        base_item = {
            'id': question_info['id'],
            'question': question_info['question'],
            'db': question_info['db'],
            'gold_sql': question_info.get('sql', ''),
        }
        
        # Option 1+: Store gold_tables for filtering
        if option >= 1:
            if option == 1 and retrieved_tables is not None:
                 base_item['tables'] = retrieved_tables[question_info['id']]
            else:
                 base_item['tables'] = question_info.get('tables', [])
            # strip db#sep# from gold_tables generic and lowercase
            if 'dw' in question_info['db'] or 'dw_real' in question_info['db']:
                # do not lowercase
                base_item['tables'] = [t.split('#sep#')[1] if '#sep#' in t else t for t in base_item['tables']]
            else:
                # lowercase
                base_item['tables'] = [t.split('#sep#')[1].lower() if '#sep#' in t else t.lower() for t in base_item['tables']]
            base_item['question'] = (
                f"{question_info['question']}\n\n"
                f"[Gold Tables]\n{', '.join(base_item['tables']).upper()}"
            )
        
        # Option 2+: Add mapping to question
        if option >= 2:
            mapping = question_info.get('column_mapping', {})
            if mapping:
                mapping_str = format_mapping_for_prompt(mapping)
                base_item['question'] = (
                    f"{base_item['question']}\n\n"
                    f"[Schema Mapping Hints]\n{mapping_str}"
                )
                base_item['column_mapping'] = mapping
        
        # Option 2+: Add join keys
        if option >= 2:
            join_keys = question_info.get('join_keys', [])
            if join_keys:
                join_keys_str = format_join_keys_for_prompt(join_keys)
                base_item['question'] = (
                    f"{base_item['question']}\n\n"
                    f"[Join Key Hints]\n{join_keys_str}"
                )
                base_item['join_keys'] = join_keys

        # Option 3: Add external knowledge and subquery info
        if option >= 3:
            domain_knowledge = question_info.get('domain_knowledge', [])
            sub_questions = question_info.get('sub_questions', [])
            
            if domain_knowledge:
                base_item['question'] += "\n\n-- Domain Knowledge (database-wide):\n"
                base_item['question'] += "\n".join(domain_knowledge)
                base_item['question'] += "\n"
                base_item['question'] += "You should use the domain knowledge to help determine which tables and columns to use in the SQL statement as well as constructing the SQL statement."

            if sub_questions:
                base_item['question'] += "\n\n-- Subquery Gold Questions (database-wide):\n"
                base_item['question'] += "\n".join(sub_questions)
                base_item['question'] += "\n"
                base_item['question'] += "You must answer each subquery individually and then combine them to form the complete SQL statement. Each subquery you generate must be explicitly used in the final query you generate; do not simplify the subqueries you generate for implementation in the final query."
            
            if templates:
                detailed_category = question_info.get('detailed_category')
                if detailed_category and detailed_category != 'real' and detailed_category in templates:
                    base_item['question'] += "\n\n Here is an explanation of which numbered subqueries you are given correspond to which query in the query structure you were provided:\n"
                    base_item['question'] += f"{templates[detailed_category]['structure']}\n\n"
                    base_item['question'] += templates[detailed_category]['subquery_decomposition']
        
        dinsql_format.append(base_item)
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dinsql_format, f, indent=2)
    
    print(f"  Converted {len(beaver_questions)} questions")
    print(f"  Saved to: {output_path}")
    
    return dinsql_format


def collect_all_gold_tables(questions):
    """Collect all unique gold tables from all questions"""
    all_gold_tables = set()
    for q in questions:
        gold_tables = q.get('tables', [])
        # Clean gold tables
        clean_gold_tables = [t.split('#sep#')[1] if '#sep#' in t else t for t in gold_tables]
        all_gold_tables.update(clean_gold_tables)
    return all_gold_tables


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess Beaver dataset for DINSQL with different options')
    parser.add_argument('--beaver_dir', default='../../data', type=str, 
                        help='Directory containing Beaver dataset files')
    parser.add_argument('--option', type=int, required=True, choices=[1, 2, 3],
                        help='Preprocessing option: 1=top20_tables, 2=+gold_tables+mapping+join_keys, 3=+external')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of questions to process (for testing)')
    parser.add_argument('--questions_file', type=str, default=None,
                        help='Specific path to questions file (overrides beaver_dir default)')
    parser.add_argument('--tables_file', type=str, default=None,
                        help='Specific path to tables file (overrides beaver_dir default)')
    parser.add_argument('--dataset', type=str, default='dw',
                        help='Dataset to preprocess (default: dw)')
    args = parser.parse_args()
    
    proj_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    beaver_base_dir = osp.join(proj_dir, args.beaver_dir)
    
    subdir_name = f'beaver_{args.dataset}_opt{args.option}'
    output_base_dir = osp.join(proj_dir, 'preprocessed_data', subdir_name)
    
    print("=" * 70)
    print(f"BEAVER V2 PREPROCESSING - OPTION {args.option}")
    print("=" * 70)
    
    option_descriptions = {
        1: "Base information (question) with top k tables available",
        2: "Base information with gold tables + mapping + join key hints",
        3: "Base information with full context (mapping, join keys, external info, subqueries)"
    }
    print(f"Option {args.option}: {option_descriptions[args.option]}")
    print("=" * 70)
    
    # Load questions first to get gold tables if needed
    if args.questions_file:
        beaver_questions_path = args.questions_file
    else:
        beaver_questions_path = osp.join(beaver_base_dir, 'dev_dw_new', 'final_combined_with_source_sampled.json')
    
    print("\n[1/2] Loading questions...")
    print(f"  Reading from: {beaver_questions_path}")
    
    with open(beaver_questions_path, 'r') as f:
        questions = json.load(f)
    
    if args.limit:
        questions = questions[:args.limit]
        print(f"  Limited to first {args.limit} questions")
    
    # Collect gold tables
    gold_tables_filter = None
    if args.option >= 1:
        print("  Collecting gold tables from questions...")
        gold_tables_filter = collect_all_gold_tables(questions)
        print(f"  Found {len(gold_tables_filter)} unique gold tables across all questions")
    
    # Convert tables
    if args.tables_file:
         beaver_tables_path = args.tables_file
    else:
         beaver_tables_path = osp.join(beaver_base_dir, 'dev_tables_new.json')
    
    output_tables_path = osp.join(output_base_dir, 'tables_preprocessed.json')
    
    print(f"\n[2/2] Converting tables...")
    convert_beaver_tables_to_dinsql_format(
        beaver_tables_path, 
        output_tables_path,
        gold_tables_filter=gold_tables_filter,
        split=args.dataset
    )
    
    output_questions_filename = f'beaver_{args.dataset}_opt{args.option}_preprocessed.json'
    output_questions_path = osp.join(output_base_dir, output_questions_filename)
    
    print(f"\n[3/3] Converting questions with option {args.option}...")
    
    # Save limited questions to temp file if needed
    if args.limit:
        temp_questions_path = beaver_questions_path + '.tmp'
        with open(temp_questions_path, 'w') as f:
            json.dump(questions, f)
        beaver_questions_path = temp_questions_path
    
    templates = None
    if args.option == 3:
        templates_path = osp.join(beaver_base_dir, 'template_structure.json')
        print(f"Loading templates from: {templates_path}")
        with open(templates_path, 'r') as f:
            templates = json.load(f)

    dinsql_questions = convert_beaver_questions_to_dinsql_format(
        args.dataset,
        beaver_questions_path, 
        output_questions_path,
        option=args.option,
        templates=templates
    )
    
    # Show sample of the first question
    if dinsql_questions:
        print("\n" + "=" * 70)
        print("SAMPLE OUTPUT (First Question)")
        print("=" * 70)
        import pprint
        pprint.pprint(dinsql_questions[0])
    
    # Clean up temp file
    if args.limit and osp.exists(beaver_questions_path + ('.tmp' if not beaver_questions_path.endswith('.tmp') else '')):
        temp_path = beaver_questions_path if beaver_questions_path.endswith('.tmp') else beaver_questions_path + '.tmp'
        if osp.exists(temp_path):
            os.remove(temp_path)
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Output directory: {output_base_dir}")
    print(f"  - tables_preprocessed.json: Database schema")
    print(f"  - {output_questions_filename}: Preprocessed questions")
