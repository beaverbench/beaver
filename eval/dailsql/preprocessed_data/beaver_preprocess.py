"""
Preprocessing script to convert Beaver dataset format to DAILSQL format
Supports 3 options for different levels of information:
1. Option 1: Base information with top k tables available
2. Option 2: Base information with gold tables + mapping + join keys
3. Option 3: Base information with full context (mapping, join keys, external info, subqueries)

Uses dev_tables_new.json which includes rows and instances for examples
"""
import json
import os
import os.path as osp
import sys
import spacy
from tqdm import tqdm
import re
from pathlib import Path


def convert_beaver_tables_to_dailsql_format(beaver_tables_path, output_path, gold_tables_filter=None, split='dw'):
    """
    Convert Beaver's dev_tables_new.json format to DAILSQL's expected format
    
    Args:
        beaver_tables_path: Path to dev_tables_new.json
        output_path: Path to save the converted schema
        gold_tables_filter: Optional set of table keys to filter (for option 2-4)
    
    Beaver format: dict with keys like "db_name#sep#table_name", each containing:
        - db_id, table_name_original, column_names_original, column_types, 
          primary_key, foreign_key, rows, instances
    
    DAILSQL format: list of dicts, each containing:
        - db_id, table_names_original, column_names_original, column_types, 
          foreign_keys, primary_keys
    """
    with open(beaver_tables_path, 'r') as f:
        beaver_tables = json.load(f)
    
    # Apply filter if provided (for option 2-4: use only gold tables)
    # If None, include all tables in the database (for option 1)
    if gold_tables_filter is not None:
        beaver_tables = {k: v for k, v in beaver_tables.items() if k in gold_tables_filter}
        print(f"  Filtered to {len(beaver_tables)} gold tables")
    else:
        print(f"  Including all {len(beaver_tables)} tables in database")
    
    # Group tables by database
    db_schemas = {}
    
    for key, table_info in beaver_tables.items():
        db_id = table_info['db']
        
        if db_id not in db_schemas:
            db_schemas[db_id] = {
                'db_id': db_id,
                'db': db_id,
                'table_names': [],  # DAILSQL needs this
                'table_names_original': [],
                'column_names': [],  # DAILSQL needs this
                'column_names_original': [],
                'column_types': [],
                'foreign_keys': [],
                'primary_keys': [],
                'column_descriptions': {},  # Required by DAILSQL prompt formatter
                'sample_rows': {},  # Required by DAILSQL prompt formatter (dict, not list)
                'db_stats': {'No. of columns': 0, 'No. of tables': 0}
            }
        
        schema = db_schemas[db_id]
        table_name = table_info['table_name']
        if split in ["neutron", "nova"]:
            table_name = table_name.lower()
        
        # Add wildcard column only once at the start
        if len(schema['table_names_original']) == 0:
            schema['column_names'].append([-1, '*'])
            schema['column_names_original'].append([-1, '*'])
            schema['column_types'].append('text')
            schema['column_descriptions'][0] = [-1, '']
        
        table_idx = len(schema['table_names_original'])
        
        # Add table name
        schema['table_names'].append(table_name)  # DAILSQL needs this
        schema['table_names_original'].append(table_name)
        schema['db_stats']['No. of tables'] += 1
        
        # Add columns with descriptions
        for col_name, col_type in zip(table_info['column_names'], table_info['column_types']):
            if split in ["neutron", "nova"]:
                col_name = col_name.lower()
            schema['column_names'].append([table_idx, col_name])
            schema['column_names_original'].append([table_idx, col_name])
            schema['column_types'].append(col_type)
            # Add empty column description in correct format [table_idx, '']
            schema['column_descriptions'][len(schema['column_names']) - 1] = [table_idx, '']
            schema['db_stats']['No. of columns'] += 1
        
        # Add sample rows if available
        if 'example_rows' in table_info and table_info['example_rows']:
            # Convert rows to dict format expected by DAILSQL
            rows_data = []
            col_names = table_info['column_names']
            for row in table_info['example_rows'][:3]:  # Limit to 3 rows
                if isinstance(row, list):
                    row_dict = {}
                    for i in range(min(len(col_names), len(row))):
                        val = row[i]
                        # Handle NaN values
                        if isinstance(val, float) and (val != val):
                            val = None
                        row_dict[col_names[i]] = val
                    rows_data.append(row_dict)
            if rows_data:
                schema['sample_rows'][table_name] = rows_data
        
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

                    
                    if split == "dw" or split == "dw_real":
                        # Find source column index
                        source_idx = None
                        for idx, (tbl_idx, col_name) in enumerate(schema['column_names']):
                            if tbl_idx == table_idx and col_name == source_col:
                                source_idx = idx
                                break
                    
                        # Find target column index
                        target_idx = None
                        target_table_idx = None
                        try:
                            target_table_idx = schema['table_names_original'].index(target_table)
                        except ValueError:
                            # Target table might not be in filtered set, skip
                            continue
                        
                        if target_table_idx is not None:
                            for idx, (tbl_idx, col_name) in enumerate(schema['column_names']):
                                if tbl_idx == target_table_idx and col_name == target_col:
                                    target_idx = idx
                                    break
                        
                        if source_idx is not None and target_idx is not None:
                            schema['foreign_keys'].append([source_idx, target_idx])


                        if source_idx is not None and target_idx is not None:
                            schema['foreign_keys'].append([source_idx, target_idx])


                    elif split in ["sp", "neutron", "nova"]:
                        # Case insensitive lookup for target table
                        target_table_idx = None
                        target_idx = None
                        try:
                            # Try exact match first
                            target_table_idx = schema['table_names_original'].index(target_table)
                        except ValueError:
                            # Try case insensitive match
                            for i, name in enumerate(schema['table_names_original']):
                                if name.lower() == target_table.lower():
                                    target_table_idx = i
                                    break
                        
                        if target_table_idx is not None:
                            for idx, (tbl_idx, col_name) in enumerate(schema['column_names']):
                                if tbl_idx == target_table_idx and col_name == target_col:
                                    target_idx = idx
                                    break
                        
                        if source_idx is not None and target_idx is not None:
                            schema['foreign_keys'].append([source_idx, target_idx])
    
    # Convert to list format
    dailsql_format = list(db_schemas.values())
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dailsql_format, f, indent=2)
    
    print(f"  Converted {len(beaver_tables)} tables into {len(dailsql_format)} databases")
    print(f"  Saved to: {output_path}")
    
    return dailsql_format


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

def convert_beaver_questions_to_dailsql_format(dataset, beaver_questions_path, output_path, option=1, templates=None):
    """
    Convert Beaver's dev_dw.json format to DAILSQL's expected format
    
    Args:
        beaver_questions_path: Path to dev_dw.json
        output_path: Path to save the converted questions
        option: Preprocessing option (1, 2, or 3)
            1: Base information with top k tables
            2: Base information with gold tables + mapping + join keys
            3: Base information with full context (mapping, join keys, external info, subqueries)
        templates: Optional templates dict for option 3
    
    Beaver format: list of dicts with:
        - question, db_id, sql, oracle_sql, gold_tables, mapping, join_keys
    
    DAILSQL format: list of dicts with:
        - instance_id, question, question_toks, db_id, query (gold_sql)
        - For option 3+: additional mapping information in question
        - For option 4: additional join keys information in question
        - For option 5: additional external knowledge, subquery questions, and template hints
    """
    with open(beaver_questions_path, 'r') as f:
        beaver_questions = json.load(f)
    
    # Load spacy for tokenization
    nlp = spacy.load("en_core_web_sm")

    retrieved_tables = None
    if option == 1:
        retrieved_tables = get_retrieved_tables(dataset)
    
    dailsql_format = []
    for question_info in tqdm(beaver_questions):
        question_text = question_info['question']

        # Option 1+: Add gold tables to question
        if option >= 1:
            if option == 1 and retrieved_tables is not None:
                 gold_tables = retrieved_tables[question_info['id']]
            else:
                 gold_tables = question_info.get('tables', [])
            # remove the #sep# prefix (generic)
            display_gold_tables = [table.split('#sep#')[1] if '#sep#' in table else table for table in gold_tables]
            
            if display_gold_tables:
                question_text = (
                    f"{question_text}\n\n"
                    f"[Gold Tables]\n{', '.join(display_gold_tables)}"
                )
        
        # Option 2+: Add mapping to question
        if option >= 2:
            mapping = question_info.get('column_mapping', {})
            if mapping:
                mapping_str = format_mapping_for_prompt(mapping)
                question_text = (
                    f"{question_text}\n\n"
                    f"[Schema Mapping Hints]\n{mapping_str}"
                )
        
        # Option 2+: Add join keys
        if option >= 2:
            join_keys = question_info.get('join_keys', [])
            if join_keys:
                join_keys_str = format_join_keys_for_prompt(join_keys)
                question_text = (
                    f"{question_text}\n\n"
                    f"[Join Key Hints]\n{join_keys_str}"
                )

        # Option 3: Add external knowledge and subquery info
        if option >= 3:
            domain_knowledge = question_info.get('domain_knowledge', [])
            sub_questions = question_info.get('sub_questions', [])
            
            if domain_knowledge:
                question_text += "\n\n-- Domain Knowledge (database-wide):\n"
                question_text += "\n".join(domain_knowledge)
                question_text += "\n"
                question_text += "You should use the domain knowledge to help determine which tables and columns to use in the SQL statement as well as constructing the SQL statement."

            if sub_questions:
                question_text += "\n\n-- Subquery Gold Questions (database-wide):\n"
                question_text += "\n".join(sub_questions)
                question_text += "\n"
                question_text += "You must answer each subquery individually and then combine them to form the complete SQL statement. Each subquery you generate must be explicitly used in the final query you generate; do not simplify the subqueries you generate for implementation in the final query."
            
            if templates:
                detailed_category = question_info.get('detailed_category')
                if detailed_category and detailed_category != 'real' and detailed_category in templates:
                    question_text += "\n\n Here is an explanation of which numbered subqueries you are given correspond to which query in the query structure you were provided:\n"
                    question_text += f"{templates[detailed_category]['structure']}\n\n"
                    question_text += templates[detailed_category]['subquery_decomposition']
        
        # Tokenize question
        doc = nlp(question_text)
        question_toks = [token.text for token in doc]
        query = process_sql(question_info.get('sql', ''))
        
        base_item = {
            'id': question_info['id'],
            'question': question_text,
            'question_toks': question_toks,
            'db': question_info['db'],
            'query': query,
            'No. of candidate columns': 0,  # Will be updated
            'No. of gold tables': 0,  # Will be updated
            'column_descriptions': {},  # Required by DAILSQL prompt formatter
            'external_knowledge': None,  # Required field for DAILSQL
            'special_function': None,  # Required field for DAILSQL
            'plan': None  # Required field for DAILSQL
        }
        
        # Option 1+: Store gold_tables for per-question schema filtering
        if option >= 1:
            # Use lowercase for logic to match schema
            if option == 1:
                raw_gold_tables = retrieved_tables[question_info['id']]
            else:
                raw_gold_tables = question_info.get('tables', [])

            if 'dw' in question_info['db']:
                # do not lower case for dw
                base_item["gold_tables"] = [table.replace(f"{question_info['db']}#sep#", '') for table in raw_gold_tables]
                base_item["gold_tables"] = [table.replace('dw#sep#', '') for table in base_item["gold_tables"]]
            else:
                base_item["gold_tables"] = [t.split('#sep#')[1].lower() if '#sep#' in t else t.lower() for t in raw_gold_tables]
        else:
            # Option None: No filtering, include all tables
            base_item["gold_tables"] = []
        
        # Store original data for evaluation
        if option >= 2:
            base_item['column_mapping'] = question_info.get('column_mapping', {})
        
        if option >= 2:
            base_item['join_keys'] = question_info.get('join_keys', [])
        
        dailsql_format.append(base_item)
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dailsql_format, f, indent=2)
    
    print(f"  Converted {len(beaver_questions)} questions")
    print(f"  Saved to: {output_path}")
    
    return dailsql_format

def process_sql(sql):
    sql = sql.strip()
    # Remove any db_id#sep# prefix generic
    sql = re.sub(r'\w+#sep#', '', sql)
    sql = sql.replace('\n', ' ')      
    sql = sql.replace('"', '')
    sql = ' '.join(sql.split())   
    return sql


def collect_all_gold_tables(questions):
    """Collect all unique gold tables from all questions"""
    all_gold_tables = set()
    for q in questions:
        gold_tables = q.get('tables', [])
        all_gold_tables.update(gold_tables)
    return all_gold_tables


if __name__ == '__main__':
    import argparse
    import random
    parser = argparse.ArgumentParser(description='Preprocess Beaver dataset for DAILSQL with different options')
    parser.add_argument('--beaver_dir', default='../../data', type=str, 
                        help='Directory containing Beaver dataset files')
    parser.add_argument('--option', type=int, required=True, choices=[1, 2, 3],
                        help='Preprocessing option: 1=top20_tables, 2=+gold_tables+mapping+join_keys, 3=+external')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of questions to process (for testing)')
    parser.add_argument('--random', action='store_true',
                        help='Randomly sample questions when using --limit')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible sampling (default: 42)')
    parser.add_argument('--questions_file', type=str, default=None,
                        help='Specific path to questions file (overrides beaver_dir default)')
    parser.add_argument('--tables_file', type=str, default=None,
                        help='Specific path to tables file (overrides beaver_dir default)')
    parser.add_argument('--dataset', type=str, default='dw',
                        help='Split to preprocess (default: dw)')
    args = parser.parse_args()
    
    proj_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    beaver_base_dir = osp.join(proj_dir, args.beaver_dir)

    subdir_name = f'beaver_{args.dataset}_opt{args.option}'
    output_base_dir = osp.join(proj_dir, 'preprocessed_data', subdir_name)
    
    print("=" * 70)
    print(f"BEAVER PREPROCESSING FOR DAILSQL - OPTION {args.option}")
    print("=" * 70)
    
    option_descriptions = {
        1: "Base information (question) with top k tables available",
        2: "Base information with gold tables + mapping + join key hints",
        3: "Base information with full context (mapping, join keys, external info, subqueries)"
    }
    print(f"Option {args.option}: {option_descriptions[args.option]}")
    print("=" * 70)
    
    if args.questions_file:
        beaver_questions_path = args.questions_file
    else:
        # beaver_questions_path = osp.join(beaver_base_dir, 'dev_dw.json')
        beaver_questions_path = osp.join(beaver_base_dir, 'dev_dw_new', 'final_combined_with_source_sampled.json')
    
    print("\n[1/3] Loading questions...")
    print(f"  Reading from: {beaver_questions_path}")
    
    with open(beaver_questions_path, 'r') as f:
        questions = json.load(f)
    
    if args.limit:
        if args.random:
            # Set random seed for reproducibility
            random.seed(args.seed)
            # Randomly sample questions
            questions = random.sample(questions, min(args.limit, len(questions)))
            print(f"  Randomly sampled {len(questions)} questions (seed={args.seed})")
        else:
            questions = questions[:args.limit]
            print(f"  Limited to first {args.limit} questions")
    
    # Convert tables
    # For all options: Create full database schema with all tables
    # Per-question filtering will be applied during prompt generation based on gold_tables
    if args.tables_file:
        beaver_tables_path = args.tables_file
    else:
        beaver_tables_path = osp.join(beaver_base_dir, 'dev_tables_new.json')
    output_tables_path = osp.join(output_base_dir, 'tables_preprocessed.json')
    
    print(f"\n[2/3] Converting tables...")
    print(f"  For Option {args.option}: Creating full database schema")
    print(f"  Per-question filtering will be applied during prompt generation")
    convert_beaver_tables_to_dailsql_format(
        beaver_tables_path, 
        output_tables_path,
        gold_tables_filter=None,  # Include all tables for all options
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

    dailsql_questions = convert_beaver_questions_to_dailsql_format(
        args.dataset,
        beaver_questions_path, 
        output_questions_path,
        option=args.option,
        templates=templates
    )
    
    # Show sample of the first question
    if dailsql_questions:
        print("\n" + "=" * 70)
        print("SAMPLE OUTPUT (First Question)")
        print("=" * 70)
        import pprint
        pprint.pprint(dailsql_questions[0])
    
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
