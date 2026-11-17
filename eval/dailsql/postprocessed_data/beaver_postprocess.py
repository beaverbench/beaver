"""
Postprocess script for Beaver dataset with MySQL support
- Executes generated SQL on MySQL database
- Saves generated SQL and results for each question
- Creates subdirectory structure for organization
"""
import os
import json
import os.path as osp
import argparse
import sys
import mysql.connector
import csv
import re
import signal


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Query execution timed out")

proj_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path = [osp.join(proj_dir, '../')] + sys.path


def load_mysql_credentials(credential_path, dataset=None):
    """Load MySQL credentials from JSON file or environment variables."""
    if credential_path and os.path.exists(credential_path):
        with open(credential_path, 'r') as f:
            return json.load(f)
    # Fallback to environment variables
    host = os.environ.get("MYSQL_HOST")
    user = os.environ.get("MYSQL_USER")
    password = os.environ.get("MYSQL_PASSWORD")
    if dataset == "dw_real":
        db_id = "dw"
    else:
        db_id = dataset
    if host and user and password:
        return {"host": host, "user": user, "password": password, "database": db_id}
    raise FileNotFoundError(f"MySQL credentials not found at '{credential_path}' and MYSQL_HOST/USER/PASSWORD env vars not set.")


def execute_mysql_query(query, credentials, timeout=10):
    """Execute a SQL query on MySQL and return results with timeout"""
    try:
        # Set up timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)  # Set timeout in seconds
        
        # Connect to MySQL
        conn = mysql.connector.connect(
            host=credentials['host'],
            user=credentials['user'],
            password=credentials['password'],
            database=credentials['database']
        )
        cursor = conn.cursor()
        
        # Execute query
        # Set MySQL server-side timeout (in milliseconds)
        # This is more robust than client-side signals for long running queries (like sorts)
        try:
            cursor.execute(f"SET SESSION max_execution_time={timeout * 1000}")
        except Exception as e:
            print(f"  Warning: Could not set max_execution_time: {e}")

        print(f"  > Executing query with {timeout}s server timeout...")
        cursor.execute(query)
        print("  > Query executed. Fetching results...")
        
        # Fetch results with safety limit
        MAX_ROWS = 20000 
        batch_size = 100
        results = []
        
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            results.extend(rows)
            if len(results) > MAX_ROWS:
                # Cancel alarm before cleaning up
                signal.alarm(0)
                try:
                    cursor.close()
                    conn.close()
                except:
                    pass
                return False, f"Query result exceeded maximum row limit ({MAX_ROWS})", []
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description] if cursor.description else []
        
        # Cancel the alarm
        signal.alarm(0)
        
        cursor.close()
        conn.close()
        
        return True, results, column_names
        
    except TimeoutError:
        # Don't cancel alarm immediately, or set a new one for cleanup
        # The connection might be stuck, so closing it might also hang.
        try:
            # Set a short timeout for cleanup
            signal.alarm(5)
            try:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()
            except Exception:
                pass
        finally:
            signal.alarm(0)  # Ensure alarm is cancelled
            
        return False, f"Query execution exceeded {timeout} second timeout", []
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        return False, str(e), []


def save_results_to_csv(results, column_names, output_path):
    """Save query results to CSV file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        if column_names:
            writer.writerow(column_names)
        
        # Write data rows
        for row in results:
            writer.writerow(row)


def robust_clean_sql(sql, schema_tables):
    """
    Clean SQL more robustly:
    1. Remove prefixes like dw#sep#
    2. Replace Uppercase Table Names with lowercase if they exist in schema_tables (which are lowercase)
    """
    # 1. Remove prefixes
    sql = re.sub(r'\b\w+#sep#(\w+)', r'\1', sql)
    
    # 2. Case-insensitive replacement of table names
    # Sort tables by length desc to avoid partial replacements
    sorted_tables = sorted(schema_tables, key=len, reverse=True)
    
    for table in sorted_tables:
        pattern = re.compile(r'\b' + re.escape(table) + r'\b', re.IGNORECASE)
        sql = pattern.sub(table, sql)
        
    return sql

def postprocess_sql(sql, schema_tables=None):
    """Clean and postprocess SQL query"""
    # Remove markdown code blocks if present
    sql = re.sub(r'```sql\s*', '', sql)
    sql = re.sub(r'```\s*', '', sql)
    
    # Remove extra whitespace
    sql = sql.strip()
    
    # Remove trailing semicolons (MySQL connector handles this)
    sql = sql.rstrip(';')
    
    # Robust casing fix
    if schema_tables:
         sql = robust_clean_sql(sql, schema_tables)
    
    return sql


def main_postprocess(dataset, root_path, dev_json, mysql_credential_path, option, tables_json_path=None):
    """
    Main postprocessing function for Beaver dataset
    
    Args:
        dataset: dataset
        root_path: Directory containing LLM-generated SQL files
        dev_json: Path to preprocessed questions JSON
        mysql_credential_path: Path to MySQL credentials JSON
        option: Which option (1-5) is being processed
        tables_json_path: Path to tables JSON for casing fix
    """
    # Load schema tables if provided
    schema_tables = []
    if tables_json_path and os.path.exists(tables_json_path):
        print(f"Loading tables from {tables_json_path} for casing fix...")
        with open(tables_json_path, "r") as f:
            tables_data = json.load(f)
            
        if isinstance(tables_data, list):
            # Spider format
            for db in tables_data:
                schema_tables.extend(db['table_names_original'])
        elif isinstance(tables_data, dict):
            # Beaver format
            for table_key, table_info in tables_data.items():
                if 'table_name' in table_info:
                    schema_tables.append(table_info['table_name'])
                else:
                     schema_tables.append(table_key)
        
        # Lowercase all
        if 'dw' not in tables_json_path:
            schema_tables = [str(t).lower() for t in schema_tables]
        else:
            schema_tables = [str(t) for t in schema_tables]
    print("=" * 70)
    print(f"BEAVER POSTPROCESSING FOR DAILSQL - OPTION {option}")
    print("=" * 70)

    # Load credentials
    credentials = load_mysql_credentials(mysql_credential_path, dataset)
    print(f"Loaded MySQL credentials for database: {credentials['database']}")
    
    # Load dev data
    with open(dev_json, 'r') as f:
        dev_data = json.load(f)
    
    print(f"Loaded {len(dev_data)} questions from {dev_json}")
    
    # Process each question
    success_count = 0
    error_count = 0
    
    for idx, item in enumerate(dev_data):
        instance_id = item['id']
        print(f"\n[{idx+1}/{len(dev_data)}] Processing {instance_id}...")
        
        # Create output directory for this instance
        instance_dir = osp.join(root_path, instance_id)
        os.makedirs(instance_dir, exist_ok=True)
        
        # Read generated SQL (try with @0 suffix first, then without)
        sql_file = osp.join(root_path, f"{instance_id}@0.sql")
        if not os.path.exists(sql_file):
            sql_file = osp.join(root_path, f"{instance_id}.sql")
        
        if not os.path.exists(sql_file):
            print(f"  WARNING: SQL file not found: {sql_file}")
            error_count += 1
            continue
        
        with open(sql_file, 'r') as f:
            generated_sql = f.read()
        
        # Postprocess SQL
        generated_sql = postprocess_sql(generated_sql, schema_tables)
        
        # Save cleaned SQL to instance directory
        instance_sql_path = osp.join(instance_dir, f"{instance_id}.sql")
        with open(instance_sql_path, 'w') as f:
            f.write(generated_sql)
        
        # Execute SQL and save results
        success, results, column_names = execute_mysql_query(generated_sql, credentials)
        
        if success:
            # Save results to CSV
            results_csv_path = osp.join(instance_dir, "results.csv")
            save_results_to_csv(results, column_names, results_csv_path)
            print(f"  ✓ Executed successfully, saved {len(results)} rows to results.csv")
            success_count += 1
        else:
            # Save error message
            error_path = osp.join(instance_dir, "error.txt")
            with open(error_path, 'w') as f:
                f.write(f"SQL Execution Error:\n{results}")
            print(f"  ✗ Execution failed: {results[:100]}...")
            error_count += 1
    
    print("\n" + "=" * 70)
    print("POSTPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Success: {success_count}/{len(dev_data)}")
    print(f"Errors: {error_count}/{len(dev_data)}")
    print(f"Output directory: {root_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dw', type=str)
    parser.add_argument('--option', type=int, required=True, choices=[1, 2, 3],
                        help='Preprocessing option (1-3)')
    parser.add_argument('--model', default='gpt-4.1', type=str)
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--max_tokens', type=int, default=200)
    parser.add_argument('--mysql_credential', default=None,
                        type=str, help='Path to MySQL credentials JSON (optional, falls back to env vars)')
    parser.add_argument('--dev_file', type=str, default=None, help='Path to preprocessed dev questions json')
    parser.add_argument('--dev', type=str, default=None, help='Dev dataset name (for folder naming)')
    parser.add_argument('--tables_file', type=str, default=None, help='Path to tables json (for casing fix)')
    parser.add_argument('--results_dir', type=str, default=None, help="Folder of this run's results, as given to ask_llm.py")
    args = parser.parse_args()

    # Construct path based on dev name if provided, else output default
    if args.dev:
        # Match generate_question.py: {comment}_{dev}_CTX-{max_tokens}
        folder_name = f"{args.comment}_{args.dev}_CTX-{args.max_tokens}"
    else:
        # Fallback to old behavior
        folder_name = f"{args.comment}_beaver_opt{args.option}_CTX-{args.max_tokens}"

    root_path = osp.join(proj_dir, args.results_dir or f'postprocessed_data/{folder_name}/RESULTS_MODEL-{args.model}-SQL')
    
    if args.dev_file:
         dev_json = args.dev_file
    else:
         dev_json = osp.join(proj_dir, f'preprocessed_data/beaver_opt{args.option}/beaver_opt{args.option}_preprocessed.json')
    
    if args.mysql_credential is None:
        mysql_credential_path = None
    else:
        mysql_credential_path = osp.join(proj_dir, args.mysql_credential)

    main_postprocess(args.dataset, root_path, dev_json, mysql_credential_path, args.option, args.tables_file)
