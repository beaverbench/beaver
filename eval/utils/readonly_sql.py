import re
from typing import Any, Dict, List, Optional, Tuple

import mysql.connector
import sqlparse

# Read-only execution of model-generated SQL, shared by CHESS and Claude Code. Each call opens its own
# connection, so worker threads never share one. MAX_EXECUTION_TIME makes the server abort a slow SELECT
# instead of leaving it running.

SQL_TIMEOUT = 30
CONNECTION_TIMEOUT = 10

# Generated SQL is run read-only: it must be a single query, and the session refuses writes. The single-
# statement check matters because mysql-connector 9 runs every statement of a multi-statement string,
# and a later one could switch the session back to read-write. Leading comments (-- , # and /* */) are
# skipped, and SHOW / DESCRIBE / EXPLAIN are allowed besides queries.
# A line comment must run to its line end, so a keyword inside the comment cannot satisfy the check.
_LEADING = r"(?:\s|\(|--[^\n]*\n|#[^\n]*\n|/\*.*?\*/)*"
_FLAGS = re.IGNORECASE | re.DOTALL
_QUERY_PREFIX = re.compile(rf"^{_LEADING}(SELECT|WITH|SHOW|DESCRIBE|DESC|EXPLAIN)\b", _FLAGS)
# EXPLAIN ANALYZE (DESCRIBE is a synonym of EXPLAIN) runs the statement it explains, so it must explain a query
_EXPLAIN_ANALYZE = re.compile(rf"^{_LEADING}(?:EXPLAIN|DESCRIBE|DESC)\s+ANALYZE\b", _FLAGS)
_EXPLAIN_ANALYZE_QUERY = re.compile(
    rf"^{_LEADING}(?:EXPLAIN|DESCRIBE|DESC)\s+ANALYZE\s+(?:FORMAT\s*=\s*\w+\s+)?{_LEADING}(SELECT|WITH)\b", _FLAGS)


def execute_readonly(mysql_creds: Dict[str, Any], sql: str, timeout: int = SQL_TIMEOUT,
                     max_rows: Optional[int] = None) -> Tuple[List[str], List[tuple], int]:
    """Returns the column names, the first max_rows rows (all when None) and the total row count."""
    if not _QUERY_PREFIX.match(sql) or (_EXPLAIN_ANALYZE.match(sql) and not _EXPLAIN_ANALYZE_QUERY.match(sql)):
        raise ValueError("Only SELECT/WITH queries (or SHOW/DESCRIBE/EXPLAIN) can be executed.")
    if len([s for s in sqlparse.split(sql) if s.strip().strip(";").strip()]) > 1:
        raise ValueError("Only a single SQL statement can be executed.")
    conn = mysql.connector.connect(**mysql_creds, connection_timeout=CONNECTION_TIMEOUT)
    try:
        cursor = conn.cursor(buffered=True)
        cursor.execute("SET SESSION TRANSACTION READ ONLY")
        cursor.execute(f"SET SESSION MAX_EXECUTION_TIME = {int(timeout * 1000)}")
        cursor.execute(sql)
        columns = [d[0] for d in cursor.description or []]
        rows = cursor.fetchall() if max_rows is None else cursor.fetchmany(max_rows)
        return columns, rows, cursor.rowcount
    finally:
        conn.close()
