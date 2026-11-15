import re
from enum import Enum
from typing import Any, Dict, List, Union

import mysql.connector
import sqlparse

# MySQL replacement for CHESS's SQLite execute_sql. Each call opens its own connection, so worker threads
# never share one. MAX_EXECUTION_TIME makes the server abort a slow SELECT instead of leaving it running.

SQL_TIMEOUT = 30
CONNECTION_TIMEOUT = 10

# Generated SQL is run read-only: it must be a single query, and the session refuses writes. The single-
# statement check matters because mysql-connector 9 runs every statement of a multi-statement string,
# and a later one could switch the session back to read-write.
_QUERY_PREFIX = re.compile(r"^[\s(]*(SELECT|WITH)\b", re.IGNORECASE)


def execute_sql(mysql_creds: Dict[str, Any], sql: str, fetch: Union[str, int] = "all", timeout: int = SQL_TIMEOUT) -> Any:
    if not _QUERY_PREFIX.match(sql):
        raise ValueError("Only SELECT/WITH queries can be executed.")
    if len([s for s in sqlparse.split(sql) if s.strip().strip(";").strip()]) > 1:
        raise ValueError("Only a single SQL statement can be executed.")
    conn = mysql.connector.connect(**mysql_creds, connection_timeout=CONNECTION_TIMEOUT)
    try:
        cursor = conn.cursor(buffered=True)
        cursor.execute("SET SESSION TRANSACTION READ ONLY")
        cursor.execute(f"SET SESSION MAX_EXECUTION_TIME = {int(timeout * 1000)}")
        cursor.execute(sql)
        if fetch == "all":
            return cursor.fetchall()
        elif fetch == "one":
            return cursor.fetchone()
        elif isinstance(fetch, int):
            return cursor.fetchmany(fetch)
        raise ValueError("Invalid fetch argument. Must be 'all', 'one', or an integer.")
    finally:
        conn.close()


class ExecutionStatus(Enum):
    SYNTACTICALLY_CORRECT = "SYNTACTICALLY_CORRECT"
    EMPTY_RESULT = "EMPTY_RESULT"
    SYNTACTICALLY_INCORRECT = "SYNTACTICALLY_INCORRECT"


def get_execution_status(execution_result: List = None, error: Exception = None) -> ExecutionStatus:
    if error is not None:
        return ExecutionStatus.SYNTACTICALLY_INCORRECT
    if (execution_result is None) or (execution_result == []):
        return ExecutionStatus.EMPTY_RESULT
    return ExecutionStatus.SYNTACTICALLY_CORRECT
