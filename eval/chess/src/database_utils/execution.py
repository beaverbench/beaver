from enum import Enum
from typing import Any, Dict, List, Union

from utils.readonly_sql import SQL_TIMEOUT, execute_readonly

# MySQL replacement for CHESS's SQLite execute_sql, over the read-only execution in eval/utils/readonly_sql.py.


def execute_sql(mysql_creds: Dict[str, Any], sql: str, fetch: Union[str, int] = "all", timeout: int = SQL_TIMEOUT) -> Any:
    if fetch == "all":
        return execute_readonly(mysql_creds, sql, timeout)[1]
    elif fetch == "one":
        rows = execute_readonly(mysql_creds, sql, timeout, max_rows=1)[1]
        return rows[0] if rows else None
    elif isinstance(fetch, int):
        return execute_readonly(mysql_creds, sql, timeout, max_rows=fetch)[1]
    raise ValueError("Invalid fetch argument. Must be 'all', 'one', or an integer.")


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
