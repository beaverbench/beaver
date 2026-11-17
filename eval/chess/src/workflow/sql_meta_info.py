from typing import Any, List, Optional

from pydantic import BaseModel, PrivateAttr

from database_utils.execution import ExecutionStatus, get_execution_status
from runner.database_manager import DatabaseManager


class SQLMetaInfo(BaseModel):
    SQL: str
    plan: str = ""
    chain_of_thought_reasoning: str = ""
    error: str = ""
    need_fixing: bool = False

    # Executed once on first access; CHESS re-ran the query for the status and again for the result
    _executed: bool = PrivateAttr(default=False)
    _execution_result: List[Any] = PrivateAttr(default=None)
    _execution_error: Optional[Exception] = PrivateAttr(default=None)

    def _execute(self) -> None:
        if not self._executed:
            try:
                self._execution_result = DatabaseManager().execute_sql(self.SQL, "all")
            except Exception as e:
                self._execution_error = e
            self._executed = True

    @property
    def execution_result(self) -> List[Any]:
        """The rows, or raises the execution error."""
        self._execute()
        if self._execution_error is not None:
            raise self._execution_error
        return self._execution_result

    @property
    def execution_status(self) -> ExecutionStatus:
        self._execute()
        return get_execution_status(self._execution_result, self._execution_error)
