from typing import Any, Dict, List

from pydantic import BaseModel

from runner.database_manager import DatabaseManager
from runner.task import Task
from workflow.sql_meta_info import SQLMetaInfo


class SystemState(BaseModel):
    """
    Represents the state of a question during execution. CHESS's gold-SQL schema check
    (check_schema_status) and the unit-tester and column-description fields are removed.
    """

    executing_tool: str = ""
    task: Task
    tentative_schema: Dict[str, List[str]]
    execution_history: List[Any]

    keywords: List[str] = []

    similar_columns: Dict[str, List[str]] = {}
    schema_with_examples: Dict[str, Dict[str, List[str]]] = {}

    SQL_meta_infos: Dict[str, List[SQLMetaInfo]] = {}
    errors: Dict[str, str] = {}

    def add_columns_to_tentative_schema(self, selected_columns: Dict[str, List[str]]) -> None:
        for table_name, columns in selected_columns.items():
            target_table_name = next((t for t in self.tentative_schema.keys() if t.lower() == table_name.lower()), None)
            if target_table_name:
                for column in columns:
                    if column.lower() not in [c.lower() for c in self.tentative_schema[target_table_name]]:
                        self.tentative_schema[target_table_name].append(column)
            else:
                self.tentative_schema[table_name] = columns

    def add_connections_to_tentative_schema(self):
        DatabaseManager().add_connections_to_tentative_schema(self.tentative_schema)

    def get_schema_string(self, schema_type: str = "tentative") -> str:
        if schema_type == "tentative":
            schema = self.tentative_schema
        elif schema_type == "complete":
            schema = DatabaseManager().get_db_schema()
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")

        return DatabaseManager().get_database_schema_string(schema, self.schema_with_examples)
