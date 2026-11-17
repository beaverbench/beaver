import contextvars
import logging
import math
import random
from typing import Any, Dict, List, Optional

from database_utils.db_values import query_lsh
from database_utils.execution import execute_sql
from database_utils.schema import ColumnInfo, DatabaseSchema, TableSchema
from database_utils.schema_generator import DatabaseSchemaGenerator


class DatabaseCatalog:
    """Run-wide data shared by every question: dev_tables.json, the value index and the MySQL credentials."""

    def __init__(self, tables: Dict[str, Dict[str, Any]], lsh_index: Dict[str, Any], mysql_creds: Dict[str, Any]):
        self.tables = tables
        self.lsh_index = lsh_index
        self.mysql_creds = mysql_creds


def _clean_examples(examples: List[Any]) -> List[str]:
    # Mirrors CHESS's "SELECT DISTINCT col ... IS NOT NULL LIMIT 3", kept only when the first value is short.
    # example_columns can repeat values and uses NaN for NULL; MySQL SET values arrive as lists.
    values = []
    for v in examples:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        v = ",".join(map(str, v)) if isinstance(v, list) else str(v)
        if v not in values:
            values.append(v)
    values = values[:3]
    return values if values and len(values[0]) < 50 else []


class DatabaseManager:
    """
    One question's view of a BEAVER database, replacing CHESS's per-db_id singleton over a SQLite file.
    The schema comes from dev_tables.json, restricted to the question's allowed tables (the retrieved
    top-k in setting 0, the gold tables in settings 1/2), and join keys become foreign keys when given.
    Every schema, value-index and execution access in the workflow goes through this class, so nothing
    outside the allowed tables is shown to the model. SQL still runs against the full MySQL database.

    DatabaseManager() returns the current question's instance; it is context-local, so concurrent
    questions do not share one (threading_utils.ordered_concurrent_function_calls carries the context into its threads).
    """
    _current: contextvars.ContextVar = contextvars.ContextVar("database_manager")

    def __new__(cls):
        try:
            return cls._current.get()
        except LookupError:
            raise ValueError("DatabaseManager has not been opened for this question.")

    @classmethod
    def open(cls, catalog: DatabaseCatalog, db_id: str, allowed_tables: List[str],
             join_keys: Optional[List[List[str]]] = None, seed: Any = None) -> "DatabaseManager":
        self = object.__new__(cls)
        self.catalog = catalog
        self.db_id = db_id
        self.rng = random.Random(seed)
        self.db_schema = self._build_schema(allowed_tables, join_keys or [])
        self.allowed_tables = {t.lower() for t in self.db_schema.tables}
        cls._current.set(self)
        return self

    def _build_schema(self, allowed_tables: List[str], join_keys: List[List[str]]) -> DatabaseSchema:
        tables_lower = {name.lower(): name for name in self.catalog.tables}
        schema = DatabaseSchema()
        for table in allowed_tables:
            name = tables_lower.get(table.lower())
            if name is None:
                logging.warning(f"Table {table} is not in dev_tables.json; skipping it")
                continue
            if name in schema.tables:
                continue
            t = self.catalog.tables[name]
            schema.tables[name] = TableSchema(columns={
                column_name: ColumnInfo(type=column_type, examples=_clean_examples(examples))
                for column_name, column_type, examples in zip(t["column_names"], t["column_types"], t["example_columns"])
            })
        for left, right in join_keys:
            left_table, left_column = left.split(".", 1)
            right_table, right_column = right.split(".", 1)
            left_info = schema.get_column_info(left_table, left_column)
            right_info = schema.get_column_info(right_table, right_column)
            if left_info is None or right_info is None:
                logging.warning(f"Join key {left}={right} is outside the allowed tables; skipping it")
                continue
            right_ref = (schema.get_actual_table_name(right_table), schema.get_actual_column_name(right_table, right_column))
            left_ref = (schema.get_actual_table_name(left_table), schema.get_actual_column_name(left_table, left_column))
            if right_ref not in left_info.foreign_keys:
                left_info.foreign_keys.append(right_ref)
                right_info.referenced_by.append(left_ref)
        return schema

    def get_db_schema(self) -> Dict[str, List[str]]:
        return self.db_schema.to_dict()

    def query_lsh(self, keyword: str, signature_size: int = 100, n_gram: int = 3, top_n: int = 10) -> Dict[str, Dict[str, List[str]]]:
        return query_lsh(self.catalog.lsh_index, keyword, signature_size, n_gram, top_n, allowed_tables=self.allowed_tables)

    def execute_sql(self, sql: str, fetch="all") -> Any:
        return execute_sql(self.catalog.mysql_creds, sql, fetch)

    def _schema_generator(self, tentative_schema: Dict[str, List[str]],
                          schema_with_examples: Optional[Dict[str, Dict[str, List[str]]]] = None) -> DatabaseSchemaGenerator:
        return DatabaseSchemaGenerator(
            db_schema=self.db_schema,
            tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema),
            schema_with_examples=DatabaseSchema.from_schema_dict_with_examples(schema_with_examples) if schema_with_examples else None,
            rng=self.rng,
        )

    def get_column_profiles(self, schema_with_examples: Dict[str, Dict[str, List[str]]], with_keys: bool,
                            with_references: bool, tentative_schema: Dict[str, List[str]] = None) -> Dict[str, Dict[str, str]]:
        generator = self._schema_generator(tentative_schema if tentative_schema else self.get_db_schema(), schema_with_examples)
        return generator.get_column_profiles(with_keys, with_references)

    def get_database_schema_string(self, tentative_schema: Dict[str, List[str]],
                                   schema_with_examples: Dict[str, Dict[str, List[str]]]) -> str:
        return self._schema_generator(tentative_schema, schema_with_examples).generate_schema_string()

    def add_connections_to_tentative_schema(self, tentative_schema: Dict[str, List[str]]) -> None:
        """Adds join-key columns of the tentative tables in place (CHESS computed this but dropped the result)."""
        with_connections = self._schema_generator(tentative_schema).get_schema_with_connections()
        for table_name, columns in with_connections.items():
            target = next((t for t in tentative_schema if t.lower() == table_name.lower()), table_name)
            existing = tentative_schema.setdefault(target, [])
            for column in columns:
                if column.lower() not in [c.lower() for c in existing]:
                    existing.append(column)
