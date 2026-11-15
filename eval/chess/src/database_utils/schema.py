from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

# CHESS database_utils/schema.py without the BIRD description fields (column/value descriptions, value statistics)


@dataclass
class ColumnInfo:
    type: str = ""
    examples: List[str] = field(default_factory=list)
    primary_key: bool = False
    foreign_keys: List[Tuple[str, str]] = field(default_factory=list)
    referenced_by: List[Tuple[str, str]] = field(default_factory=list)


def set_field(column_info: ColumnInfo, field_name: str, value: Any) -> None:
    if field_name in column_info.__dataclass_fields__:
        setattr(column_info, field_name, value)
    else:
        raise ValueError(f"{field_name} is not a valid field of ColumnInfo")


@dataclass
class TableSchema:
    columns: Dict[str, ColumnInfo] = field(default_factory=dict)


def get_primary_keys(table_schema: TableSchema) -> List[str]:
    return [name for name, info in table_schema.columns.items() if info.primary_key]


@dataclass
class DatabaseSchema:
    tables: Dict[str, TableSchema] = field(default_factory=dict)

    @classmethod
    def from_schema_dict(cls, schema_dict: Dict[str, List[str]]) -> "DatabaseSchema":
        return cls(tables={
            table_name: TableSchema(columns={column_name: ColumnInfo() for column_name in column_names})
            for table_name, column_names in schema_dict.items()
        })

    @classmethod
    def from_schema_dict_with_examples(cls, schema_dict_with_info: Dict[str, Dict[str, List[str]]]) -> "DatabaseSchema":
        return cls(tables={
            table_name: TableSchema(columns={
                column_name: ColumnInfo(examples=column_info)
                for column_name, column_info in column_dict.items()
            })
            for table_name, column_dict in schema_dict_with_info.items()
        })

    def get_actual_table_name(self, table_name: str) -> Optional[str]:
        table_name_lower = table_name.lower()
        return next((name for name in self.tables if name.lower() == table_name_lower), None)

    def get_table_info(self, table_name: str) -> Optional[TableSchema]:
        actual_name = self.get_actual_table_name(table_name)
        return self.tables.get(actual_name)

    def get_actual_column_name(self, table_name: str, column_name: str) -> Optional[str]:
        table_info = self.get_table_info(table_name)
        if table_info:
            column_name_lower = column_name.lower()
            return next((name for name in table_info.columns if name.lower() == column_name_lower), None)
        return None

    def get_column_info(self, table_name: str, column_name: str) -> Optional[ColumnInfo]:
        actual_table_name = self.get_actual_table_name(table_name)
        actual_name = self.get_actual_column_name(table_name, column_name)
        if actual_name:
            return self.tables[actual_table_name].columns[actual_name]
        return None

    def subselect_schema(self, selected_database_schema: "DatabaseSchema") -> "DatabaseSchema":
        """Keeps the selected tables and columns that exist in this schema, under their actual names."""
        new_schema = DatabaseSchema({})
        for table_name, table_info in selected_database_schema.tables.items():
            actual_table_name = self.get_actual_table_name(table_name)
            if actual_table_name is None:
                continue
            new_table_info = TableSchema()
            for column_name, column_info in table_info.columns.items():
                actual_column_name = self.get_actual_column_name(table_name, column_name)
                if actual_column_name is None:
                    continue
                new_table_info.columns[actual_column_name] = column_info
            new_schema.tables[actual_table_name] = new_table_info
        return new_schema

    def add_info_from_schema(self, schema: "DatabaseSchema", field_names: List[str]) -> None:
        for table_name, table_info in self.tables.items():
            actual_table_name = schema.get_actual_table_name(table_name)
            if actual_table_name is None:
                continue
            for column_name, column_info in table_info.columns.items():
                actual_column_name = schema.get_actual_column_name(table_name, column_name)
                if actual_column_name is None:
                    continue
                new_column_info = schema.tables[actual_table_name].columns[actual_column_name]
                for field_name in field_names:
                    set_field(column_info, field_name, getattr(new_column_info, field_name))

    def to_dict(self) -> Dict[str, List[str]]:
        return {table_name: list(table_info.columns.keys()) for table_name, table_info in self.tables.items()}
