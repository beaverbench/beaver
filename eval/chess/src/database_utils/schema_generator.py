import random
import re
from typing import Dict, List, Optional

from database_utils.schema import DatabaseSchema

# CHESS database_utils/schema_generator.py over a schema built from dev_tables.json (see
# DatabaseManager) instead of a live SQLite database:
# - CREATE TABLE statements are composed from column names and types, not read from sqlite_master
# - columns without retrieved values (and date columns) show the dev_tables.json examples, not values queried
#   live; CHESS's live lists of all values for low-cardinality columns are dropped
# - value statistics and BIRD column descriptions are dropped
# so CHESS sees the same table metadata as the other baselines.

_SIMPLE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote(name: str) -> str:
    return name if _SIMPLE_IDENTIFIER.match(name) else f"`{name}`"


class DatabaseSchemaGenerator:
    def __init__(self, db_schema: DatabaseSchema, tentative_schema: Optional[DatabaseSchema] = None,
                 schema_with_examples: Optional[DatabaseSchema] = None, rng: Optional[random.Random] = None):
        # db_schema: every allowed table and column, with types, examples and join-key references
        self.db_schema = db_schema
        self.rng = rng or random.Random()
        self.schema_structure = tentative_schema or DatabaseSchema()
        self.schema_with_examples = schema_with_examples or DatabaseSchema()
        self._initialize_schema_structure()

    def _initialize_schema_structure(self) -> None:
        self._load_table_and_column_info()
        self._load_column_examples()

    def _load_table_and_column_info(self) -> None:
        self.schema_structure = self.db_schema.subselect_schema(self.schema_structure)
        self.schema_structure.add_info_from_schema(schema=self.db_schema,
                                                   field_names=["type", "primary_key", "foreign_keys", "referenced_by"])

    def _load_column_examples(self) -> None:
        self.schema_structure.add_info_from_schema(schema=self.schema_with_examples, field_names=["examples"])
        for table_name, table_schema in self.schema_structure.tables.items():
            for column_name, column_info in table_schema.columns.items():
                if (not column_info.examples) or (column_info.type.lower() == "date" or ("date" in column_name.lower())):
                    column_info.examples = self.db_schema.get_column_info(table_name, column_name).examples

    def _is_connection(self, table_name: str, column_name: str) -> bool:
        column_info = self.db_schema.get_column_info(table_name, column_name)
        if column_info is None:
            return False
        if column_info.primary_key:
            return True
        for target_table, _ in column_info.foreign_keys:
            if self.schema_structure.get_table_info(target_table):
                return True
        for target_table, _ in column_info.referenced_by:
            if self.schema_structure.get_table_info(target_table):
                return True
        for target_table_name, table_schema in self.schema_structure.tables.items():
            if table_name.lower() == target_table_name.lower():
                continue
            for target_column_name, target_column_info in table_schema.columns.items():
                if target_column_name.lower() == column_name.lower() and target_column_info.primary_key:
                    return True
        return False

    def _get_connections(self) -> Dict[str, List[str]]:
        connections = {}
        for table_name, table_schema in self.schema_structure.tables.items():
            connections[table_name] = []
            for column_name in self.db_schema.tables[table_name].columns:
                if self._is_connection(table_name, column_name):
                    connections[table_name].append(column_name)
        return connections

    def get_schema_with_connections(self) -> Dict[str, List[str]]:
        schema_structure_dict = self.schema_structure.to_dict()
        connections = self._get_connections()
        for table_name, connected_columns in connections.items():
            for column_name in connected_columns:
                if column_name.lower() not in [col.lower() for col in schema_structure_dict[table_name]]:
                    schema_structure_dict[table_name].append(column_name)
        return schema_structure_dict

    def _get_example_column_name_description(self, table_name: str, column_name: str) -> str:
        column_info = self.schema_structure.get_column_info(table_name, column_name)
        if column_info and column_info.examples:
            return f" -- Example Values: {', '.join([f'`{str(x)}`' for x in column_info.examples])}"
        return ""

    def generate_schema_string(self, shuffle_cols: bool = True, shuffle_tables: bool = True) -> str:
        """
        Generates CREATE TABLE statements for the tentative schema, with examples as column comments.
        Tables and column definitions are shuffled as in CHESS (seeded per question by DatabaseManager).
        """
        table_names = list(self.schema_structure.tables.keys())
        if shuffle_tables:
            self.rng.shuffle(table_names)
        ddl_commands = []
        for table_name in table_names:
            targeted_columns = self.schema_structure.tables[table_name].columns
            # (column name or None for a constraint, definition)
            definitions = [(column_name, f"{_quote(column_name)} {column_info.type}")
                           for column_name, column_info in self.db_schema.tables[table_name].columns.items()]
            for column_name, column_info in self.db_schema.tables[table_name].columns.items():
                for target_table, target_column in column_info.foreign_keys:
                    if self.schema_structure.get_table_info(target_table):
                        definitions.append((None, f"FOREIGN KEY ({_quote(column_name)}) REFERENCES {_quote(target_table)}({_quote(target_column)})"))
            if shuffle_cols:
                definitions = self.rng.sample(definitions, len(definitions))
            schema_lines = [f"CREATE TABLE {_quote(table_name)}", "("]
            for column_name, column_def in definitions:
                if column_name is None:
                    schema_lines.append(f"\t{column_def},")
                elif (column_name in targeted_columns) or self._is_connection(table_name, column_name):
                    schema_lines.append(f"\t{column_def}," + self._get_example_column_name_description(table_name, column_name))
            schema_lines.append(");")
            ddl_commands.append('\n'.join(schema_lines))
        return "\n\n".join(ddl_commands)

    def get_column_profiles(self, with_keys: bool = False, with_references: bool = False) -> Dict[str, Dict[str, str]]:
        column_profiles = {}
        for table_name, table_schema in self.schema_structure.tables.items():
            column_profiles[table_name] = {}
            for column_name, column_info in table_schema.columns.items():
                if with_keys or not (column_info.primary_key or column_info.foreign_keys or column_info.referenced_by):
                    column_profile = f"Table name: `{table_name}`\nOriginal column name: `{column_name}`\n"
                    if column_info.type:
                        column_profile += f"Data type: {column_info.type}\n"
                    if column_info.examples:
                        column_profile += f"Example of values in the column: {', '.join([f'`{str(x)}`' for x in column_info.examples])}\n"
                    if column_info.primary_key:
                        column_profile += "This column is a primary key.\n"
                    if with_references:
                        if column_info.foreign_keys:
                            column_profile += "This column references the following columns:\n"
                            for target_table, target_column in column_info.foreign_keys:
                                column_profile += f"    Table: `{target_table}`, Column: `{target_column}`\n"
                        if column_info.referenced_by:
                            column_profile += "This column is referenced by the following columns:\n"
                            for source_table, source_column in column_info.referenced_by:
                                column_profile += f"    Table: `{source_table}`, Column: `{source_column}`\n"
                    column_profiles[table_name][column_name] = column_profile
        return column_profiles
