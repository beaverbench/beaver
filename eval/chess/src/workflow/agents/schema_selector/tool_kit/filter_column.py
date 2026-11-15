import logging
from typing import Any, Dict, List, Optional, Tuple

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from runner.logger import Logger
from runner.database_manager import DatabaseManager
from workflow.system_state import SystemState
from workflow.agents.tool import Tool

class FilterColumn(Tool):
    """
    Tool for filtering columns based on profiles and updating the tentative schema.

    CHESS makes one LLM call per column, i.e. ~400 per question over BEAVER's top-15 tables. Here the columns
    are judged in batches, one call per table (split into chunks of max_columns_per_call), with the same
    column profiles, instructions and examples; each column still gets its own reasoning and Yes/No.
    A JSON schema requiring every column is sent as structured output where the endpoint supports it;
    columns missing from an answer are asked again once, and still-missing ones are dropped, as CHESS
    drops a column whose call fails.
    """

    def __init__(self, template_name: str = None, engine_config: str = None, parser_name: str = None,
                 max_columns_per_call: int = 50):
        super().__init__()
        self.template_name = template_name
        self.engine_config = engine_config
        self.parser_name = parser_name
        self.max_columns_per_call = max_columns_per_call
        self.num_calls = 0

    @staticmethod
    def _response_format(column_names: List[str]) -> Dict[str, Any]:
        column_schema = {
            "type": "object",
            "properties": {
                "chain_of_thought_reasoning": {"type": "string"},
                "is_column_information_relevant": {"type": "string", "enum": ["Yes", "No"]},
            },
            "required": ["chain_of_thought_reasoning", "is_column_information_relevant"],
            "additionalProperties": False,
        }
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "column_relevance",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {column_name: column_schema for column_name in column_names},
                    "required": column_names,
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _decisions(response: Optional[Dict[str, Any]], column_names: List[str]) -> Dict[str, bool]:
        """Maps the answered columns (matched case-insensitively, backticks stripped) to whether they are relevant."""
        decisions = {}
        if not isinstance(response, dict):
            return decisions
        by_lower = {name.lower(): name for name in column_names}
        for key, value in response.items():
            column_name = by_lower.get(str(key).strip().strip("`").lower())
            if column_name is None or not isinstance(value, dict):
                continue
            answer = value.get("is_column_information_relevant")
            if isinstance(answer, str):
                decisions[column_name] = answer.strip().lower() == "yes"
        return decisions

    def _run(self, state: SystemState):
        """
        Executes the column filtering process.

        Args:
            state (SystemState): The current system state.
        """

        column_profiles = DatabaseManager().get_column_profiles(
            schema_with_examples=state.schema_with_examples,
            with_keys=True,
            with_references=True,
            tentative_schema=state.tentative_schema
        )

        pending: List[Tuple[str, List[str]]] = []
        for table_name, columns in column_profiles.items():
            column_names = list(columns.keys())
            for start in range(0, len(column_names), self.max_columns_per_call):
                pending.append((table_name, column_names[start:start + self.max_columns_per_call]))

        chosen: Dict[Tuple[str, str], bool] = {}
        self.num_calls = 0
        # The first pass, then one more for the columns an answer left out
        for _ in range(2):
            if not pending:
                break
            list_of_kwargs = [{
                "QUESTION": state.task.question,
                "HINT": state.task.evidence,
                "COLUMN_PROFILES": "\n".join(column_profiles[table_name][column_name] for column_name in column_names),
            } for table_name, column_names in pending]

            response = async_llm_chain_call(
                prompt=get_prompt(template_name=self.template_name),
                engine=get_llm_chain(**self.engine_config),
                parser=get_parser(self.parser_name),
                request_list=list_of_kwargs,
                step=self.tool_name,
                sampling_count=1,
                response_formats=[self._response_format(column_names) for _, column_names in pending],
            )
            self.num_calls += len(pending)

            still_pending = []
            for (table_name, column_names), result in zip(pending, response):
                decisions = self._decisions(result[0], column_names)
                for column_name, relevant in decisions.items():
                    chosen[(table_name, column_name)] = relevant
                missing = [c for c in column_names if c not in decisions]
                if missing:
                    still_pending.append((table_name, missing))
            pending = still_pending

        for table_name, column_names in pending:
            Logger().log(f"({state.task.db_id}, {state.task.question_id}) No column filtering answer for {table_name}: {column_names}", "error")
            logging.error(f"Error in column filtering for table '{table_name}', columns {column_names}: no answer")

        tentative_schema = state.tentative_schema
        for table_name, columns in column_profiles.items():
            tentative_schema[table_name] = [column_name for column_name in columns if chosen.get((table_name, column_name))]

        state.add_columns_to_tentative_schema(state.similar_columns)
        state.add_connections_to_tentative_schema()


    def _get_updates(self, state: SystemState) -> Dict:
        return {"tentative_schema": state.tentative_schema, "num_calls": self.num_calls}
