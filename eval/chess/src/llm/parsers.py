import json
import logging
import re
from ast import literal_eval
from typing import Any, Callable, Dict

# CHESS llm/parsers.py for the parsers used by the IR+SS+CG pipeline, without langchain. JSON outputs are
# read like langchain's JsonOutputParser: the ```json block if there is one, else the outermost {...}.


class OutputParserException(ValueError):
    pass


def _parse_json_markdown(output: str) -> Any:
    match = re.search(r"```(?:json)?\s*(.*?)```", output, re.DOTALL)
    text = match.group(1) if match else output
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise OutputParserException(f"No JSON object in output: {output[:200]}")
    try:
        return json.loads(text[start:end + 1], strict=False)
    except json.JSONDecodeError as e:
        raise OutputParserException(f"Invalid JSON: {e}")


def json_output_parser(output: str) -> Dict[str, Any]:
    parsed = _parse_json_markdown(output)
    if not isinstance(parsed, dict):
        raise OutputParserException("Expected a JSON object")
    return parsed


def python_list_output_parser(output: str) -> Any:
    """Parses output embedded in markdown code blocks containing Python lists."""
    if "```python" in output:
        output = output.split("```python")[1].split("```")[0]
    output = re.sub(r"^\s+", "", output)
    try:
        # CHESS used eval() here; literal_eval accepts the same list literals without running code
        parsed = literal_eval(output.strip())
    except Exception as e:
        raise OutputParserException(f"Error parsing list: {e}")
    if not isinstance(parsed, list):
        raise OutputParserException("Expected a Python list")
    return parsed


def generate_candidate_gemini_markdown_cot(output: str) -> Dict[str, str]:
    """Parses the SQL inside <FINAL_ANSWER> tags, keeping the reasoning before it as the plan."""
    logging.debug(f"Parsing output with RecapOutputParserCOT: {output}")
    plan = ""
    if "<FINAL_ANSWER>" in output and "</FINAL_ANSWER>" in output:
        plan = output.split("<FINAL_ANSWER>")[0]
        output = output.split("<FINAL_ANSWER>")[1].split("</FINAL_ANSWER>")[0]
    query = output.replace("```sql", "").replace("```", "").replace("\n", " ")
    return {"SQL": query, "plan": plan}


def revise_new(output: str) -> Dict[str, str]:
    """Parses the corrected SQL inside <FINAL_ANSWER> tags."""
    logging.debug(f"Parsing output with CheckerOutputParser: {output}")
    if "<FINAL_ANSWER>" in output and "</FINAL_ANSWER>" in output:
        output = output.split("<FINAL_ANSWER>")[1].split("</FINAL_ANSWER>")[0]
    if "<FINAL_ANSWER>" in output:
        output = output.split("<FINAL_ANSWER>")[1]
    query = output.replace("```sql", "").replace("```", "").replace("\n", " ")
    return {"refined_sql_query": query}


def get_parser(parser_name: str) -> Callable[[str], Any]:
    parser_configs = {
        "python_list_output_parser": python_list_output_parser,
        "filter_column": json_output_parser,
        "select_tables": json_output_parser,
        "select_columns": json_output_parser,
        "generate_candidate_gemini_markdown_cot": generate_candidate_gemini_markdown_cot,
        "revise_new": revise_new,
    }
    if parser_name not in parser_configs:
        logging.error(f"Invalid parser name: {parser_name}")
        raise ValueError(f"Invalid parser name: {parser_name}")
    return parser_configs[parser_name]
