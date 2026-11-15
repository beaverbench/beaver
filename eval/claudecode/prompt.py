import math
from typing import Any, Dict, List, Optional

# Prompt for the Claude Code agent. SYSTEM_APPEND is added to Claude Code's own system prompt; the user prompt
# holds the schema of the question's tables (the retrieved top-k in setting 0, the gold tables in settings 1/2)
# and the shared hint text. Join keys go in the schema as foreign keys (as for XiYan-SQL and CHESS), so the
# hint text holds only mapping, domain knowledge and decomposition. The schema and hints are a starting point:
# the agent can query any table of the database.

SYSTEM_APPEND = """You are answering a question about a MySQL database by writing one SQL query.
Use the execute_sql tool to explore the database (tables, columns and values) and to test candidate queries. \
When you are confident in your answer, call submit_sql once with the final query; this ends the task. The final \
query must be a single MySQL SELECT (or WITH) statement that answers the question."""

PROMPT_TEMPLATE = """Database: {db}

Schema of the tables relevant to the question:
{schema}
{evidence}
Question: {question}"""

MAX_EXAMPLES = 3
MAX_EXAMPLE_CHARS = 50


def _examples(values: List[Any]) -> List[str]:
    # example_columns can repeat values and uses NaN for NULL; MySQL SET values arrive as lists
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        v = ",".join(map(str, v)) if isinstance(v, list) else str(v)
        if v and len(v) <= MAX_EXAMPLE_CHARS and v not in out:
            out.append(v)
    return out[:MAX_EXAMPLES]


def get_schema(tables: List[str], corpus_tables: Dict[str, Dict[str, Any]],
               join_keys: Optional[List[List[str]]] = None) -> str:
    """CREATE TABLE statements from dev_tables.json with example values as comments; join keys between shown
    tables become FOREIGN KEY lines of the left table."""
    names = {t.lower(): t for t in corpus_tables}
    shown = list(dict.fromkeys(names[t.lower()] for t in tables if t.lower() in names))
    foreign_keys: Dict[str, List[str]] = {}
    for left, right in join_keys or []:
        (lt, lc), (rt, rc) = left.split(".", 1), right.split(".", 1)
        if lt.lower() in names and rt.lower() in names and names[lt.lower()] in shown and names[rt.lower()] in shown:
            foreign_keys.setdefault(names[lt.lower()], []).append(f"FOREIGN KEY ({lc}) REFERENCES {names[rt.lower()]}({rc})")

    ddl = []
    for name in shown:
        t = corpus_tables[name]
        # (definition, comment) pairs; commas go before the comments
        defs = [(f"{column} {column_type}", ", ".join(_examples(examples)))
                for column, column_type, examples in zip(t["column_names"], t["column_types"], t["example_columns"])]
        defs += [(fk, "") for fk in foreign_keys.get(name, [])]
        body = [f"  {d}{',' if i < len(defs) - 1 else ''}" + (f" -- examples: {c}" if c else "")
                for i, (d, c) in enumerate(defs)]
        ddl.append(f"CREATE TABLE {name} (\n" + "\n".join(body) + "\n);")
    return "\n\n".join(ddl)


def get_prompt(q: Dict[str, Any], schema: str, evidence: str) -> str:
    return PROMPT_TEMPLATE.format(db=q["db"], schema=schema, question=q["question"],
                                  evidence=f"\nHints:\n{evidence}\n" if evidence else "")
