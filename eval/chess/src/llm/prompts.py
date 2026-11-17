import logging
from pathlib import Path

TEMPLATES_ROOT_PATH = Path(__file__).resolve().parents[2] / "templates"


def get_prompt(template_name: str) -> str:
    """
    Loads a template; placeholders such as {QUESTION} are filled with str.format, which (like langchain's
    f-string templates that CHESS used) reads {{ and }} as literal braces.
    """
    template_path = TEMPLATES_ROOT_PATH / f"template_{template_name}.txt"
    try:
        return template_path.read_text()
    except FileNotFoundError:
        logging.error(f"Template file not found: {template_path}")
        raise
