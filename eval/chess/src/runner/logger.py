import contextvars
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from runner.task import Task


class Logger:
    """
    Per-question logger (CHESS used a process-wide singleton). Logger() returns the current question's
    instance; like DatabaseManager it is context-local. Writes <instance_dir>/generation.log (LLM inputs and
    outputs) and <instance_dir>/history.json (CHESS's execution history).
    """
    _current: contextvars.ContextVar = contextvars.ContextVar("logger")

    def __new__(cls):
        try:
            return cls._current.get()
        except LookupError:
            raise ValueError("Logger instance has not been initialized.")

    @classmethod
    def open(cls, question_id: str, instance_dir: str) -> "Logger":
        self = object.__new__(cls)
        self.question_id = question_id
        self.instance_dir = Path(instance_dir)
        self.instance_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_lock = Lock()
        cls._current.set(self)
        return self

    def log(self, message: str, log_level: str = "info", task: Task = None):
        log_method = getattr(logging, log_level, None)
        if log_method is None:
            raise ValueError(f"Invalid log level: {log_level}")
        if task is not None:
            log_method(f"({task.db_id}, {task.question_id}) {message}")
        else:
            log_method(f"({self.question_id}) {message}")

    def log_conversation(self, conversations: List[Dict[str, Any]]):
        with self.log_file_lock:
            with (self.instance_dir / "generation.log").open("a") as file:
                for conversation in conversations:
                    text = conversation["text"]
                    file.write(f"############################## {conversation['from']} at step {conversation['step']} ##############################\n\n")
                    if isinstance(text, str):
                        file.write(text)
                    else:
                        file.write(json.dumps(text, indent=4, default=str))
                    file.write("\n\n")

    def dump_history_to_file(self, execution_history: List[Dict[str, Any]]):
        with (self.instance_dir / "history.json").open("w") as file:
            json.dump(execution_history, file, indent=4, default=str)
