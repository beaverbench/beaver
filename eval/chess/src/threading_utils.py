import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor

# CHESS opened one thread per call (e.g. one per column in filter_column); cap it so that concurrent LLM
# requests stay at about workers x MAX_CONCURRENT_CALLS
MAX_CONCURRENT_CALLS = 16


def _threaded(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception in thread with kwargs: {kwargs}\n{e}")
            return None
    return wrapper


def ordered_concurrent_function_calls(call_list: list) -> list:
    """
    Executes multiple functions concurrently using a thread pool, and returns the results in the order of the input list.
    Each call runs in a copy of the caller's context, so DatabaseManager() and Logger() resolve to the caller's question.
    """
    if not call_list:
        return []
    with ThreadPoolExecutor(max_workers=min(len(call_list), MAX_CONCURRENT_CALLS)) as executor:
        futures = [
            executor.submit(contextvars.copy_context().run, _threaded(call['function']), **call['kwargs'])
            for call in call_list
        ]
        return [future.result() for future in futures]
