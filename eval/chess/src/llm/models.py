from typing import Any, Callable, Dict, List, Optional

from llm.engine import chat
from llm.parsers import OutputParserException
from runner.logger import Logger
from threading_utils import ordered_concurrent_function_calls


def get_llm_chain(engine_name: str, temperature: float = 0) -> Dict[str, Any]:
    """Returns the engine settings for chat(); there is no langchain chain to build."""
    return {"engine_name": engine_name, "temperature": temperature}


def call_llm_chain(prompt: str, engine: Dict[str, Any], parser: Callable[[str], Any], request_kwargs: Dict[str, Any],
                   step: str, max_attempts: int = 3, response_format: Optional[Dict[str, Any]] = None) -> Any:
    """
    Fills the prompt, calls the LLM and parses the output. An unparsable output is regenerated
    (CHESS asked langchain's OutputFixingParser to repair it instead). response_format optionally
    requests structured output where the endpoint supports it.
    """
    logger = Logger()
    prompt_text = prompt.format(**request_kwargs)
    for attempt in range(max_attempts):
        try:
            output = chat(engine["engine_name"], prompt_text, engine["temperature"], response_format, step=step)
        except Exception as e:
            logger.log(f"Failed to invoke the chain {attempt + 1} times.\n{type(e)} <{e}>\n", "error")
            raise
        # The template is fixed, so only the values filled into it are logged
        logger.log_conversation([
            {"text": request_kwargs, "from": "Human", "step": step},
            {"text": output, "from": "AI", "step": step},
        ])
        try:
            if output.strip() == "":
                raise OutputParserException("Empty output")
            return parser(output)
        except Exception as e:
            logger.log(f"OutputParserException at {step} (attempt {attempt + 1}/{max_attempts}): {e}", "warning")
            if attempt == max_attempts - 1:
                raise OutputParserException(str(e))


def async_llm_chain_call(
    prompt: str,
    engine: Dict[str, Any],
    parser: Callable[[str], Any],
    request_list: List[Dict[str, Any]],
    step: str,
    sampling_count: int = 1,
    response_formats: Optional[List[Optional[Dict[str, Any]]]] = None
) -> List[List[Any]]:
    """
    Calls the LLM chain for every request (sampling_count times each) on concurrent threads.
    response_formats, if given, holds one optional structured-output schema per request.
    Returns one list of results per request; a failed call yields None.
    """
    call_list = []
    for request_id, request_kwargs in enumerate(request_list):
        for _ in range(sampling_count):
            call_list.append({
                'function': call_llm_chain,
                'kwargs': {
                    'prompt': prompt,
                    'engine': engine,
                    'parser': parser,
                    'request_kwargs': request_kwargs,
                    'step': step,
                    'response_format': response_formats[request_id] if response_formats else None,
                }
            })

    results = ordered_concurrent_function_calls(call_list)

    return [
        results[i * sampling_count: (i + 1) * sampling_count]
        for i in range(len(request_list))
    ]
