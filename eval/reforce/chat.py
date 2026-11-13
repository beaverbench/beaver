import sys
from abc import ABC, abstractmethod
from utils import extract_all_blocks
import time

class BaseChat(ABC):
    def __init__(self, model: str, temperature: float = 1.0):
        self.model = model
        self.temperature = float(temperature)
        self.messages = []

    @abstractmethod
    def get_response(self, prompt) -> str:
        pass

    def get_model_response(self, prompt, code_format=None) -> list:
        code_blocks = []
        max_try = 3
        last_response = None
        while code_blocks == [] and max_try > 0:
            max_try -= 1
            try:
                response = self.get_response(prompt)
            except Exception as e:
                print(f"max_try: {max_try}, exception: {e}")
                continue
            last_response = response
            code_blocks = extract_all_blocks(response, code_format)
        if max_try == 0 or code_blocks == []:
            print(f"get_model_response() exit, max_try: {max_try}, code_blocks: {code_blocks}")
            # Diagnostic: show what made the vote give up (prompt tail holds the instruction; reply is truncated)
            print(f"[Exit diagnostic] model={self.model} format={code_format} "
                  f"prompt_tail={prompt[-250:]!r} last_reply={(last_response or '')[:400]!r}")
            sys.exit(0)
        return code_blocks

    def get_model_response_txt(self, prompt) -> str:
        max_try = 3
        while max_try > 0:
            max_try -= 1
            try:
                response = self.get_response(prompt)
                return response
            except Exception as e:
                print(f"max_try: {max_try}, exception: {e}")
                continue
        print(f"get_model_response_txt() exit, max_try: {max_try}")
        sys.exit(0)

    def get_message_len(self):
        return {
            "prompt_len": sum(len(item["content"]) for item in self.messages if item["role"] == "user"),
            "response_len": sum(len(item["content"]) for item in self.messages if item["role"] == "assistant"),
            "num_calls": len(self.messages) // 2
        }

    def init_messages(self):
        self.messages = []


from openai import OpenAI, AzureOpenAI, RateLimitError, APIConnectionError, InternalServerError
import os
import random
import threading

# Process-wide request pacing for the custom endpoint, shared by all vote threads.
# LLM_RPM caps total requests per minute (unset/0 = no cap); a 429 pauses every thread, not just the one that hit it.
_request_lock = threading.Lock()
_next_request_time = 0.0

def _acquire_request_slot():
    global _next_request_time
    rpm = float(os.environ.get("LLM_RPM") or 0)
    interval = 60.0 / rpm if rpm > 0 else 0.0
    with _request_lock:
        now = time.monotonic()
        start = max(now, _next_request_time)
        _next_request_time = start + interval
    if start > now:
        time.sleep(start - now)

def _pause_all_requests(seconds):
    global _next_request_time
    with _request_lock:
        _next_request_time = max(_next_request_time, time.monotonic() + seconds)

class GPTChat(BaseChat):
    def __init__(self, azure=False, model="gpt-4o", temperature=1.0):
        super().__init__(model, temperature)

        # Generic OpenAI-compatible endpoint (LLM_BASE_URL + LLM_API_KEY) takes precedence over model-name routing
        base_url = os.environ.get("LLM_BASE_URL")
        self.custom_endpoint = bool(base_url) and not azure

        if self.custom_endpoint:
            # Retries are handled in _create_with_backoff; the SDK's quick retries would keep a rate-limit block alive
            self.client = OpenAI(
                base_url=base_url,
                api_key=os.environ.get("LLM_API_KEY"),
                max_retries=0,
            )
        elif not azure:
            if model in ["o1-preview", "o1-mini"]:
                self.client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    api_version="2024-12-01-preview"
                )
            elif model in ["deepseek-reasoner"]:
                self.client = OpenAI(
                    base_url="https://api.deepseek.com",
                    api_key=os.environ.get("DS_API_KEY"),
                )
            elif "gemini" in model:
                self.client = OpenAI(
                    api_key=os.environ.get("GEMINI_API_KEY"),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
            elif "/" in model:
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                )
            else:
                self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        else:
            if model in ["o1-preview", "o1-mini", "o3", "o4-mini"]:
                version = "2024-12-01-preview"
            elif model in ["o3-pro"]:
                version = "2025-03-01-preview"
            else:
                version = "2024-05-01-preview"

            self.client = AzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_KEY"),
                api_version=version
            )
    
    def close(self):
        """Release the underlying httpx connection pool to avoid leaking sockets/file descriptors."""
        client = getattr(self, "client", None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def _create_with_backoff(self):
        """Paced chat completion that waits out rate limits (429) instead of failing the vote."""
        max_rate_limit_retries = int(os.environ.get("LLM_MAX_RATE_LIMIT_RETRIES", 20))
        rate_limit_retries = 0
        transient_retries = 0
        while True:
            _acquire_request_slot()
            try:
                # Plain chat completions; skip provider-specific params keyed on model name
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.temperature
                )
            except RateLimitError:
                if rate_limit_retries >= max_rate_limit_retries:
                    raise
                wait = min(60 * 2 ** rate_limit_retries, 300) + random.uniform(0, 10)
                rate_limit_retries += 1
                print(f"[Rate limited] model={self.model} pausing all requests {wait:.0f}s (retry {rate_limit_retries}/{max_rate_limit_retries})")
                _pause_all_requests(wait)
            except (APIConnectionError, InternalServerError):
                if transient_retries >= 3:
                    raise
                transient_retries += 1
                time.sleep(10 * transient_retries)

    def get_response(self, prompt) -> str:
        self.messages.append({"role": "user", "content": prompt})
        if self.custom_endpoint:
            response = self._create_with_backoff()
            choice = response.choices[0]
            main_content = choice.message.content
            if main_content is None or not main_content.strip():
                # Diagnose empty replies (None or blank text), then raise so get_model_response retries instead of crashing the thread
                usage = getattr(response, "usage", None)
                details = getattr(usage, "completion_tokens_details", None)
                reasoning = getattr(choice.message, "reasoning_content", None) or (choice.message.model_extra or {}).get("reasoning_content")
                print(
                    f"[Empty response] model={self.model} content={'None' if main_content is None else 'blank'} finish_reason={choice.finish_reason} "
                    f"completion_tokens={getattr(usage, 'completion_tokens', None)} "
                    f"reasoning_tokens={getattr(details, 'reasoning_tokens', None)} "
                    f"reasoning_content_len={len(reasoning) if reasoning else 0} "
                    f"refusal={choice.message.refusal!r}"
                )
                self.messages.pop()  # drop the user message so the retry doesn't duplicate it
                raise ValueError(f"Empty response content (finish_reason={choice.finish_reason})")
        elif self.model == "o3-pro":
            response = self.client.responses.create(
                model=self.model,
                input=self.messages,
                temperature=self.temperature
            )
            main_content = response.output_text
        elif 'gemini' in self.model:
            if '2' in self.model:
                time.sleep(1)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.temperature
                )
            else:
                response = self.client.chat.completions.create(
                model=self.model,
                reasoning_effort="none",
                messages=self.messages,
                temperature=self.temperature
            )
            main_content = response.choices[0].message.content
        elif 'claude' in self.model or 'qwen' in self.model:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                extra_body={
                    "reasoning": {
                        "effort": "none",
                    }
                },
            )
            main_content = response.choices[0].message.content
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature
            )
            main_content = response.choices[0].message.content

        self.messages.append({"role": "assistant", "content": main_content})
        return main_content