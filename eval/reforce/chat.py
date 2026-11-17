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


from openai import OpenAI, AzureOpenAI
import os

# eval/utils is appended, not prepended, so this folder's utils.py still wins the `utils` name.
# llm_client paces every chat completion by LLM_RPM (shared by all vote threads), waits out rate limits,
# and records each call's token usage for the current question (cost.py).
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
import cost
import llm_client

class GPTChat(BaseChat):
    def __init__(self, azure=False, model="gpt-4o", temperature=1.0):
        super().__init__(model, temperature)
        # Step label of this session's calls in cost.json (run.py names the column exploration, format and vote sessions)
        self.cost_step = "generation"

        # Generic OpenAI-compatible endpoint (LLM_BASE_URL + LLM_API_KEY) takes precedence over model-name routing
        base_url = os.environ.get("LLM_BASE_URL")
        self.custom_endpoint = bool(base_url) and not azure

        if self.custom_endpoint:
            # Retries are handled by llm_client; the SDK's quick retries would keep a rate-limit block alive
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
        self.client = llm_client.LLMClient(self.client, step=lambda: self.cost_step)
    
    def close(self):
        """Release the underlying httpx connection pool to avoid leaking sockets/file descriptors."""
        client = getattr(self, "client", None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def _create_with_backoff(self):
        """Paced chat completion that waits out rate limits (429) instead of failing the vote (see llm_client)."""
        # Plain chat completions; skip provider-specific params keyed on model name
        return self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature
        )

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
            self._record_usage(response)  # the Responses API bypasses LLMClient's chat recording
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

    def _record_usage(self, response):
        """Adds a Responses API call's token usage to the current question's cost recorder (eval/utils/cost.py)."""
        cost.record(cost.usage_record(self.cost_step, self.model, "chat", getattr(response, "usage", None), None,
                                      prompt_text="\n".join(m["content"] for m in self.messages),
                                      completion_text=getattr(response, "output_text", None)))