import json.decoder
from utils.enums import LLM
import os
import sys

# eval/utils is appended, not prepended, so this folder's utils package still wins the `utils` name
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"))
import cost
import llm_client

CLIENT = None


def init_chatgpt(OPENAI_API_KEY, OPENAI_GROUP_ID, model):
    global CLIENT
    if OPENAI_API_KEY:  # --openai_api_key, used for OpenAI models
        os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
    # LLM_BASE_URL, else Gemini / OpenRouter ("/" in the name) / OpenAI; requests are paced and their usage recorded
    CLIENT = llm_client.LLMClient(llm_client.make_client(model), step="generate")

def ask_chat(model, messages: list, temperature, n, max_tokens):
    global CLIENT
    # print('>>>model:', model)
    # print('>>>max_tokens:', max_tokens)
    
    # Ensure client is initialized
    if CLIENT is None:
        raise RuntimeError("OpenAI Client not initialized. Call init_chatgpt first.")

    if "gemini" in model:
         # Gemini logic
        try:
            if "2" in model:
                response = CLIENT.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    # reasoning_effort="none"
                )
            else:
             response = CLIENT.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                reasoning_effort="none"
            )
        except Exception as e:
            # Fallback if reasoning_effort is not supported or other error, retry without it
            print(f"Gemini call failed with reasoning_effort='none', retrying without it. Error: {e}")
            response = CLIENT.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n
            )
    elif "/" in model:
        if "qwen" in model or "claude" in model:
            response = CLIENT.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                # OpenRouter-only parameter; a custom endpoint (LLM_BASE_URL) gets none
                **({} if llm_client.custom_endpoint() else {"extra_body": {"reasoning": {"effort": "none"}}}),
        )
        elif "minimax" in model:
            response = CLIENT.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                # max_tokens=max_tokens,
                n=n
        )

    elif "gpt-5" in model or "o4-mini" in model:
        response = CLIENT.chat.completions.create(
            model=model,
            messages=messages,
            # temperature=temperature,
            # max_completion_tokens=max_tokens,
            n=n
        )
    elif model == LLM.GPT_o1:
        response = CLIENT.chat.completions.create(
            model=model,
            messages=messages,
            # temperature=temperature, # o1 doesn't support temp
            # max_completion_tokens=max_tokens,
            n=n
        )
    else:
        response = CLIENT.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n
        )
    
    response_clean = [choice.message.content for choice in response.choices]
    return dict(
        response=response_clean,
        # usage might be different object in v1, access safely
        total_tokens=response.usage.total_tokens,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens
    )


def ask_llm(model: str, batch: list, temperature: float, n:int, max_tokens):
    try:
        # assert model in LLM.TASK_CHAT, "LLM.TASK_COMPLETIONS is too old. not supported."
        # batch size must be 1
        assert len(batch) == 1, "batch must be 1 in this mode"
        messages = [{"role": "user", "content": batch[0]}]
        response = ask_chat(model, messages, temperature, n, max_tokens)
        response['response'] = [response['response']]  # hard-code for batch_size=1
    except Exception as e:
        print(f"Error occurred in ask_llm: {e}")
        # import traceback
        # traceback.print_exc()
        # Return hard-coded response
        print("Return hard-coded response")
        response = {"total_tokens": 0, "response": [["SELECT" for _ in range(n)]]}
    
    return response

