from typing import List, Literal
from multimodal_rag.log_config import logger

try:
    import tiktoken
except ImportError:
    tiktoken = None
    logger.info("tiktoken is not installed. Token counting will be skipped.")

ChatMessage = dict[Literal["role", "content"], str]

MODEL_CONTEXT_LIMITS = {
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
}


def get_tokenizer(model: str):
    """
    Return the tiktoken tokenizer associated with the given model.
    Defaults to 'cl100k_base' if model is unknown.
    """

    if not tiktoken:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_chat_tokens(messages: List[ChatMessage], model) -> int:
    """
    Count how many tokens are used by a list of chat messages.
    Includes OpenAI-specific formatting tokens for role/content structure.
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """

    encoding = get_tokenizer(model)
    if encoding is None:
        return 0

    if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
        num_tokens = 0
        for message in messages:
            num_tokens += 3  # message start overhead
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
        num_tokens += 3  # reply start overhead
        return num_tokens
    else:
        joined = " ".join(m["content"] for m in messages)
        return len(encoding.encode(joined))


def validate_token_limit(payload: dict, model: str, context_limit_override: int | None = None) -> None:
    """
    Validates that the sum of prompt tokens and max_tokens does not exceed
    the context window size for the selected model.
    """

    if not tiktoken:
        logger.info("Skipping token limit validation because tiktoken is not installed.")
        return

    max_total_tokens = context_limit_override or MODEL_CONTEXT_LIMITS.get(model)
    if max_total_tokens is None:
        raise ValueError(f"Unknown context length for model: {model}")

    messages = payload["messages"]
    max_tokens = payload.get("max_tokens", 0)
    prompt_tokens = count_chat_tokens(messages, model)
    total = prompt_tokens + max_tokens

    if total > max_total_tokens:
        raise ValueError(f"Token limit exceeded: {prompt_tokens} + {max_tokens} = {total} > {max_total_tokens}")
