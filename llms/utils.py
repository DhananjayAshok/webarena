import os
from typing import Any

from llms import (
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)

APIInput = str | list[Any] | dict[str, Any]

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
) -> str:
    response: str
    assert isinstance(prompt, list)

    provider = lm_config.provider
    if provider == "openai":
        base_url = None
        api_key = os.environ.get("OPENAI_API_KEY")
    elif provider == "openrouter":
        base_url = _OPENROUTER_BASE_URL
        api_key = os.environ["OPENROUTER_API_KEY"]
    elif provider == "vllm":
        base_url = lm_config.gen_config["base_url"]
        api_key = "EMPTY"
    else:
        raise NotImplementedError(f"provider '{provider}' not supported in call_llm")

    response = generate_from_openai_chat_completion(
        messages=prompt,
        model=lm_config.model,
        temperature=lm_config.gen_config["temperature"],
        top_p=lm_config.gen_config["top_p"],
        context_length=lm_config.gen_config["context_length"],
        max_tokens=lm_config.gen_config["max_tokens"],
        stop_token=None,
        base_url=base_url,
        api_key=api_key,
    )
    return response
