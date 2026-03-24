# quantintel/llm_clients/anthropic_client.py
from typing import Any, Optional
from langchain_anthropic import ChatAnthropic
from .base_client import BaseLLMClient, normalize_content

_PASSTHROUGH = ("timeout", "max_retries", "api_key", "max_tokens",
                "callbacks", "http_client", "http_async_client", "effort")


class NormalizedChatAnthropic(ChatAnthropic):
    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        llm_kwargs = {"model": self.model}
        for key in _PASSTHROUGH:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        return True
