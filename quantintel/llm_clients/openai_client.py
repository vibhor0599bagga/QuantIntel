# quantintel/llm_clients/openai_client.py
import os
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from .base_client import BaseLLMClient, normalize_content

_PASSTHROUGH = ("timeout", "max_retries", "reasoning_effort",
                "api_key", "callbacks", "http_client", "http_async_client")

_PROVIDER_CONFIG = {
    "xai":        ("https://api.x.ai/v1",             "XAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1",     "OPENROUTER_API_KEY"),
    "ollama":     ("http://localhost:11434/v1",         None),
}


class NormalizedChatOpenAI(ChatOpenAI):
    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str, base_url: Optional[str] = None,
                 provider: str = "openai", **kwargs):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        llm_kwargs = {"model": self.model}

        if self.provider in _PROVIDER_CONFIG:
            base_url, api_key_env = _PROVIDER_CONFIG[self.provider]
            llm_kwargs["base_url"] = base_url
            if api_key_env:
                api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY") or "placeholder_key"
                llm_kwargs["api_key"] = api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url
            if "api_key" not in llm_kwargs:
                llm_kwargs["api_key"] = os.environ.get("OPENAI_API_KEY") or "placeholder_key"
        else:
            if "api_key" not in llm_kwargs and not os.environ.get("OPENAI_API_KEY"):
                llm_kwargs["api_key"] = "placeholder_key"


        for key in _PASSTHROUGH:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True

        if self.provider == "openrouter" and "max_tokens" not in llm_kwargs:
            llm_kwargs["max_tokens"] = 6000

        return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        return True
