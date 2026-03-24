# quantintel/llm_clients/google_client.py
from typing import Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from .base_client import BaseLLMClient, normalize_content


class NormalizedChatGoogle(ChatGoogleGenerativeAI):
    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class GoogleClient(BaseLLMClient):
    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        llm_kwargs = {"model": self.model}
        for key in ("timeout", "max_retries", "google_api_key", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            if "gemini-3" in self.model.lower():
                if "pro" in self.model.lower() and thinking_level == "minimal":
                    thinking_level = "low"
                llm_kwargs["thinking_level"] = thinking_level
            else:
                llm_kwargs["thinking_budget"] = -1 if thinking_level == "high" else 0

        return NormalizedChatGoogle(**llm_kwargs)

    def validate_model(self) -> bool:
        return True
