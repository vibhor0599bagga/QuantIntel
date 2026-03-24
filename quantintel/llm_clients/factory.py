# quantintel/llm_clients/factory.py
from typing import Optional
from .base_client import BaseLLMClient
from .anthropic_client import AnthropicClient
from .openai_client import OpenAIClient
from .google_client import GoogleClient


def create_llm_client(provider: str, model: str,
                      base_url: Optional[str] = None, **kwargs) -> BaseLLMClient:
    p = provider.lower()
    if p in ("openai", "ollama", "openrouter", "xai"):
        return OpenAIClient(model, base_url, provider=p, **kwargs)
    if p == "anthropic":
        return AnthropicClient(model, base_url, **kwargs)
    if p == "google":
        return GoogleClient(model, base_url, **kwargs)
    raise ValueError(f"Unsupported provider: {provider}")
