# quantintel/llm_clients/base_client.py
import json
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict


def extract_raw_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Extract and recover tool calls when open-source LLMs (e.g. Llama-3.1-8B)
    output JSON in text content instead of structured tool_calls.
    """
    if not text or not isinstance(text, str):
        return []

    trimmed = text.strip()
    
    # Strip markdown json code fences if present
    if trimmed.startswith("```json"):
        trimmed = trimmed[7:]
    elif trimmed.startswith("```"):
        trimmed = trimmed[3:]
    if trimmed.endswith("```"):
        trimmed = trimmed[:-3]
    trimmed = trimmed.strip()

    # Try full JSON parse
    candidates = []
    if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
        try:
            parsed = json.loads(trimmed)
            candidates = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            candidates = []

    # If full parse failed, search for {"name": ...} blocks using regex
    if not candidates:
        pattern = re.compile(r'\{[^{}]*?"name"\s*:\s*"[^"]+?"[^{}]*?\}', re.DOTALL)
        matches = pattern.findall(trimmed)
        for m in matches:
            try:
                candidates.append(json.loads(m))
            except Exception:
                pass

    recovered = []
    for item in candidates:
        if isinstance(item, dict) and "name" in item:
            name = item["name"]
            raw_args = item.get("parameters") or item.get("args") or item.get("arguments") or {}
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except Exception:
                    pass
            recovered.append({
                "name": str(name),
                "args": raw_args if isinstance(raw_args, dict) else {},
                "id": f"call_{uuid.uuid4().hex[:9]}",
                "type": "tool_call",
            })

    return recovered


def normalize_content(response):
    """Normalize LLM response content to a plain string and recover raw text tool calls."""
    content = getattr(response, "content", "")
    if isinstance(content, list):
        texts = [
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in content
        ]
        response.content = "\n".join(t for t in texts if t)

    # AUTO-HEAL: If native tool_calls is empty, inspect content for text-formatted tool calls
    if not getattr(response, "tool_calls", None) and isinstance(response.content, str):
        raw_text = response.content.strip()
        if '"name"' in raw_text and ('"parameters"' in raw_text or '"args"' in raw_text or '"arguments"' in raw_text):
            recovered = extract_raw_tool_calls(raw_text)
            if recovered:
                response.tool_calls = recovered
                response.content = ""

    return response



class BaseLLMClient(ABC):
    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        self.model    = model
        self.base_url = base_url
        self.kwargs   = kwargs

    @abstractmethod
    def get_llm(self) -> Any:
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        pass
