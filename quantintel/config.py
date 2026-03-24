"""
quantintel/config.py
All configuration in one place. No external dependencies.
"""

import os

# ── default config ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "project_dir":    os.path.abspath(os.path.dirname(__file__)),
    "results_dir":    os.getenv("QUANTINTEL_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(os.path.abspath(os.path.dirname(__file__)), "dataflows/data_cache"),

    # LLM settings  —  change these to match your provider
    "llm_provider":    "openrouter",           # anthropic | openai | google | ollama | openrouter
    "deep_think_llm":  "gpt-4o-mini",          # used by Supervisor Agent (update to OpenRouter-served model)
    "quick_think_llm": "gpt-4o-mini",          # used by all other agents
    "backend_url":     None,

    # Provider-specific thinking config (leave None unless you need it)
    "anthropic_effort":       None,   # "high" | "medium" | "low"
    "openai_reasoning_effort": None,  # "high" | "medium" | "low"
    "google_thinking_level":  None,   # "high" | "minimal" etc.

    # Graph settings
    "max_recur_limit": 100,

    # Data vendors  —  yfinance is free, no API key needed
    "data_vendors": {
        "core_stock_apis":      "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data":     "yfinance",
        "news_data":            "yfinance",
    },
}

# ── runtime config store ───────────────────────────────────────────────────────
_config = None


def get_config() -> dict:
    global _config
    if _config is None:
        _config = DEFAULT_CONFIG.copy()
    return _config.copy()


def set_config(cfg: dict):
    global _config
    if _config is None:
        _config = DEFAULT_CONFIG.copy()
    _config.update(cfg)
