"""
main.py  —  QuantIntel entry point
Run: python main.py
"""

import os
from dotenv import load_dotenv
load_dotenv()  # loads .env file automatically

from quantintel.graph import QuantIntelGraph

# ── config overrides (edit here) ──────────────────────────────────────────────
config = {
    "llm_provider":    "openrouter",              # anthropic | openai | google | ollama | openrouter
    "deep_think_llm":  "gpt-4o-mini",             # Supervisor Agent (use an OpenRouter-served model)
    "quick_think_llm": "gpt-4o-mini",             # All other agents
}
# config = {
#     "llm_provider":    "google",
#     "deep_think_llm":  "gemini-2.5-flash",      # or gemini-1.0-pro, or gemini-2.5-flash if your key supports it
#     "quick_think_llm": "gemini-2.5-flash",
# }

# ── optional portfolio context ─────────────────────────────────────────────────
portfolio_context = {
    "sector_exposure": "tech_heavy",        # tech_heavy | banking_focused | diversified
    "horizon":         "medium_term",         # short_term | medium_term | long_term
    "risk_tolerance":  "conservative",          # conservative | moderate | aggressive
    "holdings":        ["MSFT", "GOOGL", "PLTR"],   # existing holdings for concentration check
}

# ── run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("  QuantIntel — Multi-Agent Investment Analysis")
    print("=" * 62)

    qi = QuantIntelGraph(config=config, debug=False)

    result = qi.run(
        ticker            = "INCY",          # ← change ticker here
        trade_date        = "2026-03-23",    # ← change date here
        portfolio_context = portfolio_context,
    )

    # result = qi.run(
    #     ticker            = "PLTR",          # ← change ticker here
    #     trade_date        = "2026-03-23",    # ← change date here
    #     portfolio_context = portfolio_context,
    # )


    # Print individual agent reports
    sections = [
        ("AGENT 1 — FUNDAMENTALS", "fundamentals_report"),
        ("AGENT 2 — SENTIMENT",    "sentiment_report"),
        ("AGENT 3 — TECHNICAL",    "technical_report"),
        ("AGENT 4 — RISK",         "risk_report"),
        ("AGENT 5 — MACRO/REGIME", "macro_report"),
    ]

    for title, key in sections:
        print(f"\n{'─' * 62}")
        print(f"  {title}")
        print(f"{'─' * 62}")
        print(result.get(key, "Report not generated."))

    print(f"\n{'=' * 62}")
    print("  AGENT 6 — SUPERVISOR — FINAL RECOMMENDATION")
    print(f"{'=' * 62}")
    print(result.get("final_recommendation", "No recommendation generated."))
