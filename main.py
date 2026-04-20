"""
main.py  —  QuantIntel entry point
Run: python main.py
"""

import os
from dotenv import load_dotenv
load_dotenv()  # loads .env file automatically

from quantintel.mcp_graph import McpQuantIntelGraph
import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

# ── config overrides (edit here) ──────────────────────────────────────────────
config = {
    "llm_provider":    "openrouter",              # anthropic | openai | google | ollama | openrouter
    "deep_think_llm":  "gpt-4o-mini",             # Supervisor Agent (use an OpenRouter-served model)
    "quick_think_llm": "gpt-4o-mini",             # All other agents
}

# ── optional portfolio context ─────────────────────────────────────────────────
portfolio_context = {
    "sector_exposure": "tech_heavy",        # tech_heavy | banking_focused | diversified
    "horizon":         "medium_term",         # short_term | medium_term | long_term
    "risk_tolerance":  "conservative",          # conservative | moderate | aggressive
    "holdings":        ["MSFT", "GOOGL", "PLTR"],   # existing holdings for concentration check
}

import os

async def async_main():
    print("=" * 62)
    print("  QuantIntel — Multi-Agent MCP Swarm Analysis")
    print("=" * 62)
    
    python_cmd = os.path.abspath("venv/Scripts/python.exe")
    if not os.path.exists(python_cmd):
        python_cmd = sys.executable

    server_params = StdioServerParameters(
        command=python_cmd,
        args=["-m", "quantintel.mcp_servers.agent_swarm_server"],
        env={**os.environ, "PYTHONPATH": os.path.abspath(".")}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await load_mcp_tools(session)

            qi = McpQuantIntelGraph(session, mcp_tools, config=config, debug=True)

            result = await qi.run(
                ticker            = "BA",          # ← ticker here
                trade_date        = "2026-04-17",    # ← date here
                portfolio_context = portfolio_context,
            )
            print("\n\nFINAL MCP RESULT:\n" + result["final_recommendation"])

# ── run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(async_main())
