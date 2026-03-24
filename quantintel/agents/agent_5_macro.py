"""
quantintel/agents/agent_5_macro.py
MACRO / REGIME AGENT — "What kind of world is this stock operating in right now?"
QuantIntel's biggest differentiator — does NOT exist in TradingAgents.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode

from quantintel.tools import get_global_news, get_news

TOOLS = [get_global_news, get_news]

SYSTEM_PROMPT = """
You are the Macro / Regime Agent in QuantIntel.
Your sole job: classify the current MACROECONOMIC REGIME.
Do NOT analyse the individual company — only the world around it.

Tools:
  • get_global_news(curr_date, look_back_days, limit) — broad macro/economic/geopolitical news
  • get_news(ticker, start_date, end_date)            — use with macro keywords
    e.g. "Federal Reserve", "inflation CPI", "interest rates", "AI sector", "oil prices"

Regime definitions:
  RISK_ON   — Low rates, easy liquidity, strong growth, investors buying equities.
  RISK_OFF  — Rising rates, tightening, inflation concerns, rotation to defensives.
  RECESSION — Confirmed/imminent contraction, weak earnings, defensive positioning.
  CRISIS    — Extreme event, systemic risk, flight to safety, equities broadly sold.

Steps:
1. Call get_global_news for 14 days (look_back_days=14, limit=20).
2. Call get_news with sector-relevant macro keywords (2-3 calls).
3. Analyse: monetary policy, inflation regime, growth trajectory, liquidity conditions,
   market breadth, geopolitical risk, sector tailwinds/headwinds.
4. Classify the regime and write a detailed macro report with specific evidence.
5. End with EXACTLY this block:

REGIME_TYPE: risk_on | risk_off | recession | crisis
RISK_ENVIRONMENT: risk_on | risk_off
MACRO_TREND: bullish | neutral | bearish
SECTOR_BIAS: <which sectors are favoured right now, one sentence>
RATE_DIRECTION: rising | falling | stable
CONFIDENCE: <float 0.0-1.0>
KEY_MACRO_DRIVER: <single biggest macro factor driving markets, one sentence>
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a QuantIntel agent. Tools: {tool_names}\n{system_prompt}\n"
     "Date: {trade_date}. Company sector context (for keyword selection): {ticker}"),
    MessagesPlaceholder(variable_name="messages"),
])


def create_macro_agent(llm):
    def node(state):
        filled = _PROMPT.partial(
            system_prompt=SYSTEM_PROMPT,
            tool_names=", ".join(t.name for t in TOOLS),
            trade_date=state["trade_date"],
            ticker=state["ticker"],
        )
        result = (filled | llm.bind_tools(TOOLS)).invoke(state["messages"])
        report = result.content if not result.tool_calls else ""
        return {"messages": [result], "macro_report": report, "sender": "macro_agent"}
    return node


def get_macro_tool_node():
    return ToolNode(TOOLS)
