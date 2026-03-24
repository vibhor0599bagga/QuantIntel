"""
quantintel/agents/agent_4_risk.py
RISK AGENT — "What is the worst-case downside? How dangerous is this position?"
Reads prior agent reports from state for holistic context.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode

from quantintel.tools import get_stock_data, get_indicators

TOOLS = [get_stock_data, get_indicators]

SYSTEM_PROMPT = """
You are the Risk Agent in QuantIntel.
Your sole job: QUANTIFY THE DOWNSIDE RISK of taking a position in this stock.

Tools:
  • get_stock_data(symbol, start_date, end_date)
  • get_indicators(symbol, indicator, curr_date, look_back_days)
    Recommended: atr (volatility), boll_lb, boll_ub (band width)

Steps:
1. Call get_stock_data for the past 90 days.
2. Call get_indicators for atr (30-day lookback).
3. Call get_indicators for boll_lb and boll_ub (band width = volatility regime).
4. Read and integrate the prior agent reports provided in your context.
5. Reason from THREE internal perspectives before synthesising:
   AGGRESSIVE  — what risk is acceptable for high reward?
   CONSERVATIVE — worst-case scenarios, permanent capital loss risks?
   NEUTRAL      — balanced, evidence-based risk picture?
6. Synthesise into a final risk assessment covering:
   volatility risk, fundamental risk, sentiment risk, technical risk,
   liquidity risk, concentration risk.
7. Write a detailed risk analysis report.
8. End with EXACTLY this block:

RISK_LEVEL: low | moderate | high | extreme
STABILITY_SCORE: <float 0.0 (unstable) to 1.0 (stable)>
MAX_DRAWDOWN_ESTIMATE: <e.g. -15% to -25% worst case>
CONFIDENCE: <float 0.0-1.0>
PRIMARY_RISK: <single biggest risk factor, one sentence>
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a QuantIntel agent. Tools: {tool_names}\n{system_prompt}\n"
     "Date: {trade_date}. Ticker: {ticker}\n\n"
     "--- PRIOR AGENT REPORTS ---\n"
     "FUNDAMENTALS:\n{fundamentals_report}\n\n"
     "SENTIMENT:\n{sentiment_report}\n\n"
     "TECHNICAL:\n{technical_report}\n"
     "---------------------------"),
    MessagesPlaceholder(variable_name="messages"),
])


def create_risk_agent(llm):
    def node(state):
        filled = _PROMPT.partial(
            system_prompt=SYSTEM_PROMPT,
            tool_names=", ".join(t.name for t in TOOLS),
            trade_date=state["trade_date"],
            ticker=state["ticker"],
            fundamentals_report=state.get("fundamentals_report", "Not yet available."),
            sentiment_report=state.get("sentiment_report",    "Not yet available."),
            technical_report=state.get("technical_report",    "Not yet available."),
        )
        result = (filled | llm.bind_tools(TOOLS)).invoke(state["messages"])
        report = result.content if not result.tool_calls else ""
        return {"messages": [result], "risk_report": report, "sender": "risk_agent"}
    return node


def get_risk_tool_node():
    return ToolNode(TOOLS)
