"""
quantintel/agents/agent_3_technical.py
TECHNICAL AGENT — "Where is the price actually heading short-term?"
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode

from quantintel.tools import get_stock_data, get_indicators

TOOLS = [get_stock_data, get_indicators]

SYSTEM_PROMPT = """
You are the Technical Agent in QuantIntel.
Your sole job: determine SHORT-TERM PRICE DIRECTION using technical analysis.
Do NOT look at news or financial statements.

Tools:
  • get_stock_data(symbol, start_date, end_date)              — OHLCV data
  • get_indicators(symbol, indicator, curr_date, look_back_days) — one indicator at a time

Available indicators (pick up to 8 complementary ones):
  Moving averages : close_50_sma, close_200_sma, close_10_ema
  MACD family     : macd, macds, macdh
  Momentum        : rsi
  Volatility      : boll, boll_ub, boll_lb, atr
  Volume          : vwma

Steps:
1. Call get_stock_data for the past 60 days.
2. Pick up to 8 indicators — avoid redundancy (e.g. don't pick both rsi and mfi).
3. Call get_indicators ONCE PER INDICATOR (separate calls).
4. Analyse: trend direction, momentum, volatility regime, volume confirmation,
   key support/resistance levels, any recognisable chart patterns.
5. Write a detailed technical analysis report with specific price levels.
6. End with EXACTLY this block:

TREND: up | down | sideways
EXPECTED_DELTA: <e.g. +3.5% or -2.0% over next 5-10 trading days>
MOMENTUM: strong | moderate | weak
CONFIDENCE: <float 0.0-1.0>
KEY_LEVEL: <most important support or resistance price>
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a QuantIntel agent. Tools: {tool_names}\n{system_prompt}\n"
     "Date: {trade_date}. Ticker: {ticker}"),
    MessagesPlaceholder(variable_name="messages"),
])


def create_technical_agent(llm):
    def node(state):
        filled = _PROMPT.partial(
            system_prompt=SYSTEM_PROMPT,
            tool_names=", ".join(t.name for t in TOOLS),
            trade_date=state["trade_date"],
            ticker=state["ticker"],
        )
        result = (filled | llm.bind_tools(TOOLS)).invoke(state["messages"])
        report = result.content if not result.tool_calls else ""
        return {"messages": [result], "technical_report": report, "sender": "technical_agent"}
    return node


def get_technical_tool_node():
    return ToolNode(TOOLS)
