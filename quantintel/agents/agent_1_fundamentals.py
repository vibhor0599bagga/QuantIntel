"""
quantintel/agents/agent_1_fundamentals.py
FUNDAMENTALS AGENT — "What is this company actually worth?"
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode

from quantintel.tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
)

TOOLS = [get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement]

SYSTEM_PROMPT = """
You are the Fundamentals Agent in QuantIntel.
Your sole job: determine the INTRINSIC VALUE of the company vs its market price.

Tools available:
  • get_fundamentals       — P/E, EPS, market cap, margins, ROE, debt ratios
  • get_balance_sheet      — assets, liabilities, equity
  • get_cashflow           — operating / investing / financing cash flows
  • get_income_statement   — revenue, gross profit, net income, EBITDA

Steps:
1. Call get_fundamentals first.
2. Call get_balance_sheet, get_cashflow, get_income_statement for depth.
3. Analyse: valuation multiples, profitability, growth, financial health, cash quality.
4. Write a detailed report.
5. End with EXACTLY this block:

VALUATION_SIGNAL: undervalued | fair | overvalued
CONFIDENCE: <float 0.0-1.0>
KEY_REASON: <one sentence>
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a QuantIntel agent. Tools: {tool_names}\n{system_prompt}\n"
     "Date: {trade_date}. Ticker: {ticker}"),
    MessagesPlaceholder(variable_name="messages"),
])


def create_fundamentals_agent(llm):
    def node(state):
        filled = _PROMPT.partial(
            system_prompt=SYSTEM_PROMPT,
            tool_names=", ".join(t.name for t in TOOLS),
            trade_date=state["trade_date"],
            ticker=state["ticker"],
        )
        result = (filled | llm.bind_tools(TOOLS)).invoke(state["messages"])
        report = result.content if not result.tool_calls else ""
        return {"messages": [result], "fundamentals_report": report, "sender": "fundamentals_agent"}
    return node


def get_fundamentals_tool_node():
    return ToolNode(TOOLS)
