"""
quantintel/agents/agent_2_sentiment.py
SENTIMENT AGENT — "What does the market emotionally believe about this stock?"
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode

from quantintel.tools import get_news

TOOLS = [get_news]

SYSTEM_PROMPT = """
You are the Sentiment Agent in QuantIntel.
Your sole job: measure HOW THE MARKET FEELS about this company.
Do NOT look at price charts or financial statements.

Tool: get_news(ticker, start_date, end_date)

Steps:
1. Call get_news for the past 30 days from trade_date.
2. Call get_news again over the same 30-day window with broader keywords (e.g. leadership, earnings, product line, sector).
3. Write a comprehensive, detailed institutional report (at least 3-4 paragraphs) organized into these sections:
   ### 1. Market Narrative & Emotional Tone
   Analyze the prevailing sentiment across headlines, media tone, and investor perception.
   ### 2. Key Catalysts & News Events
   Detail specific recent events, earnings releases, leadership updates, or regulatory coverage.
   ### 3. Retail & Institutional Sentiment Divergence
   Discuss whether institutional analyst upgrades/downgrades and general retail sentiment align or conflict.
   ### 4. Forward Sentiment Trajectory
   Evaluate whether sentiment momentum is deteriorating, stable, or accelerating positively.

4. End your response with EXACTLY this structured block at the very bottom:

SENTIMENT_SCORE: <float -1.0 to +1.0>
POLARITY: bearish | neutral | bullish
CONFIDENCE: <float 0.0-1.0>
KEY_CATALYST: <main driver of current sentiment, one sentence>
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a QuantIntel agent. Tools: {tool_names}\n{system_prompt}\n"
     "Date: {trade_date}. Ticker: {ticker}"),
    MessagesPlaceholder(variable_name="messages"),
])


def create_sentiment_agent(llm):
    def node(state):
        filled = _PROMPT.partial(
            system_prompt=SYSTEM_PROMPT,
            tool_names=", ".join(t.name for t in TOOLS),
            trade_date=state["trade_date"],
            ticker=state["ticker"],
        )
        result = (filled | llm.bind_tools(TOOLS)).invoke(state["messages"])
        report = result.content if not result.tool_calls else ""
        return {"messages": [result], "sentiment_report": report, "sender": "sentiment_agent"}
    return node


def get_sentiment_tool_node():
    return ToolNode(TOOLS)
