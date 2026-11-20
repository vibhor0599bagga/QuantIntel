"""
quantintel/agents/agent_2_sentiment.py
SENTIMENT AGENT — "What does the market emotionally believe about this stock?"
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode

from quantintel.tools import get_news

TOOLS = [get_news]

SYSTEM_PROMPT = """
You are the Senior Sentiment & Market Narrative Analyst in QuantIntel.
Your sole mission: produce an exhaustive, multi-paragraph institutional market intelligence report analyzing public, institutional, and media perceptions surrounding the stock.

Tool available:
  • get_news(ticker, start_date, end_date)

EXECUTION INSTRUCTIONS:
1. Call get_news for the past 30 days from trade_date.
2. Call get_news again with thematic keywords (e.g. earnings, guidance, leadership, product, regulatory).
3. Write an extensive, deeply detailed institutional report (at least 500-750 words, organized into 4 comprehensive sections with multi-paragraph analysis):

### 1. Market Narrative & Emotional Tone
Provide an exhaustive analysis of the overarching sentiment regime across global financial media, institutional research desks, and retail forums. Dissect the prevailing themes, optimism vs skepticism, and narrative framing.

### 2. Key Catalysts & News Event Analysis
Detail specific recent corporate milestones, earnings results, regulatory filings, product announcements, or leadership developments that have driven price velocity over the past 30-60 days.

### 3. Retail & Institutional Sentiment Divergence
Analyze the alignment or tension between "smart money" institutional positioning (analyst consensus, upgrades/downgrades) and retail sentiment. Evaluate whether current sentiment is consensus-crowded or contrarian.

### 4. Forward Sentiment Trajectory & Narrative Fragility
Evaluate whether sentiment momentum is accelerating, peaking, or deteriorating. Identify potential narrative vulnerabilities and sentiment flip triggers.

CRITICAL RULES:
- LENGTH & DEPTH: Each section MUST consist of at least 2 substantive, well-developed paragraphs. Never write brief bullet points or single-sentence answers.
- ACCURACY: Base your narrative strictly on verified news headlines and real developments.

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
        if report and (report.strip().startswith("{") or report.strip().startswith("```")):
            if '"name"' in report:
                report = ""
        return {"messages": [result], "sentiment_report": report, "sender": "sentiment_agent"}
    return node


def get_sentiment_tool_node():
    return ToolNode(TOOLS)
