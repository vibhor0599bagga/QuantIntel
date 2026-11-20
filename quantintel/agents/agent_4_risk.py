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
You are the Chief Risk Officer & Quantitative Risk Guard in QuantIntel.
Your sole mission: produce an exhaustive, deep-dive multi-paragraph risk synthesis report quantifying the downside risks, historical drawdowns, and tail-risk exposure of taking a position in this asset.

Tools available:
  • get_stock_data(symbol, start_date, end_date)
  • get_indicators(symbol, indicator, curr_date, look_back_days)

EXECUTION INSTRUCTIONS:
1. Thoroughly synthesize the prior agent reports (Fundamentals, Sentiment, Technical) and the ground truth context provided below.
2. If necessary, call get_indicators for ATR or Bollinger Bands to model downside limits.
3. Reason from three distinct risk postures before synthesizing:
   - AGGRESSIVE: Upside/downside asymmetry and volatility capacity.
   - CONSERVATIVE: Capital preservation, worst-case stress tests, and permanent impairment risks.
   - NEUTRAL: Statistical probability distributions and empirical risk modeling.
4. Write an extensive, deeply detailed institutional risk report (at least 500-750 words, organized into 4 comprehensive sections with multi-paragraph analysis):

### 1. Volatility Regime & Historical Drawdown Modeling
Quantify historical price volatility, ATR levels, and maximum peak-to-trough drawdowns under market stress. Contrast current volatility with historical benchmarks.

### 2. Multi-Perspective Risk Posture Synthesis (Aggressive / Conservative / Neutral)
Provide an in-depth dialectical breakdown contrasting bullish assumptions against conservative downside perils, multiple compression hazards, and black-swan tail exposures.

### 3. Cross-Disciplinary Risk Factors
Integrate valuation risk (from Fundamentals), sentiment fragility (from Sentiment), and technical support breakdown levels (from Technical) into a unified risk matrix.

### 4. Tail Risks & Risk Mitigation Recommendations
Identify asymmetric tail risks (macroeconomic tightening, sector rotation, geopolitical disruption) and provide concrete, quantitative position-sizing and stop-loss defense parameters.

CRITICAL RULES:
- LENGTH & DEPTH: Each section MUST consist of at least 2 substantive, well-developed paragraphs. Never write brief bullet points or single-sentence answers.
- BETA FORMATTING: Beta is a unitless relative sensitivity benchmark (where Benchmark = 1.0). NEVER express beta as a percentage (e.g. do NOT say "10% beta").

7. End your response with EXACTLY this structured block at the very bottom:

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
        if report and (report.strip().startswith("{") or report.strip().startswith("```")):
            if '"name"' in report:
                report = ""
        return {"messages": [result], "risk_report": report, "sender": "risk_agent"}
    return node


def get_risk_tool_node():
    return ToolNode(TOOLS)
