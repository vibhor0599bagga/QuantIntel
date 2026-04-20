from mcp.server.fastmcp import FastMCP
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from quantintel.config import DEFAULT_CONFIG, set_config
from quantintel.llm_clients import create_llm_client
from quantintel.agents.states import QuantIntelState

from quantintel.agents.agent_1_fundamentals import create_fundamentals_agent, get_fundamentals_tool_node
from quantintel.agents.agent_2_sentiment    import create_sentiment_agent,    get_sentiment_tool_node
from quantintel.agents.agent_3_technical    import create_technical_agent,    get_technical_tool_node
from quantintel.agents.agent_4_risk         import create_risk_agent,         get_risk_tool_node
from quantintel.agents.agent_5_macro        import create_macro_agent,        get_macro_tool_node

# We need a mini-graph for each agent to run its tool loop
def _msg_clear_node():
    from langchain_core.messages import RemoveMessage
    def clear(state):
        ops = [RemoveMessage(id=m.id) for m in state["messages"]]
        return {"messages": ops + [HumanMessage(content="Continue")]}
    return clear

def _should_continue(agent_key: str):
    def router(state):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return f"tools_{agent_key}"
        return END
    return router

def _create_runner(agent_key, agent_node, tool_node) -> callable:
    wf = StateGraph(QuantIntelState)
    wf.add_node(f"agent_{agent_key}", agent_node)
    wf.add_node(f"tools_{agent_key}", tool_node)
    
    wf.add_edge(START, f"agent_{agent_key}")
    wf.add_conditional_edges(
        f"agent_{agent_key}", 
        _should_continue(agent_key),
        {f"tools_{agent_key}": f"tools_{agent_key}", END: END}
    )
    wf.add_edge(f"tools_{agent_key}", f"agent_{agent_key}")
    compiled = wf.compile()
    
    async def runner(ticker: str, trade_date: str) -> str:
        initial_state = {
            "messages":             [HumanMessage(content=f"Analyse {ticker}")],
            "ticker":               ticker,
            "trade_date":           trade_date,
            "portfolio_context":    {},
            "fundamentals_report":  "",
            "sentiment_report":     "",
            "technical_report":     "",
            "risk_report":          "",
            "macro_report":         "",
            "final_recommendation": "",
            "sender":               "",
        }
        result = await compiled.ainvoke(initial_state)
        # extract the specific report
        return result.get(f"{agent_key}_report", "No report generated.")
    return runner

# Initialize Global config and LLM
set_config(DEFAULT_CONFIG)
llm = create_llm_client(
    provider=DEFAULT_CONFIG["llm_provider"],
    model=DEFAULT_CONFIG["quick_think_llm"],
    base_url=DEFAULT_CONFIG.get("backend_url"),
).get_llm()

fundamentals_runner = _create_runner("fundamentals", create_fundamentals_agent(llm), get_fundamentals_tool_node())
sentiment_runner    = _create_runner("sentiment",    create_sentiment_agent(llm),    get_sentiment_tool_node())
technical_runner    = _create_runner("technical",    create_technical_agent(llm),    get_technical_tool_node())
risk_runner         = _create_runner("risk",         create_risk_agent(llm),         get_risk_tool_node())
macro_runner        = _create_runner("macro",        create_macro_agent(llm),        get_macro_tool_node())

mcp = FastMCP("QuantIntel Swarm Agents")

@mcp.tool()
async def ask_fundamentals_agent(ticker: str, trade_date: str) -> str:
    """Call this agent to perform a deep-dive fundamental analysis of a company. 
    It evaluates core financial metrics including P/E ratios, profit margins, debt-to-equity ratios, and free cash flow. 
    Returns a comprehensive valuation report indicating whether the stock is overvalued, undervalued, or fairly priced, along with a firm BUY/HOLD/SELL fundamental signal."""
    return await fundamentals_runner(ticker, trade_date)

@mcp.tool()
async def ask_sentiment_agent(ticker: str, trade_date: str) -> str:
    """Call this agent to analyze recent news, market narratives, and media sentiment surrounding the stock. 
    It processes recent headlines and articles to gauge public and institutional perception. 
    Returns a detailed sentiment report summarizing key news catalysts, the overarching narrative, and a bullish/bearish/neutral sentiment score."""
    return await sentiment_runner(ticker, trade_date)

@mcp.tool()
async def ask_technical_agent(ticker: str, trade_date: str) -> str:
    """Call this agent to perform technical price-action and momentum analysis on the stock. 
    It evaluates price data, moving averages, relative strength, and momentum oscillators. 
    Returns a technical analysis report detailing current trend direction, key support/resistance levels, and short-to-medium term trading signals."""
    return await technical_runner(ticker, trade_date)

@mcp.tool()
async def ask_risk_agent(ticker: str, trade_date: str) -> str:
    """Call this agent to quantify the downside risk and volatility profile of the stock. 
    It calculates key risk metrics such as Average True Range (ATR), beta, Value at Risk (VaR), and historical drawdowns. 
    Returns a comprehensive risk report outputting the quantified risk level (low/moderate/high) and an assessment of potential downside exposure."""
    return await risk_runner(ticker, trade_date)

@mcp.tool()
async def ask_macro_agent(ticker: str, trade_date: str) -> str:
    """Call this agent to analyze the broader macroeconomic environment and monetary policy context. 
    It evaluates Federal Reserve regimes, interest rate directions, inflation dynamics, and broader market conditions. 
    Returns a macro regime report (e.g., Risk-On vs. Risk-Off), assessing how current economic headwinds or tailwinds impact the market and the specific stock."""
    return await macro_runner(ticker, trade_date)

# Expose a resource over MCP
@mcp.resource("portfolio://user/current")
def get_user_portfolio() -> str:
    """Returns the user's current portfolio and risk tolerance limits."""
    return str({
        "sector_exposure": "Tech Heavy",
        "horizon": "Long Term",
        "risk_tolerance": "moderate",
        "holdings": ["AAPL", "MSFT"]
    })

if __name__ == "__main__":
    mcp.run()
