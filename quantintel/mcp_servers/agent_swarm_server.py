from dotenv import load_dotenv
load_dotenv()

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

        # Robust fallback: inspect raw content for tool calls if model returned JSON text
        content = getattr(last, "content", "")
        if isinstance(content, str) and '"name"' in content and ('"parameters"' in content or '"args"' in content or '"arguments"' in content):
            from quantintel.llm_clients.base_client import extract_raw_tool_calls
            recovered = extract_raw_tool_calls(content)
            if recovered:
                last.tool_calls = recovered
                last.content = ""
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
    
    async def runner(ticker: str, trade_date: str, fundamentals_report: str = "", sentiment_report: str = "", technical_report: str = "") -> str:
        # Pre-fetch ground truth so agents have verified data to anchor against
        gt_context = ""
        try:
            from quantintel.dataflows.y_finance import get_ground_truth_context
            gt = get_ground_truth_context(ticker)
            if gt:
                def _fmt(v):
                    if v is None:
                        return "N/A"
                    if isinstance(v, (int, float)) and abs(v) > 1e6:
                        return f"${v/1e9:,.2f}B"
                    return str(v)
                gt_context = (
                    f"\n=== VERIFIED GROUND TRUTH (Yahoo Finance — use these numbers, do NOT invent your own) ===\n"
                    f"Company: {gt.get('company_name', ticker)} | Sector: {gt.get('actual_sector', 'N/A')} | Industry: {gt.get('actual_industry', 'N/A')}\n"
                    f"Current Price: {gt.get('currency', 'USD')} {gt.get('current_price', 'N/A')} | Market Cap: {_fmt(gt.get('market_cap'))}\n"
                    f"Trailing P/E: {gt.get('trailing_pe', 'N/A')} | Forward P/E: {gt.get('forward_pe', 'N/A')} | PEG: {gt.get('peg_ratio', 'N/A')} | P/B: {gt.get('price_to_book', 'N/A')}\n"
                    f"EPS (TTM): {gt.get('trailing_eps', 'N/A')} | Forward EPS: {gt.get('forward_eps', 'N/A')} | Book Value/Share: {gt.get('book_value', 'N/A')}\n"
                    f"Gross Margin: {gt.get('gross_margin_pct', 'N/A')}% | Operating Margin: {gt.get('operating_margin_pct', 'N/A')}% | Net Margin: {gt.get('profit_margin_pct', 'N/A')}%\n"
                    f"ROE: {gt.get('roe_pct', 'N/A')}% | ROA: {gt.get('roa_pct', 'N/A')}%\n"
                    f"Debt/Equity: {gt.get('debt_to_equity', 'N/A')} | Current Ratio: {gt.get('current_ratio', 'N/A')}\n"
                    f"Total Cash: {_fmt(gt.get('total_cash'))} | Total Debt: {_fmt(gt.get('total_debt'))} | FCF: {_fmt(gt.get('free_cashflow'))} | FCF Margin: {gt.get('fcf_margin_pct', 'N/A')}%\n"
                    f"Revenue (TTM): {_fmt(gt.get('total_revenue'))} | Revenue Growth YoY: {gt.get('revenue_growth_yoy_pct', 'N/A')}% | Earnings Growth: {gt.get('earnings_growth_pct', 'N/A')}%\n"
                    f"Beta: {gt.get('beta', 'N/A')} | 52W High: {gt.get('fifty_two_week_high', 'N/A')} | 52W Low: {gt.get('fifty_two_week_low', 'N/A')}\n"
                    f"1-Year Return: {gt.get('one_year_return_pct', 'N/A')}% | Dividend Yield: {gt.get('dividend_yield_pct', 'N/A')}%\n"
                    f"=== END GROUND TRUTH ===\n"
                )
        except Exception as e:
            print(f"[Ground Truth] Warning: Could not fetch for {ticker}: {e}")
            gt_context = ""

        initial_msg = f"Analyse {ticker}" + gt_context

        initial_state = {
            "messages":             [HumanMessage(content=initial_msg)],
            "ticker":               ticker,
            "trade_date":           trade_date,
            "portfolio_context":    {},
            "fundamentals_report":  fundamentals_report,
            "sentiment_report":     sentiment_report,
            "technical_report":     technical_report,
            "risk_report":          "",
            "macro_report":         "",
            "final_recommendation": "",
            "sender":               "",
        }
        result = await compiled.ainvoke(initial_state)
        # extract the specific report
        report = result.get(f"{agent_key}_report", "")
        if report and report.strip().startswith("{") and '"name"' in report and ('"parameters"' in report or '"args"' in report):
            report = ""

        if not report:
            for m in reversed(result.get("messages", [])):
                content = getattr(m, "content", "")
                if content and not getattr(m, "tool_calls", None):
                    trimmed = content.strip()
                    if not (trimmed.startswith("{") and '"name"' in trimmed and ('"parameters"' in trimmed or '"args"' in trimmed)):
                        report = content
                        break
        return report or "No report generated."
    return runner

# Initialize Global config
set_config(DEFAULT_CONFIG)

_runners_cache = {}

def _get_runner(agent_key: str, api_key: str = ""):
    clean_key = api_key.strip() if api_key else ""
    cache_key = (agent_key, clean_key)
    if cache_key in _runners_cache:
        return _runners_cache[cache_key]

    agent_llm = create_llm_client(
        provider=DEFAULT_CONFIG["llm_provider"],
        model=DEFAULT_CONFIG["quick_think_llm"],
        base_url=DEFAULT_CONFIG.get("backend_url"),
        api_key=clean_key if clean_key else None,
    ).get_llm()

    if agent_key == "fundamentals":
        runner = _create_runner("fundamentals", create_fundamentals_agent(agent_llm), get_fundamentals_tool_node())
    elif agent_key == "sentiment":
        runner = _create_runner("sentiment", create_sentiment_agent(agent_llm), get_sentiment_tool_node())
    elif agent_key == "technical":
        runner = _create_runner("technical", create_technical_agent(agent_llm), get_technical_tool_node())
    elif agent_key == "risk":
        runner = _create_runner("risk", create_risk_agent(agent_llm), get_risk_tool_node())
    elif agent_key == "macro":
        runner = _create_runner("macro", create_macro_agent(agent_llm), get_macro_tool_node())
    else:
        raise ValueError(f"Unknown agent key: {agent_key}")

    _runners_cache[cache_key] = runner
    return runner

mcp = FastMCP("QuantIntel Swarm Agents")

@mcp.tool()
async def ask_fundamentals_agent(ticker: str, trade_date: str, api_key: str = "") -> str:
    """Call this agent to perform a deep-dive fundamental analysis of a company. 
    It evaluates core financial metrics including P/E ratios, profit margins, debt-to-equity ratios, and free cash flow. 
    Returns a comprehensive valuation report indicating whether the stock is overvalued, undervalued, or fairly priced, along with a firm BUY/HOLD/SELL fundamental signal."""
    runner = _get_runner("fundamentals", api_key)
    return await runner(ticker, trade_date)

@mcp.tool()
async def ask_sentiment_agent(ticker: str, trade_date: str, api_key: str = "") -> str:
    """Call this agent to analyze recent news, market narratives, and media sentiment surrounding the stock. 
    It processes recent headlines and articles to gauge public and institutional perception. 
    Returns a detailed sentiment report summarizing key news catalysts, the overarching narrative, and a bullish/bearish/neutral sentiment score."""
    runner = _get_runner("sentiment", api_key)
    return await runner(ticker, trade_date)

@mcp.tool()
async def ask_technical_agent(ticker: str, trade_date: str, api_key: str = "") -> str:
    """Call this agent to perform technical price-action and momentum analysis on the stock. 
    It evaluates price data, moving averages, relative strength, and momentum oscillators. 
    Returns a technical analysis report detailing current trend direction, key support/resistance levels, and short-to-medium term trading signals."""
    runner = _get_runner("technical", api_key)
    return await runner(ticker, trade_date)

@mcp.tool()
async def ask_risk_agent(ticker: str, trade_date: str, fundamentals_report: str = "", sentiment_report: str = "", technical_report: str = "", api_key: str = "") -> str:
    """Call this agent to quantify the downside risk and volatility profile of the stock. 
    It calculates key risk metrics such as Average True Range (ATR), beta, Value at Risk (VaR), and historical drawdowns.
    IMPORTANT: Provide the fundamentals, sentiment, and technical reports generated by the other agents so the Risk agent can synthesize them.
    Returns a comprehensive risk report outputting the quantified risk level (low/moderate/high) and an assessment of potential downside exposure."""
    runner = _get_runner("risk", api_key)
    return await runner(ticker, trade_date, fundamentals_report, sentiment_report, technical_report)

@mcp.tool()
async def ask_macro_agent(ticker: str, trade_date: str, api_key: str = "") -> str:
    """Call this agent to analyze the broader macroeconomic environment and monetary policy context. 
    It evaluates Federal Reserve regimes, interest rate directions, inflation dynamics, and broader market conditions. 
    Returns a macro regime report (e.g., Risk-On vs. Risk-Off), assessing how current economic headwinds or tailwinds impact the market and the specific stock."""
    runner = _get_runner("macro", api_key)
    return await runner(ticker, trade_date)

# Dynamic in-memory user portfolio profile (updated at runtime from dashboard)
_active_user_portfolio = {
    "sector_exposure": "diversified",
    "horizon": "medium_term",
    "risk_tolerance": "moderate",
    "holdings": []
}

@mcp.tool()
async def update_user_portfolio(
    sector_exposure: str = "diversified",
    horizon: str = "medium_term",
    risk_tolerance: str = "moderate",
    holdings: list = None
) -> str:
    """Updates the user's active portfolio context, time horizon, and risk tolerance mandate on the MCP server."""
    global _active_user_portfolio
    _active_user_portfolio = {
        "sector_exposure": sector_exposure,
        "horizon": horizon,
        "risk_tolerance": risk_tolerance,
        "holdings": holdings if holdings is not None else []
    }
    return f"Active user portfolio updated: {_active_user_portfolio}"

# Expose a resource over MCP
@mcp.resource("portfolio://user/current")
def get_user_portfolio() -> str:
    """Returns the user's current portfolio and risk tolerance limits."""
    import json
    return json.dumps(_active_user_portfolio)

if __name__ == "__main__":
    mcp.run()
