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
    
    async def runner(ticker: str, trade_date: str, fundamentals_report: str = "", sentiment_report: str = "", technical_report: str = "", ground_truth: str = "") -> str:
        # Pre-fetch ground truth so agents have verified data to anchor against
        gt_context = ""
        try:
            gt = None
            if ground_truth:
                try:
                    import json
                    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
                except Exception:
                    pass

            if not gt:
                from quantintel.dataflows.y_finance import get_ground_truth_context
                gt = get_ground_truth_context(ticker)

            if gt and isinstance(gt, dict):
                def _fmt(v):
                    if v is None:
                        return None
                    if isinstance(v, (int, float)) and abs(v) > 1e6:
                        return f"${v/1e9:,.2f}B"
                    return str(v)

                lines = []
                comp_name = gt.get('company_name', ticker)
                sector = gt.get('actual_sector')
                industry = gt.get('actual_industry')
                if sector and industry:
                    lines.append(f"• Company: {comp_name} ({ticker}) | Sector: {sector} | Industry: {industry}")

                cur_p = gt.get('current_price')
                mcap = _fmt(gt.get('market_cap'))
                if cur_p:
                    lines.append(f"• Current Market Price: {gt.get('currency', 'USD')} {cur_p}" + (f" | Market Cap: {mcap}" if mcap else ""))

                # Valuation multiples (only include non-null values)
                mults = []
                if gt.get('trailing_pe'): mults.append(f"Trailing P/E: {gt.get('trailing_pe')}x")
                if gt.get('forward_pe'): mults.append(f"Forward P/E: {gt.get('forward_pe')}x")
                if gt.get('peg_ratio'): mults.append(f"PEG: {gt.get('peg_ratio')}")
                if gt.get('price_to_book'): mults.append(f"P/B: {gt.get('price_to_book')}x")
                if gt.get('trailing_eps'): mults.append(f"EPS (TTM): ${gt.get('trailing_eps')}")
                if gt.get('forward_eps'): mults.append(f"Forward EPS: ${gt.get('forward_eps')}")
                if mults: lines.append("• Valuation Multiples: " + " | ".join(mults))

                # Margins & profitability
                margs = []
                if gt.get('gross_margin_pct'): margs.append(f"Gross Margin: {gt.get('gross_margin_pct')}%")
                if gt.get('operating_margin_pct'): margs.append(f"Operating Margin: {gt.get('operating_margin_pct')}%")
                if gt.get('profit_margin_pct'): margs.append(f"Net Margin: {gt.get('profit_margin_pct')}%")
                if gt.get('roe_pct'): margs.append(f"ROE: {gt.get('roe_pct')}%")
                if gt.get('roa_pct'): margs.append(f"ROA: {gt.get('roa_pct')}%")
                if margs: lines.append("• Margins & Profitability: " + " | ".join(margs))

                # Balance sheet & cash
                bs_items = []
                if gt.get('debt_to_equity') is not None: bs_items.append(f"Debt-to-Equity: {gt.get('debt_to_equity')}x")
                if gt.get('current_ratio'): bs_items.append(f"Current Ratio: {gt.get('current_ratio')}")
                if gt.get('total_cash'): bs_items.append(f"Total Cash: {_fmt(gt.get('total_cash'))}")
                if gt.get('total_debt'): bs_items.append(f"Total Debt: {_fmt(gt.get('total_debt'))}")
                if gt.get('free_cashflow'): bs_items.append(f"Free Cash Flow: {_fmt(gt.get('free_cashflow'))}")
                if gt.get('fcf_margin_pct'): bs_items.append(f"FCF Margin: {gt.get('fcf_margin_pct')}%")
                if bs_items: lines.append("• Balance Sheet & Cash Quality: " + " | ".join(bs_items))

                # Scale & growth
                grow_items = []
                if gt.get('total_revenue'): grow_items.append(f"Revenue (TTM): {_fmt(gt.get('total_revenue'))}")
                if gt.get('revenue_growth_yoy_pct'): grow_items.append(f"Revenue Growth YoY: {gt.get('revenue_growth_yoy_pct')}%")
                if gt.get('earnings_growth_pct'): grow_items.append(f"Earnings Growth: {gt.get('earnings_growth_pct')}%")
                if grow_items: lines.append("• Revenue & Growth: " + " | ".join(grow_items))

                # Risk & 52W levels
                risk_items = []
                if gt.get('beta'): risk_items.append(f"Beta: {gt.get('beta')}")
                if gt.get('fifty_two_week_high'): risk_items.append(f"52W High: ${gt.get('fifty_two_week_high')}")
                if gt.get('fifty_two_week_low'): risk_items.append(f"52W Low: ${gt.get('fifty_two_week_low')}")
                if gt.get('one_year_return_pct') is not None: risk_items.append(f"1-Yr Stock Return: {gt.get('one_year_return_pct')}%")
                if risk_items: lines.append("• Market Benchmarks: " + " | ".join(risk_items))

                if lines:
                    gt_context = (
                        "\n=== VERIFIED INSTITUTIONAL GROUND TRUTH (Yahoo Finance) ===\n"
                        + "\n".join(lines)
                        + "\n=== END GROUND TRUTH ===\n"
                    )
        except Exception as e:
            print(f"[Ground Truth] Warning: Could not fetch for {ticker}: {e}")
            gt_context = ""

        initial_msg = (
            f"Perform an exhaustive, institutional-grade equity research analysis of {ticker} as of trading date {trade_date}."
            f"\nDeliver a detailed multi-paragraph report (at least 600-900 words) across all required sections. Do NOT write brief single-sentence summaries or refusal lines."
            + gt_context
        )

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
async def ask_fundamentals_agent(ticker: str, trade_date: str, api_key: str = "", ground_truth: str = "") -> str:
    """Call this agent to perform a deep-dive fundamental analysis of a company. 
    It evaluates core financial metrics including P/E ratios, profit margins, debt-to-equity ratios, and free cash flow. 
    Returns a comprehensive valuation report indicating whether the stock is overvalued, undervalued, or fairly priced, along with a firm BUY/HOLD/SELL fundamental signal."""
    runner = _get_runner("fundamentals", api_key)
    return await runner(ticker, trade_date, ground_truth=ground_truth)

@mcp.tool()
async def ask_sentiment_agent(ticker: str, trade_date: str, api_key: str = "", ground_truth: str = "") -> str:
    """Call this agent to analyze recent news, market narratives, and media sentiment surrounding the stock. 
    It processes recent headlines and articles to gauge public and institutional perception. 
    Returns a detailed sentiment report summarizing key news catalysts, the overarching narrative, and a bullish/bearish/neutral sentiment score."""
    runner = _get_runner("sentiment", api_key)
    return await runner(ticker, trade_date, ground_truth=ground_truth)

@mcp.tool()
async def ask_technical_agent(ticker: str, trade_date: str, api_key: str = "", ground_truth: str = "") -> str:
    """Call this agent to perform technical price-action and momentum analysis on the stock. 
    It evaluates price data, moving averages, relative strength, and momentum oscillators. 
    Returns a technical analysis report detailing current trend direction, key support/resistance levels, and short-to-medium term trading signals."""
    runner = _get_runner("technical", api_key)
    return await runner(ticker, trade_date, ground_truth=ground_truth)

@mcp.tool()
async def ask_risk_agent(ticker: str, trade_date: str, fundamentals_report: str = "", sentiment_report: str = "", technical_report: str = "", api_key: str = "", ground_truth: str = "") -> str:
    """Call this agent to quantify the downside risk and volatility profile of the stock. 
    It calculates key risk metrics such as Average True Range (ATR), beta, Value at Risk (VaR), and historical drawdowns.
    IMPORTANT: Provide the fundamentals, sentiment, and technical reports generated by the other agents so the Risk agent can synthesize them.
    Returns a comprehensive risk report outputting the quantified risk level (low/moderate/high) and an assessment of potential downside exposure."""
    runner = _get_runner("risk", api_key)
    return await runner(ticker, trade_date, fundamentals_report, sentiment_report, technical_report, ground_truth=ground_truth)

@mcp.tool()
async def ask_macro_agent(ticker: str, trade_date: str, api_key: str = "", ground_truth: str = "") -> str:
    """Call this agent to analyze the broader macroeconomic environment and monetary policy context. 
    It evaluates Federal Reserve regimes, interest rate directions, inflation dynamics, and broader market conditions. 
    Returns a macro regime report (e.g., Risk-On vs. Risk-Off), assessing how current economic headwinds or tailwinds impact the market and the specific stock."""
    runner = _get_runner("macro", api_key)
    return await runner(ticker, trade_date, ground_truth=ground_truth)

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
