# quantintel/mcp_graph.py
import os
import asyncio
from datetime import datetime
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from quantintel.config import DEFAULT_CONFIG, set_config
from quantintel.memory import FinancialSituationMemory
from quantintel.llm_clients import create_llm_client
from quantintel.agents.states import QuantIntelState
from quantintel.mcp_clients import get_mcp_agent_tools, get_mcp_resource

class McpQuantIntelGraph:
    """
    QuantIntel pipeline using Model Context Protocol (MCP) for multi-agent coordination.
    
    EXPLICIT PHASE-BASED ORCHESTRATION:
    - PHASE 1: Fundamentals, Sentiment, Technical, Macro run IN PARALLEL
    - PHASE 2: Risk agent runs AFTER all Phase 1 results are collected
    - PHASE 3: Supervisor synthesizes final decision
    """
    def __init__(self, session, mcp_tools, config: dict = None, debug: bool = False):
        self.debug  = debug
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        set_config(self.config)

        self.api_key = (
            self.config.get("api_key")
            or self.config.get("openrouter_api_key")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )

        self.deep_llm = create_llm_client(
            provider = self.config["llm_provider"],
            model    = self.config["deep_think_llm"],
            base_url = self.config.get("backend_url"),
            api_key  = self.api_key if self.api_key else None,
        ).get_llm()
        self.memory = FinancialSituationMemory("mcp_memory")
        self.session = session
        self.mcp_tools = mcp_tools
        
        # Build a tools dict for easy lookup
        self.tools_dict = {tool.name: tool for tool in mcp_tools}

    def _extract_text_content(self, val) -> str:
        """Extract pure text from various MCP result structures (lists, dicts, TextContent, AST string representations)."""
        if val is None:
            return ""
        if isinstance(val, str):
            trimmed = val.strip()
            if (trimmed.startswith("[{") and trimmed.endswith("}]")) or (trimmed.startswith("[TextContent(")):
                try:
                    import ast
                    parsed = ast.literal_eval(trimmed)
                    if isinstance(parsed, list):
                        return self._extract_text_content(parsed)
                except Exception:
                    pass
            return val
        if isinstance(val, list):
            extracted = [self._extract_text_content(item) for item in val]
            return "\n".join(e for e in extracted if e)
        if isinstance(val, dict):
            if "text" in val:
                return str(val["text"])
            if "content" in val:
                return self._extract_text_content(val["content"])
            return str(val)
        if hasattr(val, "text"):
            return str(getattr(val, "text"))
        if hasattr(val, "content"):
            return self._extract_text_content(getattr(val, "content"))
        return str(val)

    async def _execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a single MCP tool and return the clean text result."""
        if tool_name not in self.tools_dict:
            return f"ERROR: Tool '{tool_name}' not found"
        
        tool = self.tools_dict[tool_name]
        try:
            raw_result = await tool.ainvoke(kwargs)
            return self._extract_text_content(raw_result)
        except Exception as e:
            return f"ERROR executing {tool_name}: {str(e)}"

    async def _execute_tool_with_timeout(self, tool_name: str, timeout: float = 150.0, **kwargs) -> str:
        try:
            return await asyncio.wait_for(self._execute_tool(tool_name, **kwargs), timeout=timeout)
        except asyncio.TimeoutError:
            return f"[TIMED OUT: {tool_name} did not respond within {timeout}s. Proceeding with remaining agent context.]"
        except Exception as e:
            return f"[ERROR executing {tool_name}: {str(e)}]"

    async def _phase1_parallel_agents(self, state: QuantIntelState) -> dict:
        """
        PHASE 1: Execute all 4 data-gathering agents IN PARALLEL
        Returns: fundamentals_report, sentiment_report, technical_report, macro_report
        """
        ticker = state["ticker"]
        trade_date = state["trade_date"]
        
        # Pre-fetch verified ground-truth financial facts to eliminate hallucinations
        gt = state.get("ground_truth") or {}
        if not gt:
            try:
                from quantintel.dataflows.y_finance import get_ground_truth_context
                gt = get_ground_truth_context(ticker)
            except Exception as e:
                print(f"[Pre-Flight Grounding] Error fetching ground truth for {ticker}: {e}")
                gt = {}

        print("\n" + "="*80)
        print("[PHASE 1] PARALLEL DATA GATHERING (All 4 agents run concurrently)")
        print(f"[GROUND TRUTH] Ticker: {ticker} | Price: {gt.get('current_price')} | P/E: {gt.get('trailing_pe')} | P/B: {gt.get('price_to_book')} | ROE: {gt.get('roe_pct')}% | Op Margin: {gt.get('operating_margin_pct')}% | Sector: {gt.get('actual_sector')}")
        print("="*80)

        # Synchronize user portfolio context from dashboard to MCP server
        user_portfolio = state.get("portfolio_context") or {}
        if user_portfolio and "update_user_portfolio" in self.tools_dict:
            try:
                await self._execute_tool(
                    "update_user_portfolio",
                    sector_exposure=user_portfolio.get("sector_exposure", "diversified"),
                    horizon=user_portfolio.get("horizon", "medium_term"),
                    risk_tolerance=user_portfolio.get("risk_tolerance", "moderate"),
                    holdings=user_portfolio.get("holdings", [])
                )
                print(f"[Portfolio Sync] Synchronized dashboard portfolio to MCP server: {user_portfolio}")
            except Exception as e:
                print(f"[Portfolio Sync] Warning syncing user portfolio: {e}")
        
        # Execute all 4 agents with max 2 concurrent in-flight requests using Semaphore(2)
        sem = asyncio.Semaphore(2)

        async def _run_throttled(tool_name: str) -> str:
            async with sem:
                return await self._execute_tool_with_timeout(
                    tool_name,
                    timeout=150.0,
                    ticker=ticker,
                    trade_date=trade_date,
                    api_key=self.api_key
                )

        tasks = [
            _run_throttled("ask_fundamentals_agent"),
            _run_throttled("ask_sentiment_agent"),
            _run_throttled("ask_technical_agent"),
            _run_throttled("ask_macro_agent"),
        ]
        
        fundamentals_report, sentiment_report, technical_report, macro_report = await asyncio.gather(*tasks)

        
        # Print each report with clear separation
        print("\n" + "-"*80)
        print("[AGENT 1] FUNDAMENTALS OUTPUT:")
        print("-"*80)
        print(fundamentals_report)
        
        print("\n" + "-"*80)
        print("[AGENT 2] SENTIMENT OUTPUT:")
        print("-"*80)
        print(sentiment_report)
        
        print("\n" + "-"*80)
        print("[AGENT 3] TECHNICAL OUTPUT:")
        print("-"*80)
        print(technical_report)
        
        print("\n" + "-"*80)
        print("[AGENT 4] MACRO OUTPUT:")
        print("-"*80)
        print(macro_report)
        
        print("\n[SUCCESS] PHASE 1 COMPLETE: All 4 agents finished\n")
        
        return {
            "fundamentals_report": fundamentals_report,
            "sentiment_report": sentiment_report,
            "technical_report": technical_report,
            "macro_report": macro_report,
            "ground_truth": gt,
            "phase1_complete": True,
        }

    async def _phase2_risk_agent(self, state: QuantIntelState) -> dict:
        """
        PHASE 2: Execute Risk agent with Phase 1 results as context
        The Risk agent now has access to ALL 4 Phase 1 reports
        """
        ticker = state["ticker"]
        trade_date = state["trade_date"]
        
        # Verify all Phase 1 results are available
        if not state.get("phase1_complete"):
            return {"risk_report": "ERROR: Phase 1 not completed", "phase2_complete": False}
        
        # Buffer to allow Phase 1 in-flight requests to settle on OpenRouter
        await asyncio.sleep(2)

        print("\n" + "="*80)
        print("[PHASE 2] RISK SYNTHESIS (Risk agent processes Phase 1 results)")
        print("="*80)
        print("Calling: ask_risk_agent with fundamentals, sentiment, and technical reports...\n")
        
        try:
            # Build proper kwargs dict
            kwargs = {
                "ticker": ticker,
                "trade_date": trade_date,
                "fundamentals_report": state.get("fundamentals_report", ""),
                "sentiment_report": state.get("sentiment_report", ""),
                "technical_report": state.get("technical_report", ""),
                "api_key": self.api_key,
            }
            
            # Look for ask_risk_agent tool
            if "ask_risk_agent" not in self.tools_dict:
                available_tools = list(self.tools_dict.keys())
                risk_report = f"ERROR: ask_risk_agent not found. Available tools: {available_tools}"
            else:
                risk_report = await self._execute_tool_with_timeout("ask_risk_agent", timeout=150.0, **kwargs)

            
            print("\n" + "-"*80)
            print("[AGENT 5] RISK OUTPUT:")
            print("-"*80)
            print(risk_report)
            print("\n[SUCCESS] PHASE 2 COMPLETE: Risk agent finished\n")
            
            return {
                "risk_report": risk_report,
                "phase2_complete": True,
            }
        except Exception as e:
            error_msg = f"ERROR in Phase 2: {str(e)}"
            print(f"\n[ERROR] {error_msg}")
            return {
                "risk_report": error_msg,
                "phase2_complete": True,  # Mark as complete even with error
            }

    async def _phase3_supervisor_decision(self, state: QuantIntelState) -> dict:
        """
        PHASE 3: Supervisor synthesizes all reports and makes final decision
        Input: All 5 agent reports (Fundamentals, Sentiment, Technical, Macro, Risk)
        """
        print("="*80)
        print("[PHASE 3] SUPERVISOR SYNTHESIS & FINAL DECISION")
        print("="*80)
        
        # Verify all phase 2 results
        if not state.get("phase2_complete"):
            return {"final_recommendation": "ERROR: Phase 2 not completed"}
        
        # Buffer to allow Phase 2 in-flight requests to settle on OpenRouter
        await asyncio.sleep(2)

        ticker = state["ticker"]
        trade_date = state["trade_date"]
        
        # 1. Prioritize user portfolio context selected on the frontend dashboard
        user_portfolio = state.get("portfolio_context") or {}
        
        # 2. If missing or empty in state, fallback to MCP server resource
        if not user_portfolio or not any(user_portfolio.values()):
            try:
                portfolio_raw = await get_mcp_resource(self.session, "portfolio://user/current")
                try:
                    import json
                    user_portfolio = json.loads(portfolio_raw)
                except Exception:
                    import ast
                    user_portfolio = ast.literal_eval(portfolio_raw)
            except Exception:
                user_portfolio = {
                    "sector_exposure": "diversified",
                    "horizon": "medium_term",
                    "risk_tolerance": "moderate",
                    "holdings": []
                }
        else:
            # Sync user's runtime dashboard settings to MCP server so MCP resource matches
            if "update_user_portfolio" in self.tools_dict:
                try:
                    await self._execute_tool(
                        "update_user_portfolio",
                        sector_exposure=user_portfolio.get("sector_exposure", "diversified"),
                        horizon=user_portfolio.get("horizon", "medium_term"),
                        risk_tolerance=user_portfolio.get("risk_tolerance", "moderate"),
                        holdings=user_portfolio.get("holdings", [])
                    )
                except Exception:
                    pass

        portfolio = user_portfolio if isinstance(user_portfolio, dict) else {}
        user_risk_tol = portfolio.get("risk_tolerance", "moderate")
        user_horizon = portfolio.get("horizon", "medium_term")
        user_sector_bias = portfolio.get("sector_exposure", "diversified")
        user_holdings = portfolio.get("holdings", [])

        # Retrieve verified ground truth facts from state or pre-flight fetch
        gt = state.get("ground_truth") or {}
        if not gt:
            try:
                from quantintel.dataflows.y_finance import get_ground_truth_context
                gt = get_ground_truth_context(ticker)
            except Exception:
                gt = {}

        actual_sector = gt.get("actual_sector") or ("Energy, Retail & Telecom" if ticker.upper() == "RELIANCE" else "Diversified Equities")
        actual_industry = gt.get("actual_industry") or "Diversified"
        price_str = f"{gt.get('currency', 'USD')} {gt.get('current_price', 'N/A')}"
        pe_str = f"{gt.get('trailing_pe', 'N/A')}"
        fwd_pe_str = f"{gt.get('forward_pe', 'N/A')}"
        eps_str = f"{gt.get('trailing_eps', 'N/A')}"
        de_str = f"{gt.get('debt_to_equity', 'N/A')}"
        pm_str = f"{gt.get('profit_margin_pct', 'N/A')}%"
        opm_str = f"{gt.get('operating_margin_pct', 'N/A')}%"
        gm_str = f"{gt.get('gross_margin_pct', 'N/A')}%"
        roe_str = f"{gt.get('roe_pct', 'N/A')}%"
        pb_str = f"{gt.get('price_to_book', 'N/A')}"
        peg_str = f"{gt.get('peg_ratio', 'N/A')}"
        rev_growth_str = f"{gt.get('revenue_growth_yoy_pct', 'N/A')}%"
        ret_1y_str = f"{gt.get('one_year_return_pct', 'N/A')}%"
        beta_str = f"{gt.get('beta', '1.0')} (Benchmark = 1.0)"
        macro_benchmark = gt.get("macro_benchmark") or "Standard benchmark"

        def _fmt_b(val):
            if val is None:
                return "N/A"
            if isinstance(val, (int, float)) and abs(val) > 1e6:
                return f"${val/1e9:,.2f}B"
            return str(val)

        synthesis_prompt = f"""
You are the Supervisor Agent in QuantIntel. All 5 institutional agents have completed their analysis.

=== TARGET ASSET IDENTITY & VERIFIED FACTS (GROUND TRUTH) ===
• Asset: {ticker} ({gt.get('company_name', ticker)})
• True Sector: {actual_sector} | Industry: {actual_industry}
  [NOTE: The company operates in {actual_sector}. Do NOT confuse this with the user's personal portfolio strategy preference ({user_sector_bias}).]
• Current Market Price: {price_str}
• Trailing P/E: {pe_str} | Forward P/E: {fwd_pe_str} | PEG: {peg_str} | P/B: {pb_str}
• Trailing EPS: {eps_str}
• Gross Margin: {gm_str} | Operating Margin: {opm_str} | Net Profit Margin: {pm_str}
• ROE: {roe_str}
• Debt-to-Equity Ratio: {de_str} | Total Cash: {_fmt_b(gt.get('total_cash'))} | Total Debt: {_fmt_b(gt.get('total_debt'))}
• Revenue (TTM): {_fmt_b(gt.get('total_revenue'))} | Revenue Growth YoY: {rev_growth_str}
• Free Cash Flow: {_fmt_b(gt.get('free_cashflow'))} | FCF Margin: {gt.get('fcf_margin_pct', 'N/A')}%
• Actual 1-Year Historical Stock Performance: {ret_1y_str}
• Beta (Relative to Market Benchmark): {beta_str}
• Domestic Macro Reference: {macro_benchmark}

=== USER PORTFOLIO SETTINGS & MANDATE (CONFIGURED ON DASHBOARD) ===
• User Risk Tolerance: {user_risk_tol.upper()}
• User Investment Horizon: {user_horizon.upper()}
• User Target Strategy / Sector Tilt: {user_sector_bias}
• Existing Holdings: {user_holdings}

MANDATE INSTRUCTIONS:
- Directly incorporate the user's chosen Risk Tolerance ({user_risk_tol}) and Horizon ({user_horizon}) into your recommendation:
  * If risk tolerance is CONSERVATIVE: Demand higher safety margin, penalize high drawdown or volatility, prioritize capital preservation.
  * If risk tolerance is AGGRESSIVE: Weigh growth potential and price momentum more favorably.
  * If horizon is SHORT_TERM: Focus on technical levels, momentum indicators, and short-term news catalysts.
  * If horizon is LONG_TERM: Focus on intrinsic value, earnings quality, and durable competitive advantages.

=== PHASE 1 RESULTS (Parallel Execution) ===

FUNDAMENTALS AGENT:
{state.get('fundamentals_report', 'N/A')}

SENTIMENT AGENT:
{state.get('sentiment_report', 'N/A')}

TECHNICAL AGENT:
{state.get('technical_report', 'N/A')}

MACRO AGENT:
{state.get('macro_report', 'N/A')}

=== PHASE 2 RESULTS (Risk Synthesis) ===

RISK AGENT:
{state.get('risk_report', 'N/A')}

=== YOUR SYNTHESIS TASK ===
Synthesize ALL signals and resolve conflicts using this weighting:
- FUNDAMENTAL DATA:   40% weight (hardest financial facts)
- RISK ASSESSMENT:    30% weight (downside protection critical)
- MACRO ENVIRONMENT:  20% weight (regime context & headwinds/tailwinds)
- TECHNICAL + SENTIMENT: 10% weight (short-term timing, less critical)

=== CRITICAL ANTI-HALLUCINATION RULES ===
1. GROUNDING ON NUMBERS: You must ONLY cite the verified financial metrics provided above in GROUND TRUTH (e.g. Trailing P/E of {pe_str}, P/B of {pb_str}, Operating Margin of {opm_str}, ROE of {roe_str}, Revenue Growth of {rev_growth_str}, D/E of {de_str}, 1-Year Return of {ret_1y_str}). DO NOT invent, estimate, or hallucinate fictitious percentage growth rates or margins.
2. BETA FORMATTING: Beta is a unitless relative sensitivity coefficient ({beta_str}). NEVER format beta as a percentage (e.g. NEVER say "10% beta").
3. SECTOR ACCURACY: The company's actual sector is {actual_sector}. Do NOT describe it as a "{user_sector_bias}" company.
4. MACRO ACCURACY: Use the domestic macroeconomic numbers ({macro_benchmark}). Do not apply US inflation targets (2%) to Indian equities.
5. DEFINITIVE STANCE: Do NOT default to HOLD just because signals are mixed. Take a DEFINITIVE stance: BUY, HOLD, or SELL.
6. CROSS-CHECK: If any agent report contradicts the GROUND TRUTH numbers above, override with the GROUND TRUTH values.

Provide your FINAL RECOMMENDATION with:
1. Brief summary of key signals from each agent (strictly citing the verified ground truth data)
2. Conflict resolution (why this signal wins, considering user's {user_risk_tol} risk tolerance and {user_horizon} horizon)
3. Clear BUY/HOLD/SELL decision
4. Conviction level (0.0-1.0)
5. Top 2-3 risks to monitor
6. Portfolio fit (how this position aligns with user's {user_risk_tol} risk tolerance and {user_horizon} horizon)

End your response with EXACTLY this structured tag block at the very bottom:
FINAL_VERDICT: BUY | HOLD | SELL
CONVICTION: <float 0.0-1.0>
"""
        
        messages = [HumanMessage(content=synthesis_prompt)] + state.get("messages", [])
        result = await self.deep_llm.ainvoke(messages)
        
        final_rec = result.content if hasattr(result, "content") else str(result)
        
        print("\n" + "-"*80)
        print("[SUPERVISOR] FINAL RECOMMENDATION:")
        print("-"*80)
        print(final_rec)
        print("\n" + "="*80 + "\n")
        
        return {
            "final_recommendation": final_rec,
            "messages": [result],
        }

    async def _build(self):
        """Build the explicit phase-based workflow graph"""
        wf = StateGraph(QuantIntelState)
        
        # Add explicit phase nodes
        wf.add_node("phase1_parallel", self._phase1_parallel_agents)
        wf.add_node("phase2_risk", self._phase2_risk_agent)
        wf.add_node("phase3_supervisor", self._phase3_supervisor_decision)
        
        # Define the strict sequence: Phase 1 -> Phase 2 -> Phase 3
        wf.add_edge(START, "phase1_parallel")
        wf.add_edge("phase1_parallel", "phase2_risk")
        wf.add_edge("phase2_risk", "phase3_supervisor")
        wf.add_edge("phase3_supervisor", END)
        
        return wf.compile()

    async def run(self, ticker: str, trade_date: str = None, portfolio_context: dict = None) -> dict:
        resolved_date = str(trade_date or datetime.now().strftime("%Y-%m-%d"))
        initial_state = {
            "messages":             [HumanMessage(content=f"Recommend action for {ticker}")],
            "ticker":               ticker,
            "trade_date":           resolved_date,
            "portfolio_context":    portfolio_context or {},
            "ground_truth":         {},
            "fundamentals_report":  "",
            "sentiment_report":     "",
            "technical_report":     "",
            "risk_report":          "",
            "macro_report":         "",
            "final_recommendation": "",
            "sender":               "",
            "phase1_complete":      False,
            "phase2_complete":      False,
        }

        graph = await self._build()
        args = {
            "stream_mode": "values",
            "config": {"recursion_limit": self.config.get("max_recur_limit", 100)},
        }

        if self.debug:
            trace = []
            async for chunk in graph.astream(initial_state, **args):
                trace.append(chunk)
            return trace[-1]

        result = await graph.ainvoke(initial_state, **args)
        return result
