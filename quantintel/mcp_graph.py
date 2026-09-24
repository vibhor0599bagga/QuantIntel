# quantintel/mcp_graph.py
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

        self.api_key = self.config.get("api_key") or self.config.get("openrouter_api_key") or ""

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

    async def _execute_tool_with_timeout(self, tool_name: str, timeout: float = 60.0, **kwargs) -> str:
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
        
        print("\n" + "="*80)
        print("[PHASE 1] PARALLEL DATA GATHERING (All 4 agents run concurrently)")
        print("="*80)
        
        # Execute all 4 agents CONCURRENTLY with 60s (1 min) timeouts using asyncio.gather()
        tasks = [
            self._execute_tool_with_timeout("ask_fundamentals_agent", timeout=60.0, ticker=ticker, trade_date=trade_date, api_key=self.api_key),
            self._execute_tool_with_timeout("ask_sentiment_agent", timeout=60.0, ticker=ticker, trade_date=trade_date, api_key=self.api_key),
            self._execute_tool_with_timeout("ask_technical_agent", timeout=60.0, ticker=ticker, trade_date=trade_date, api_key=self.api_key),
            self._execute_tool_with_timeout("ask_macro_agent", timeout=60.0, ticker=ticker, trade_date=trade_date, api_key=self.api_key),
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
                risk_report = await self._execute_tool_with_timeout("ask_risk_agent", timeout=60.0, **kwargs)

            
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
        
        ticker = state["ticker"]
        trade_date = state["trade_date"]
        
        try:
            portfolio = await get_mcp_resource(self.session, "portfolio://user/current")
        except Exception:
            portfolio = str(state.get("portfolio_context", {}))
        
        synthesis_prompt = f"""
You are the Supervisor Agent in QuantIntel. All 5 agents have completed their analysis.

TICKER: {ticker} | DATE: {trade_date}
PORTFOLIO CONTEXT: {portfolio}

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

*CRITICAL*: Do NOT default to HOLD just because signals are mixed. Take a DEFINITIVE stance: BUY, HOLD, or SELL.

Provide your FINAL RECOMMENDATION with:
1. Brief summary of key signals from each agent
2. Conflict resolution (why this signal wins)
3. Clear BUY/HOLD/SELL decision
4. Conviction level (0.0-1.0)
5. Top 2-3 risks to monitor
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
