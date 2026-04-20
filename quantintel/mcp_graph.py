# quantintel/mcp_graph.py
import asyncio
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
    The Supervisor connects to the MCP Swarm Server and queries sub-agents as tools.
    """
    def __init__(self, session, mcp_tools, config: dict = None, debug: bool = False):
        self.debug  = debug
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        set_config(self.config)

        self.deep_llm = create_llm_client(
            provider = self.config["llm_provider"],
            model    = self.config["deep_think_llm"],
            base_url = self.config.get("backend_url"),
        ).get_llm()
        self.memory = FinancialSituationMemory("mcp_memory")
        self.session = session
        self.mcp_tools = mcp_tools

    async def _build(self):
        # Tools node
        tool_node = ToolNode(self.mcp_tools)
        
        # Bind the tools to the Deep LLM
        llm_with_tools = self.deep_llm.bind_tools(self.mcp_tools)
        
        async def supervisor_agent(state: QuantIntelState):
            # First, fetch Portfolio context via MCP Resource!
            try:
                portfolio = await get_mcp_resource(self.session, "portfolio://user/current")
            except Exception:
                portfolio = str(state.get("portfolio_context", {}))
                
            ticker = state["ticker"]
            trade_date = state["trade_date"]
            
            prompt = f"""
            You are the Supervisor Agent in QuantIntel.
            Instead of passively receiving agent reports, YOU actively coordinate them using MCP tools.
            Make sure to take maximum possible data for making the best decision. You can call tools to ask specific agents for their insights, and you can call them multiple times as needed.
            You must use your tools (`ask_fundamentals_agent`, `ask_sentiment_agent`, etc.) to gather data.
            
            TICKER: {ticker} | DATE: {trade_date}
            PORTFOLIO CONTEXT: {portfolio}
            
            1. You MUST call ALL 5 agent tools (`ask_fundamentals_agent`, `ask_sentiment_agent`, `ask_technical_agent`, `ask_risk_agent`, `ask_macro_agent`) before making a final recommendation to gather a complete 360-degree view.
            2. Synthesize their signals and resolve conflicts. 
               *CRITICAL*: Fundamental and Macro data must heavily outweigh retail/news Sentiment. Do not default to HOLD just because of mixed signals; take a definitive stance based on hard financial data and downside risk adjustments.
            3. Decide BUY | HOLD | SELL with extremely high conviction.
            
            Return your FINAL RECOMMENDATION string after checking tools.
            """
            messages = [HumanMessage(content=prompt)] + state["messages"]
            result = await llm_with_tools.ainvoke(messages)
            
            # Extract content if final
            report = result.content if not getattr(result, "tool_calls", None) else ""
            return {"messages": [result], "final_recommendation": report}

        def route_tools(state):
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        wf = StateGraph(QuantIntelState)
        wf.add_node("supervisor", supervisor_agent)
        wf.add_node("tools", tool_node)
        
        wf.add_edge(START, "supervisor")
        wf.add_conditional_edges("supervisor", route_tools, {"tools": "tools", END: END})
        wf.add_edge("tools", "supervisor")
        
        return wf.compile()

    async def run(self, ticker: str, trade_date: str, portfolio_context: dict = None) -> dict:
        initial_state = {
            "messages":             [HumanMessage(content=f"Recommend action for {ticker}")],
            "ticker":               ticker,
            "trade_date":           str(trade_date),
            "portfolio_context":    portfolio_context or {},
            "fundamentals_report":  "",
            "sentiment_report":     "",
            "technical_report":     "",
            "risk_report":          "",
            "macro_report":         "",
            "final_recommendation": "",
            "sender":               "",
        }

        graph = await self._build()
        args = {
            "stream_mode": "values",
            "config": {"recursion_limit": self.config.get("max_recur_limit", 100)},
        }

        if self.debug:
            trace = []
            async for chunk in graph.astream(initial_state, **args):
                if chunk.get("messages"):
                    chunk["messages"][-1].pretty_print()
                trace.append(chunk)
            return trace[-1]

        result = await graph.ainvoke(initial_state, **args)
        return result
