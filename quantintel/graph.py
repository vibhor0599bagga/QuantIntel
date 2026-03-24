"""
quantintel/graph.py
Wires all 6 agents into a LangGraph StateGraph.
Zero dependency on tradingagents package.

Execution order:
  1. Fundamentals  ─┐
  2. Sentiment      │  sequential, each with its own tool-loop
  3. Technical      │
  4. Risk           │  (reads 1-3 from state)
  5. Macro         ─┘
  6. Supervisor       (reads all 5 + portfolio context)
"""

from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END

from quantintel.config import DEFAULT_CONFIG, set_config
from quantintel.memory import FinancialSituationMemory
from quantintel.llm_clients import create_llm_client

from quantintel.agents.states import QuantIntelState
from quantintel.agents.agent_1_fundamentals import create_fundamentals_agent, get_fundamentals_tool_node
from quantintel.agents.agent_2_sentiment    import create_sentiment_agent,    get_sentiment_tool_node
from quantintel.agents.agent_3_technical    import create_technical_agent,    get_technical_tool_node
from quantintel.agents.agent_4_risk         import create_risk_agent,         get_risk_tool_node
from quantintel.agents.agent_5_macro        import create_macro_agent,        get_macro_tool_node
from quantintel.agents.agent_6_supervisor   import create_supervisor_agent


# ─── helpers ──────────────────────────────────────────────────────────────────

def _msg_clear_node():
    """Wipes messages between agents to keep context windows clean."""
    def clear(state):
        ops = [RemoveMessage(id=m.id) for m in state["messages"]]
        return {"messages": ops + [HumanMessage(content="Continue")]}
    return clear


def _should_continue(agent_key: str):
    """Route to tool node if last message has tool calls, else to message clear."""
    def router(state):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return f"tools_{agent_key}"
        return f"clear_{agent_key}"
    return router


# ─── main graph class ─────────────────────────────────────────────────────────

class QuantIntelGraph:
    """
    Complete self-contained QuantIntel pipeline.

    Usage:
        from quantintel.graph import QuantIntelGraph

        qi = QuantIntelGraph()
        result = qi.run(
            ticker            = "AAPL",
            trade_date        = "2026-03-23",
            portfolio_context = {
                "sector_exposure": "tech_heavy",
                "horizon":         "long_term",
                "risk_tolerance":  "moderate",
                "holdings":        ["MSFT", "GOOGL"],
            }
        )
        print(result["final_recommendation"])
    """

    def __init__(self, config: dict = None, debug: bool = False):
        self.debug  = debug
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        set_config(self.config)

        # LLMs
        deep_llm  = create_llm_client(
            provider = self.config["llm_provider"],
            model    = self.config["deep_think_llm"],
            base_url = self.config.get("backend_url"),
        ).get_llm()

        quick_llm = create_llm_client(
            provider = self.config["llm_provider"],
            model    = self.config["quick_think_llm"],
            base_url = self.config.get("backend_url"),
        ).get_llm()

        # Memory for supervisor
        self.memory = FinancialSituationMemory("supervisor_memory")

        # Build graph
        self.graph = self._build(quick_llm, deep_llm)

    def _build(self, quick_llm, deep_llm):
        wf = StateGraph(QuantIntelState)

        # Agent nodes
        agents = {
            "fundamentals": create_fundamentals_agent(quick_llm),
            "sentiment":    create_sentiment_agent(quick_llm),
            "technical":    create_technical_agent(quick_llm),
            "risk":         create_risk_agent(quick_llm),
            "macro":        create_macro_agent(quick_llm),
        }

        # Tool nodes
        tool_nodes = {
            "fundamentals": get_fundamentals_tool_node(),
            "sentiment":    get_sentiment_tool_node(),
            "technical":    get_technical_tool_node(),
            "risk":         get_risk_tool_node(),
            "macro":        get_macro_tool_node(),
        }

        # Supervisor uses deep LLM
        supervisor = create_supervisor_agent(deep_llm, self.memory)

        # Add all nodes
        ordered = ["fundamentals", "sentiment", "technical", "risk", "macro"]
        for key in ordered:
            wf.add_node(f"agent_{key}", agents[key])
            wf.add_node(f"tools_{key}", tool_nodes[key])
            wf.add_node(f"clear_{key}", _msg_clear_node())
        wf.add_node("supervisor", supervisor)

        # Edges: START → first agent
        wf.add_edge(START, "agent_fundamentals")

        # Edges: each agent → tool loop → clear → next
        for i, key in enumerate(ordered):
            wf.add_conditional_edges(
                f"agent_{key}",
                _should_continue(key),
                {f"tools_{key}": f"tools_{key}", f"clear_{key}": f"clear_{key}"},
            )
            wf.add_edge(f"tools_{key}", f"agent_{key}")  # loop back

            next_node = f"agent_{ordered[i+1]}" if i < len(ordered) - 1 else "supervisor"
            wf.add_edge(f"clear_{key}", next_node)

        wf.add_edge("supervisor", END)
        return wf.compile()

    def run(self, ticker: str, trade_date: str, portfolio_context: dict = None) -> dict:
        """
        Run the full QuantIntel pipeline.

        Returns a dict with keys:
            final_recommendation, fundamentals_report, sentiment_report,
            technical_report, risk_report, macro_report
        """
        initial_state = {
            "messages":             [HumanMessage(content=ticker)],
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

        args = {
            "stream_mode": "values",
            "config":      {"recursion_limit": self.config.get("max_recur_limit", 100)},
        }

        if self.debug:
            trace = []
            for chunk in self.graph.stream(initial_state, **args):
                if chunk.get("messages"):
                    chunk["messages"][-1].pretty_print()
                trace.append(chunk)
            return trace[-1]

        return self.graph.invoke(initial_state, **args)
