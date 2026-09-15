import os
import sys
import json
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

from quantintel.config import DEFAULT_CONFIG, get_config, set_config
from quantintel.mcp_graph import McpQuantIntelGraph
from quantintel.api.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    HealthResponse,
    PortfolioContext,
)

# Global variables for session and transport cleanup
_transport_ctx = None
_session_ctx = None
_mcp_session = None
_mcp_tools = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Manager:
    Starts the FastMCP agent swarm server as a background stdio process on startup,
    initializes the MCP ClientSession and loads tools, and cleans up on shutdown.
    """
    global _transport_ctx, _session_ctx, _mcp_session, _mcp_tools
    print("[QuantIntel API] Initializing FastMCP Swarm Server...")

    python_cmd = os.path.abspath("venv/Scripts/python.exe")
    if not os.path.exists(python_cmd):
        python_cmd = sys.executable

    server_params = StdioServerParameters(
        command=python_cmd,
        args=["-m", "quantintel.mcp_servers.agent_swarm_server"],
        env={**os.environ, "PYTHONPATH": os.path.abspath(".")}
    )

    try:
        _transport_ctx = stdio_client(server_params)
        read, write = await _transport_ctx.__aenter__()
        _session_ctx = ClientSession(read, write)
        _mcp_session = await _session_ctx.__aenter__()
        await _mcp_session.initialize()

        _mcp_tools = await load_mcp_tools(_mcp_session)
        app.state.mcp_session = _mcp_session
        app.state.mcp_tools = _mcp_tools
        print(f"[QuantIntel API] FastMCP Swarm Server connected successfully with {len(_mcp_tools)} tools loaded.")
    except Exception as e:
        print(f"[QuantIntel API] Failed to start MCP server: {e}")
        app.state.mcp_session = None
        app.state.mcp_tools = []

    yield

    # Shutdown / Cleanup
    print("[QuantIntel API] Shutting down FastMCP Swarm Server connection...")
    if _session_ctx:
        try:
            await _session_ctx.__aexit__(None, None, None)
        except Exception:
            pass
    if _transport_ctx:
        try:
            await _transport_ctx.__aexit__(None, None, None)
        except Exception:
            pass
    print("[QuantIntel API] QuantIntel API shutdown complete.")



app = FastAPI(
    title="QuantIntel API",
    description="Multi-Agent Quantitative Financial Analysis Engine powered by FastMCP and LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    mcp_status = "healthy" if app.state.mcp_session and app.state.mcp_tools else "degraded"
    return HealthResponse(
        status="ok",
        version="1.0.0",
        mcp_server=mcp_status
    )


@app.post("/api/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_stock(req: AnalysisRequest):
    """
    Run full multi-agent quantitative analysis for a stock ticker.
    Returns complete JSON with reports from all 5 agents and the final recommendation.
    """
    if not app.state.mcp_session or not app.state.mcp_tools:
        raise HTTPException(status_code=503, detail="MCP Swarm server is not available.")

    trade_date = req.trade_date or datetime.now().strftime("%Y-%m-%d")
    portfolio_dict = req.portfolio_context.dict() if req.portfolio_context else {
        "sector_exposure": "diversified",
        "horizon": "medium_term",
        "risk_tolerance": "moderate",
        "holdings": []
    }

    # Merge configuration overrides if provided
    config = DEFAULT_CONFIG.copy()
    if req.config_overrides:
        config.update(req.config_overrides)

    try:
        qi = McpQuantIntelGraph(app.state.mcp_session, app.state.mcp_tools, config=config, debug=False)
        result = await qi.run(
            ticker=req.ticker.upper(),
            trade_date=trade_date,
            portfolio_context=portfolio_dict
        )

        return AnalysisResponse(
            status="success",
            ticker=req.ticker.upper(),
            trade_date=trade_date,
            fundamentals_report=result.get("fundamentals_report", ""),
            sentiment_report=result.get("sentiment_report", ""),
            technical_report=result.get("technical_report", ""),
            macro_report=result.get("macro_report", ""),
            risk_report=result.get("risk_report", ""),
            final_recommendation=result.get("final_recommendation", "")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/analyze/stream", tags=["Analysis"])
async def stream_stock_analysis(req: AnalysisRequest):
    """
    Stream live multi-agent quantitative analysis progress using Server-Sent Events (SSE).
    Clients receive real-time updates as Phase 1 parallel agents complete, Phase 2 risk runs, and Phase 3 finishes.
    """
    if not app.state.mcp_session or not app.state.mcp_tools:
        raise HTTPException(status_code=503, detail="MCP Swarm server is not available.")

    trade_date = req.trade_date or datetime.now().strftime("%Y-%m-%d")
    portfolio_dict = req.portfolio_context.dict() if req.portfolio_context else {
        "sector_exposure": "diversified",
        "horizon": "medium_term",
        "risk_tolerance": "moderate",
        "holdings": []
    }

    config = DEFAULT_CONFIG.copy()
    if req.config_overrides:
        config.update(req.config_overrides)

    async def event_generator() -> AsyncGenerator[dict, None]:
        yield {
            "event": "start",
            "data": json.dumps({"ticker": req.ticker.upper(), "trade_date": trade_date, "message": "Analysis started"})
        }

        qi = McpQuantIntelGraph(app.state.mcp_session, app.state.mcp_tools, config=config, debug=True)
        
        initial_state = {
            "messages": [],
            "ticker": req.ticker.upper(),
            "trade_date": trade_date,
            "portfolio_context": portfolio_dict,
            "fundamentals_report": "",
            "sentiment_report": "",
            "technical_report": "",
            "risk_report": "",
            "macro_report": "",
            "final_recommendation": "",
            "sender": "",
            "phase1_complete": False,
            "phase2_complete": False,
        }

        try:
            graph = await qi._build()
            async for chunk in graph.astream(initial_state, config={"recursion_limit": 100}):
                for node_name, node_state in chunk.items():
                    if node_name == "phase1_parallel":
                        yield {
                            "event": "phase1_complete",
                            "data": json.dumps({
                                "fundamentals_report": node_state.get("fundamentals_report"),
                                "sentiment_report": node_state.get("sentiment_report"),
                                "technical_report": node_state.get("technical_report"),
                                "macro_report": node_state.get("macro_report"),
                            })
                        }
                    elif node_name == "phase2_risk":
                        yield {
                            "event": "phase2_complete",
                            "data": json.dumps({
                                "risk_report": node_state.get("risk_report")
                            })
                        }
                    elif node_name == "phase3_supervisor":
                        yield {
                            "event": "final_recommendation",
                            "data": json.dumps({
                                "final_recommendation": node_state.get("final_recommendation")
                            })
                        }

            yield {
                "event": "complete",
                "data": json.dumps({"status": "success", "message": "Analysis pipeline finished"})
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_generator())
