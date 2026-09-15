from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date


class PortfolioContext(BaseModel):
    sector_exposure: Optional[str] = Field(default="diversified", description="e.g. tech_heavy, banking_focused, diversified")
    horizon: Optional[str] = Field(default="medium_term", description="short_term, medium_term, long_term")
    risk_tolerance: Optional[str] = Field(default="moderate", description="conservative, moderate, aggressive")
    holdings: Optional[List[str]] = Field(default_factory=list, description="Existing stock tickers in portfolio")


class AnalysisRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL, MSFT, HDFCBANK)")
    trade_date: Optional[str] = Field(default=None, description="Analysis date in YYYY-MM-DD format. Defaults to today.")
    portfolio_context: Optional[PortfolioContext] = Field(default=None, description="User portfolio context")
    config_overrides: Optional[Dict[str, Any]] = Field(default=None, description="LLM settings overrides (e.g. llm_provider, models)")


class AnalysisResponse(BaseModel):
    status: str = "success"
    ticker: str
    trade_date: str
    fundamentals_report: str
    sentiment_report: str
    technical_report: str
    macro_report: str
    risk_report: str
    final_recommendation: str


class HealthResponse(BaseModel):
    status: str
    version: str
    mcp_server: str
