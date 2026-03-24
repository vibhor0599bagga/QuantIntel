"""
quantintel/tools.py

All LangChain @tool wrappers in one file.
Agents import directly from here — no tradingagents dependency.
"""

from langchain_core.tools import tool
from typing import Annotated

from quantintel.dataflows.y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    get_fundamentals       as _get_fundamentals,
    get_balance_sheet      as _get_balance_sheet,
    get_cashflow           as _get_cashflow,
    get_income_statement   as _get_income_statement,
    get_insider_transactions as _get_insider_transactions,
)
from quantintel.dataflows.yfinance_news import (
    get_news_yfinance,
    get_global_news_yfinance,
)


# ─── stock price ──────────────────────────────────────────────────────────────

@tool
def get_stock_data(
    symbol:     Annotated[str, "ticker symbol, e.g. AAPL"],
    start_date: Annotated[str, "start date yyyy-mm-dd"],
    end_date:   Annotated[str, "end date   yyyy-mm-dd"],
) -> str:
    """Retrieve OHLCV stock price data for a given ticker."""
    return get_YFin_data_online(symbol, start_date, end_date)


# ─── technical indicators ─────────────────────────────────────────────────────

@tool
def get_indicators(
    symbol:         Annotated[str, "ticker symbol"],
    indicator:      Annotated[str, "indicator name, e.g. rsi, macd, boll_ub"],
    curr_date:      Annotated[str, "current trading date YYYY-mm-dd"],
    look_back_days: Annotated[int, "days to look back"] = 30,
) -> str:
    """Retrieve a single technical indicator. Call once per indicator name."""
    indicators = [i.strip() for i in indicator.split(",") if i.strip()]
    if len(indicators) > 1:
        return "\n\n".join(
            get_stock_stats_indicators_window(symbol, ind, curr_date, look_back_days)
            for ind in indicators
        )
    return get_stock_stats_indicators_window(symbol, indicator.strip(), curr_date, look_back_days)


# ─── fundamentals ─────────────────────────────────────────────────────────────

@tool
def get_fundamentals(
    ticker:    Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date yyyy-mm-dd"] = None,
) -> str:
    """Retrieve comprehensive fundamental data: P/E, ROE, EPS, market cap, margins."""
    return _get_fundamentals(ticker, curr_date)


@tool
def get_balance_sheet(
    ticker:    Annotated[str, "ticker symbol"],
    freq:      Annotated[str, "annual or quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date yyyy-mm-dd"] = None,
) -> str:
    """Retrieve balance sheet data (assets, liabilities, equity)."""
    return _get_balance_sheet(ticker, freq, curr_date)


@tool
def get_cashflow(
    ticker:    Annotated[str, "ticker symbol"],
    freq:      Annotated[str, "annual or quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date yyyy-mm-dd"] = None,
) -> str:
    """Retrieve cash flow statement (operating, investing, financing)."""
    return _get_cashflow(ticker, freq, curr_date)


@tool
def get_income_statement(
    ticker:    Annotated[str, "ticker symbol"],
    freq:      Annotated[str, "annual or quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date yyyy-mm-dd"] = None,
) -> str:
    """Retrieve income statement (revenue, gross profit, net income, EBITDA)."""
    return _get_income_statement(ticker, freq, curr_date)


@tool
def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Retrieve recent insider buy/sell transactions."""
    return _get_insider_transactions(ticker)


# ─── news ─────────────────────────────────────────────────────────────────────

@tool
def get_news(
    ticker:     Annotated[str, "ticker symbol or search keyword"],
    start_date: Annotated[str, "start date yyyy-mm-dd"],
    end_date:   Annotated[str, "end date   yyyy-mm-dd"],
) -> str:
    """Retrieve news articles for a stock or keyword query."""
    return get_news_yfinance(ticker, start_date, end_date)


@tool
def get_global_news(
    curr_date:      Annotated[str, "current date yyyy-mm-dd"],
    look_back_days: Annotated[int, "days to look back"] = 7,
    limit:          Annotated[int, "max articles to return"] = 10,
) -> str:
    """Retrieve broad macroeconomic and global market news."""
    return get_global_news_yfinance(curr_date, look_back_days, limit)
