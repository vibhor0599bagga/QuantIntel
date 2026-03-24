"""
quantintel/dataflows/y_finance.py
Copied from TradingAgents — internal imports updated.
"""

from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import os

from .stockstats_utils import StockstatsUtils, _clean_dataframe, yf_retry


# ─── OHLCV ────────────────────────────────────────────────────────────────────

def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    ticker = yf.Ticker(symbol.upper())
    data   = yf_retry(lambda: ticker.history(start=start_date, end=end_date))

    if data.empty:
        return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        if col in data.columns:
            data[col] = data[col].round(2)

    header  = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + data.to_csv()


# ─── TECHNICAL INDICATORS ─────────────────────────────────────────────────────

def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator name"],
    curr_date: Annotated[str, "The current trading date YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:

    best_ind_params = {
        "close_50_sma":  "50 SMA: Medium-term trend. Dynamic support/resistance.",
        "close_200_sma": "200 SMA: Long-term trend benchmark. Golden/death cross.",
        "close_10_ema":  "10 EMA: Short-term responsive average. Entry signals.",
        "macd":          "MACD: Momentum via EMA differences. Crossover signals.",
        "macds":         "MACD Signal: EMA of MACD line. Crossover trigger.",
        "macdh":         "MACD Histogram: Gap between MACD and signal. Momentum strength.",
        "rsi":           "RSI: Overbought (>70) / oversold (<30) momentum.",
        "boll":          "Bollinger Middle (20 SMA): Dynamic price benchmark.",
        "boll_ub":       "Bollinger Upper Band: Overbought / breakout zone.",
        "boll_lb":       "Bollinger Lower Band: Oversold / reversal zone.",
        "atr":           "ATR: Raw volatility measure. Stop-loss sizing.",
        "vwma":          "VWMA: Volume-weighted MA. Confirms trend with volume.",
        "mfi":           "MFI: Price+volume momentum. Overbought (>80) / oversold (<20).",
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator '{indicator}' not supported. Choose from: {list(best_ind_params.keys())}"
        )

    end_date    = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before      = curr_date_dt - relativedelta(days=look_back_days)

    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator, curr_date)
        current_dt = curr_date_dt
        date_values = []
        while current_dt >= before:
            date_str = current_dt.strftime("%Y-%m-%d")
            val = indicator_data.get(date_str, "N/A: Not a trading day (weekend or holiday)")
            date_values.append((date_str, val))
            current_dt -= relativedelta(days=1)

        ind_string = "".join(f"{d}: {v}\n" for d, v in date_values)

    except Exception as e:
        print(f"Bulk stockstats error: {e} — falling back to per-day fetch")
        ind_string = ""
        current_dt = curr_date_dt
        while current_dt >= before:
            val = StockstatsUtils.get_stock_stats(symbol, indicator, current_dt.strftime("%Y-%m-%d"))
            ind_string += f"{current_dt.strftime('%Y-%m-%d')}: {val}\n"
            current_dt -= relativedelta(days=1)

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "")
    )


def _get_stock_stats_bulk(symbol, indicator, curr_date) -> dict:
    from quantintel.config import get_config
    import pandas as pd
    from stockstats import wrap

    config = get_config()
    today_date   = pd.Timestamp.today()
    start_date   = today_date - pd.DateOffset(years=15)
    start_str    = start_date.strftime("%Y-%m-%d")
    end_str      = today_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{symbol}-YFin-data-{start_str}-{end_str}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip")
    else:
        data = yf_retry(lambda: yf.download(
            symbol, start=start_str, end=end_str,
            multi_level_index=False, progress=False, auto_adjust=True,
        ))
        data = data.reset_index()
        data.to_csv(data_file, index=False)

    data = _clean_dataframe(data)
    df   = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df[indicator]  # trigger calculation

    result = {}
    for _, row in df.iterrows():
        import pandas as _pd
        val = row[indicator]
        result[row["Date"]] = "N/A" if _pd.isna(val) else str(val)
    return result


# ─── FUNDAMENTALS ─────────────────────────────────────────────────────────────

def get_fundamentals(ticker, curr_date=None):
    try:
        t    = yf.Ticker(ticker.upper())
        info = yf_retry(lambda: t.info)
        if not info:
            return f"No fundamentals data for '{ticker}'"

        fields = [
            ("Name",             info.get("longName")),
            ("Sector",           info.get("sector")),
            ("Industry",         info.get("industry")),
            ("Market Cap",       info.get("marketCap")),
            ("PE Ratio (TTM)",   info.get("trailingPE")),
            ("Forward PE",       info.get("forwardPE")),
            ("PEG Ratio",        info.get("pegRatio")),
            ("Price to Book",    info.get("priceToBook")),
            ("EPS (TTM)",        info.get("trailingEps")),
            ("Forward EPS",      info.get("forwardEps")),
            ("Dividend Yield",   info.get("dividendYield")),
            ("Beta",             info.get("beta")),
            ("52W High",         info.get("fiftyTwoWeekHigh")),
            ("52W Low",          info.get("fiftyTwoWeekLow")),
            ("Revenue (TTM)",    info.get("totalRevenue")),
            ("Gross Profit",     info.get("grossProfits")),
            ("EBITDA",           info.get("ebitda")),
            ("Net Income",       info.get("netIncomeToCommon")),
            ("Profit Margin",    info.get("profitMargins")),
            ("Operating Margin", info.get("operatingMargins")),
            ("ROE",              info.get("returnOnEquity")),
            ("ROA",              info.get("returnOnAssets")),
            ("Debt/Equity",      info.get("debtToEquity")),
            ("Current Ratio",    info.get("currentRatio")),
            ("Free Cash Flow",   info.get("freeCashflow")),
        ]

        lines  = [f"{label}: {val}" for label, val in fields if val is not None]
        header = f"# Fundamentals for {ticker.upper()}\n"
        header += f"# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + "\n".join(lines)

    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {e}"


def get_balance_sheet(ticker, freq="quarterly", curr_date=None):
    try:
        t = yf.Ticker(ticker.upper())
        data = yf_retry(lambda: t.quarterly_balance_sheet if freq.lower() == "quarterly" else t.balance_sheet)
        if data.empty:
            return f"No balance sheet data for '{ticker}'"
        header = f"# Balance Sheet for {ticker.upper()} ({freq})\n# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + data.to_csv()
    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {e}"


def get_cashflow(ticker, freq="quarterly", curr_date=None):
    try:
        t = yf.Ticker(ticker.upper())
        data = yf_retry(lambda: t.quarterly_cashflow if freq.lower() == "quarterly" else t.cashflow)
        if data.empty:
            return f"No cash flow data for '{ticker}'"
        header = f"# Cash Flow for {ticker.upper()} ({freq})\n# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + data.to_csv()
    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {e}"


def get_income_statement(ticker, freq="quarterly", curr_date=None):
    try:
        t = yf.Ticker(ticker.upper())
        data = yf_retry(lambda: t.quarterly_income_stmt if freq.lower() == "quarterly" else t.income_stmt)
        if data.empty:
            return f"No income statement data for '{ticker}'"
        header = f"# Income Statement for {ticker.upper()} ({freq})\n# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + data.to_csv()
    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {e}"


def get_insider_transactions(ticker):
    try:
        t    = yf.Ticker(ticker.upper())
        data = yf_retry(lambda: t.insider_transactions)
        if data is None or data.empty:
            return f"No insider transactions for '{ticker}'"
        header = f"# Insider Transactions for {ticker.upper()}\n# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + data.to_csv()
    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {e}"
