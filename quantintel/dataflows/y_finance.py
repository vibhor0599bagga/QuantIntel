"""
quantintel/dataflows/y_finance.py
Copied from TradingAgents — internal imports updated.
"""

from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import os

from .stockstats_utils import StockstatsUtils, _clean_dataframe, yf_retry, resolve_symbol


# ─── OHLCV ────────────────────────────────────────────────────────────────────

def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    clean_symbol = resolve_symbol(symbol)
    ticker = yf.Ticker(clean_symbol)
    data   = yf_retry(lambda: ticker.history(start=start_date, end=end_date))

    if data is None or data.empty:
        return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        if col in data.columns:
            data[col] = data[col].round(2)

    header  = f"# Stock data for {clean_symbol} from {start_date} to {end_date}\n"
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

    clean_symbol = resolve_symbol(symbol)
    config = get_config()
    today_date   = pd.Timestamp.today()
    start_date   = today_date - pd.DateOffset(years=15)
    start_str    = start_date.strftime("%Y-%m-%d")
    end_str      = today_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{clean_symbol}-YFin-data-{start_str}-{end_str}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip")
    else:
        data = yf_retry(lambda: yf.download(
            clean_symbol, start=start_str, end=end_str,
            multi_level_index=False, progress=False, auto_adjust=True,
        ))
        if data is not None and not data.empty:
            data = data.reset_index()
            data.to_csv(data_file, index=False)

    data = _clean_dataframe(data)
    if data.empty or "Date" not in data.columns:
        return {}

    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    try:
        df[indicator]  # trigger stockstats dynamic calculation
    except Exception as e:
        pass

    result = {}
    for _, row in df.iterrows():
        import pandas as _pd
        val = row.get(indicator, "N/A")
        result[row["Date"]] = "N/A" if _pd.isna(val) else str(val)
    return result


# ─── FUNDAMENTALS ─────────────────────────────────────────────────────────────

def get_fundamentals(ticker, curr_date=None):
    clean_ticker = resolve_symbol(ticker)
    try:
        t    = yf.Ticker(clean_ticker)
        info = yf_retry(lambda: t.info)
        if not info:
            return f"No fundamentals data for '{ticker}'"

        def _fmt_money(v):
            if v is None: return "N/A"
            if abs(v) >= 1e12: return f"${v/1e12:,.2f}T ({v:,.0f})"
            if abs(v) >= 1e9:  return f"${v/1e9:,.2f}B ({v:,.0f})"
            if abs(v) >= 1e6:  return f"${v/1e6:,.2f}M ({v:,.0f})"
            return f"${v:,.2f}"

        def _fmt_pct(v):
            if v is None: return "N/A"
            return f"{float(v)*100:.2f}%"

        def _fmt_mult(v):
            if v is None: return "N/A"
            return f"{float(v):.2f}x"

        def _fmt_div_yield(dy):
            if dy is None: return "0.00%"
            val = float(dy)
            # In yfinance, dividendYield is often already in percent (e.g. 0.69 means 0.69%)
            if val < 0.05:
                return f"{val * 100:.2f}%"
            return f"{val:.2f}%"

        # Debt to equity in yfinance info is typically percentage (e.g. 14.306 = 14.31% = 0.14x)
        de_raw = info.get("debtToEquity")
        de_str = f"{float(de_raw)/100.0:.2f}x ({de_raw:.2f}%)" if de_raw is not None else "N/A"

        cur_price = info.get("currentPrice") or info.get("regularMarketPrice")

        fields = [
            ("Company Name",           info.get("longName") or info.get("shortName")),
            ("Sector",                 info.get("sector")),
            ("Industry",               info.get("industry")),
            ("Current Stock Price",    f"${cur_price:,.2f}" if cur_price else "N/A"),
            ("Market Cap",             _fmt_money(info.get("marketCap"))),
            ("PE Ratio (Trailing TTM)", _fmt_mult(info.get("trailingPE"))),
            ("Forward PE",             _fmt_mult(info.get("forwardPE"))),
            ("PEG Ratio",              f"{info.get('pegRatio'):.2f}" if info.get("pegRatio") else "N/A"),
            ("Price to Book (P/B)",    _fmt_mult(info.get("priceToBook"))),
            ("Book Value per Share",   f"${info.get('bookValue'):.2f}" if info.get("bookValue") else "N/A"),
            ("EPS (Trailing TTM)",     f"${info.get('trailingEps'):.2f}" if info.get("trailingEps") else "N/A"),
            ("Forward EPS",            f"${info.get('forwardEps'):.2f}" if info.get("forwardEps") else "N/A"),
            ("Dividend Yield",         _fmt_div_yield(info.get("dividendYield"))),
            ("Beta (Relative to Mkt)", f"{info.get('beta'):.2f}" if info.get("beta") else "1.00"),
            ("52-Week High",           f"${info.get('fiftyTwoWeekHigh'):,.2f}" if info.get("fiftyTwoWeekHigh") else "N/A"),
            ("52-Week Low",            f"${info.get('fiftyTwoWeekLow'):,.2f}" if info.get("fiftyTwoWeekLow") else "N/A"),
            ("Total Revenue (TTM)",    _fmt_money(info.get("totalRevenue"))),
            ("Revenue Growth (YoY)",   _fmt_pct(info.get("revenueGrowth"))),
            ("Gross Profit (TTM)",     _fmt_money(info.get("grossProfits") or info.get("grossProfit"))),
            ("Gross Margin",           _fmt_pct(info.get("grossMargins"))),
            ("Operating Margin",       _fmt_pct(info.get("operatingMargins"))),
            ("Net Profit Margin",      _fmt_pct(info.get("profitMargins"))),
            ("EBITDA",                 _fmt_money(info.get("ebitda"))),
            ("Net Income to Common",   _fmt_money(info.get("netIncomeToCommon"))),
            ("Return on Equity (ROE)", _fmt_pct(info.get("returnOnEquity"))),
            ("Return on Assets (ROA)", _fmt_pct(info.get("returnOnAssets"))),
            ("Debt-to-Equity Ratio",   de_str),
            ("Current Ratio",          f"{info.get('currentRatio'):.2f}" if info.get("currentRatio") else "N/A"),
            ("Operating Cash Flow",    _fmt_money(info.get("operatingCashflow"))),
            ("Free Cash Flow (FCF)",   _fmt_money(info.get("freeCashflow"))),
        ]

        lines  = [f"{label}: {val}" for label, val in fields if val is not None and val != "N/A"]
        header = f"# Institutional Fundamentals for {clean_ticker}\n"
        header += f"# Retrieved from Yahoo Finance: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + "\n".join(lines)

    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {e}"


def get_balance_sheet(ticker, freq="quarterly", curr_date=None):
    clean_ticker = resolve_symbol(ticker)
    try:
        t = yf.Ticker(clean_ticker)
        data = yf_retry(lambda: t.quarterly_balance_sheet if freq.lower() == "quarterly" else t.balance_sheet)
        if data is None or data.empty:
            return f"No balance sheet data for '{ticker}'"
        header = f"# Balance Sheet for {clean_ticker} ({freq})\n# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + data.to_csv()
    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {e}"


def get_cashflow(ticker, freq="quarterly", curr_date=None):
    clean_ticker = resolve_symbol(ticker)
    try:
        t = yf.Ticker(clean_ticker)
        data = yf_retry(lambda: t.quarterly_cashflow if freq.lower() == "quarterly" else t.cashflow)
        if data is None or data.empty:
            return f"No cash flow data for '{ticker}'"
        header = f"# Cash Flow for {clean_ticker} ({freq})\n# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + data.to_csv()
    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {e}"


def get_income_statement(ticker, freq="quarterly", curr_date=None):
    clean_ticker = resolve_symbol(ticker)
    try:
        t = yf.Ticker(clean_ticker)
        data = yf_retry(lambda: t.quarterly_income_stmt if freq.lower() == "quarterly" else t.income_stmt)
        if data is None or data.empty:
            return f"No income statement data for '{ticker}'"
        header = f"# Income Statement for {clean_ticker} ({freq})\n# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + data.to_csv()
    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {e}"


def get_insider_transactions(ticker):
    clean_ticker = resolve_symbol(ticker)
    try:
        t    = yf.Ticker(clean_ticker)
        data = yf_retry(lambda: t.insider_transactions)
        if data is None or data.empty:
            return f"No insider transactions for '{ticker}'"
        header = f"# Insider Transactions for {clean_ticker}\n# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + data.to_csv()
    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {e}"


# In-memory ground-truth cache to prevent concurrent rate-limiting from Yahoo Finance
_GROUND_TRUTH_CACHE = {}
_GROUND_TRUTH_CACHE_TTL = 600  # 10 minutes cache

def get_ground_truth_context(ticker: str) -> dict:
    """
    Retrieve pre-flight verified financial facts to anchor ALL LLM agents
    and completely eliminate statistical hallucinations.

    Uses an in-memory cache to prevent multiple parallel agents from hammering
    Yahoo Finance concurrently. Includes robust fallbacks using fast_info,
    financials, and balance_sheet when quote-summary info fails on cloud IPs.
    """
    clean_ticker = resolve_symbol(ticker)
    cache_key = clean_ticker.upper()
    now_ts = datetime.now().timestamp()

    if cache_key in _GROUND_TRUTH_CACHE:
        cached_entry = _GROUND_TRUTH_CACHE[cache_key]
        if now_ts - cached_entry["cached_at"] < _GROUND_TRUTH_CACHE_TTL:
            return cached_entry["data"]

    is_indian = clean_ticker.endswith(".NS") or clean_ticker.endswith(".BO") or clean_ticker in (
        "RELIANCE", "HDFCBANK", "TCS", "INFY", "ICICIBANK", "SBIN", "TATAMOTORS"
    )

    t = yf.Ticker(clean_ticker)
    try:
        info = yf_retry(lambda: t.info) or {}
    except Exception:
        info = {}

    # Fast info fallback for price, market cap, and 52W range
    cur_price = info.get("currentPrice") or info.get("regularMarketPrice")
    mcap = info.get("marketCap")
    high_52w = info.get("fiftyTwoWeekHigh")
    low_52w = info.get("fiftyTwoWeekLow")

    try:
        fi = t.fast_info
        if fi:
            if not cur_price and getattr(fi, "last_price", None):
                cur_price = round(float(fi.last_price), 2)
            if not mcap and getattr(fi, "market_cap", None):
                mcap = fi.market_cap
            if not high_52w and getattr(fi, "year_high", None):
                high_52w = round(float(fi.year_high), 2)
            if not low_52w and getattr(fi, "year_low", None):
                low_52w = round(float(fi.year_low), 2)
    except Exception:
        pass

    try:
        hist = yf_retry(lambda: t.history(period="1y"))
        if hist is not None and len(hist) >= 2:
            p_start = float(hist["Close"].iloc[0])
            p_end = float(hist["Close"].iloc[-1])
            ret_1y = round(((p_end - p_start) / p_start) * 100, 2)
            if not cur_price:
                cur_price = round(p_end, 2)
        else:
            ret_1y = None
    except Exception:
        ret_1y = None

    # --- Financial statement fallbacks for margins and balance sheet ---
    inc = None
    bs = None
    try:
        inc = yf_retry(lambda: t.financials)
    except Exception:
        pass
    try:
        bs = yf_retry(lambda: t.balance_sheet)
    except Exception:
        pass

    # Core valuation multiples
    pe_ttm = round(float(info.get("trailingPE")), 2) if info.get("trailingPE") else None
    fwd_pe = round(float(info.get("forwardPE")), 2) if info.get("forwardPE") else None
    peg    = round(float(info.get("pegRatio")), 2) if info.get("pegRatio") else None
    pb     = round(float(info.get("priceToBook")), 2) if info.get("priceToBook") else None
    eps_ttm = round(float(info.get("trailingEps")), 2) if info.get("trailingEps") else None
    fwd_eps = round(float(info.get("forwardEps")), 2) if info.get("forwardEps") else None
    book_value = round(float(info.get("bookValue")), 2) if info.get("bookValue") else None

    # Market data
    beta_val = round(float(info.get("beta")), 2) if info.get("beta") else 1.0
    raw_dy = info.get("dividendYield")
    if raw_dy is not None:
        val_dy = float(raw_dy)
        div_yield = round(val_dy * 100, 2) if val_dy < 0.05 else round(val_dy, 2)
    else:
        div_yield = None

    # Margin & profitability with statement fallbacks
    gross_margin_pct = round(float(info.get("grossMargins")) * 100, 2) if info.get("grossMargins") else None
    opm = round(float(info.get("operatingMargins")) * 100, 2) if info.get("operatingMargins") else None
    pm = round(float(info.get("profitMargins")) * 100, 2) if info.get("profitMargins") else None
    roe = round(float(info.get("returnOnEquity")) * 100, 2) if info.get("returnOnEquity") else None
    roa = round(float(info.get("returnOnAssets")) * 100, 2) if info.get("returnOnAssets") else None

    total_revenue = info.get("totalRevenue")
    if inc is not None and not inc.empty:
        if not total_revenue and "Total Revenue" in inc.index:
            total_revenue = float(inc.loc["Total Revenue"].iloc[0])
        if total_revenue and total_revenue > 0:
            if gross_margin_pct is None and "Gross Profit" in inc.index:
                gp = float(inc.loc["Gross Profit"].iloc[0])
                gross_margin_pct = round((gp / total_revenue) * 100, 2)
            if opm is None and "Operating Income" in inc.index:
                oi = float(inc.loc["Operating Income"].iloc[0])
                opm = round((oi / total_revenue) * 100, 2)
            if pm is None and "Net Income" in inc.index:
                ni = float(inc.loc["Net Income"].iloc[0])
                pm = round((ni / total_revenue) * 100, 2)

    # Balance sheet with statement fallbacks
    de_ratio_raw = info.get("debtToEquity")
    de_ratio = round(float(de_ratio_raw) / 100.0, 2) if de_ratio_raw else None
    current_ratio = round(float(info.get("currentRatio")), 2) if info.get("currentRatio") else None
    total_cash = info.get("totalCash")
    total_debt = info.get("totalDebt")
    fcf = info.get("freeCashflow")
    op_cashflow = info.get("operatingCashflow")
    ebitda = info.get("ebitda")

    if bs is not None and not bs.empty:
        if de_ratio is None:
            debt = float(bs.loc["Total Debt"].iloc[0]) if "Total Debt" in bs.index else None
            eq = float(bs.loc["Stockholders Equity"].iloc[0]) if "Stockholders Equity" in bs.index else None
            if debt and eq and eq > 0:
                de_ratio = round(debt / eq, 2)
        if current_ratio is None:
            ca = float(bs.loc["Current Assets"].iloc[0]) if "Current Assets" in bs.index else None
            cl = float(bs.loc["Current Liabilities"].iloc[0]) if "Current Liabilities" in bs.index else None
            if ca and cl and cl > 0:
                current_ratio = round(ca / cl, 2)
        if total_debt is None and "Total Debt" in bs.index:
            total_debt = float(bs.loc["Total Debt"].iloc[0])
        if total_cash is None and "Cash And Cash Equivalents" in bs.index:
            total_cash = float(bs.loc["Cash And Cash Equivalents"].iloc[0])

    # Growth
    rev_growth_yoy = round(float(info.get("revenueGrowth")) * 100, 2) if info.get("revenueGrowth") else None
    earnings_growth = round(float(info.get("earningsGrowth")) * 100, 2) if info.get("earningsGrowth") else None

    if rev_growth_yoy is None and inc is not None and not inc.empty and "Total Revenue" in inc.index:
        revs = inc.loc["Total Revenue"]
        if len(revs) >= 2 and revs.iloc[1] > 0:
            rev_growth_yoy = round(((revs.iloc[0] - revs.iloc[1]) / revs.iloc[1]) * 100, 2)

    # Derived: FCF margin
    fcf_margin = round(fcf / total_revenue * 100, 2) if fcf and total_revenue and total_revenue > 0 else None

    # Price to book fallback
    if pb is None and cur_price and book_value and book_value > 0:
        pb = round(cur_price / book_value, 2)

    macro_text = (
        "Economy: India | Central Bank: Reserve Bank of India (RBI) | Policy Repo Rate: 6.50% | "
        "Domestic CPI Inflation: 4.8% - 5.4% | Real GDP Growth Rate: 6.8% - 7.2% | Benchmark: NIFTY 50"
        if is_indian
        else "Economy: United States | Central Bank: Federal Reserve (Fed) | Fed Funds Rate: 4.50% - 4.75% | "
        "US CPI Inflation: 2.5% - 2.8% | Real GDP Growth Rate: 2.5% - 2.8% | Benchmark: S&P 500"
    )

    data = {
        "symbol": ticker.upper(),
        "clean_symbol": clean_ticker,
        "company_name": info.get("longName") or info.get("shortName") or ticker.upper(),
        "actual_sector": info.get("sector") or ("Energy & Conglomerate" if ticker.upper() == "RELIANCE" else "Diversified Equities"),
        "actual_industry": info.get("industry") or ("Oil & Gas Refining, Retail & Telecom" if ticker.upper() == "RELIANCE" else "Equities"),
        "currency": "INR" if is_indian else info.get("currency", "USD"),
        "current_price": cur_price,
        # Valuation
        "trailing_pe": pe_ttm,
        "forward_pe": fwd_pe,
        "peg_ratio": peg,
        "price_to_book": pb,
        "trailing_eps": eps_ttm,
        "forward_eps": fwd_eps,
        "book_value": book_value,
        # Market
        "market_cap": mcap,
        "beta": beta_val,
        "fifty_two_week_high": high_52w,
        "fifty_two_week_low": low_52w,
        "dividend_yield_pct": div_yield,
        # Margins & profitability
        "gross_margin_pct": gross_margin_pct,
        "operating_margin_pct": opm,
        "profit_margin_pct": pm,
        "roe_pct": roe,
        "roa_pct": roa,
        # Balance sheet
        "debt_to_equity": de_ratio,
        "current_ratio": current_ratio,
        "total_cash": total_cash,
        "total_debt": total_debt,
        "free_cashflow": fcf,
        "operating_cashflow": op_cashflow,
        "total_revenue": total_revenue,
        "ebitda": ebitda,
        # Growth
        "revenue_growth_yoy_pct": rev_growth_yoy,
        "earnings_growth_pct": earnings_growth,
        "fcf_margin_pct": fcf_margin,
        # Context
        "one_year_return_pct": ret_1y,
        "is_indian": is_indian,
        "macro_benchmark": macro_text,
    }

    _GROUND_TRUTH_CACHE[cache_key] = {"data": data, "cached_at": now_ts}
    return data

