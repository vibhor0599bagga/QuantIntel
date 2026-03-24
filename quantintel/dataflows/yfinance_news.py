"""
quantintel/dataflows/yfinance_news.py
Copied from TradingAgents — no changes needed.
"""

import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta


def _extract_article_data(article: dict) -> dict:
    if "content" in article:
        content   = article["content"]
        title     = content.get("title", "No title")
        summary   = content.get("summary", "")
        provider  = content.get("provider", {})
        publisher = provider.get("displayName", "Unknown")
        url_obj   = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        link      = url_obj.get("url", "")
        pub_date_str = content.get("pubDate", "")
        pub_date = None
        if pub_date_str:
            try:
                pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        return {"title": title, "summary": summary, "publisher": publisher,
                "link": link, "pub_date": pub_date}
    else:
        return {
            "title":     article.get("title", "No title"),
            "summary":   article.get("summary", ""),
            "publisher": article.get("publisher", "Unknown"),
            "link":      article.get("link", ""),
            "pub_date":  None,
        }


def get_news_yfinance(ticker: str, start_date: str, end_date: str) -> str:
    try:
        stock = yf.Ticker(ticker)
        news  = stock.get_news(count=20)
        if not news:
            return f"No news found for {ticker}"

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")

        news_str = ""
        filtered = 0
        for article in news:
            data = _extract_article_data(article)
            if data["pub_date"]:
                naive = data["pub_date"].replace(tzinfo=None)
                if not (start_dt <= naive <= end_dt + relativedelta(days=1)):
                    continue
            news_str += f"### {data['title']} (source: {data['publisher']})\n"
            if data["summary"]:
                news_str += f"{data['summary']}\n"
            if data["link"]:
                news_str += f"Link: {data['link']}\n"
            news_str += "\n"
            filtered += 1

        if filtered == 0:
            return f"No news found for {ticker} between {start_date} and {end_date}"
        return f"## {ticker} News, from {start_date} to {end_date}:\n\n{news_str}"

    except Exception as e:
        return f"Error fetching news for {ticker}: {e}"


def get_global_news_yfinance(curr_date: str, look_back_days: int = 7, limit: int = 10) -> str:
    queries = [
        "stock market economy",
        "Federal Reserve interest rates",
        "inflation economic outlook",
        "global markets trading",
    ]

    all_news, seen_titles = [], set()
    try:
        for query in queries:
            search = yf.Search(query=query, news_count=limit, enable_fuzzy_query=True)
            if search.news:
                for article in search.news:
                    data  = _extract_article_data(article)
                    title = data["title"]
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        all_news.append(article)
            if len(all_news) >= limit:
                break

        if not all_news:
            return f"No global news found for {curr_date}"

        curr_dt    = datetime.strptime(curr_date, "%Y-%m-%d")
        start_date = (curr_dt - relativedelta(days=look_back_days)).strftime("%Y-%m-%d")

        news_str = ""
        for article in all_news[:limit]:
            data = _extract_article_data(article)
            news_str += f"### {data['title']} (source: {data['publisher']})\n"
            if data["summary"]:
                news_str += f"{data['summary']}\n"
            if data["link"]:
                news_str += f"Link: {data['link']}\n"
            news_str += "\n"

        return f"## Global Market News, from {start_date} to {curr_date}:\n\n{news_str}"

    except Exception as e:
        return f"Error fetching global news: {e}"
