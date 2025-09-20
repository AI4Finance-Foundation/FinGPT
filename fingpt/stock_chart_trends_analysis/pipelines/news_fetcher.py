from datetime import datetime, timedelta
from config import finnhub_client

def get_company_news(symbol: str, start_date: str, end_date: str):
    weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
    return [{
        "date": datetime.fromtimestamp(n["datetime"]).strftime("%Y-%m-%d %H:%M:%S"),
        "headline": n.get("headline", ""),
        "summary": n.get("summary", ""),
        "source": n.get("source", ""),
        "url": n.get("url", "")
    } for n in (weekly_news or []) if not str(n.get("summary", "")).startswith("Looking for stock market analysis")]

def news_for_window(symbol: str, anchor_day: str, weeks: int = 1):
    try:
        end_dt = datetime.strptime(anchor_day, "%Y-%m-%d")
    except Exception:
        end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=7 * weeks)
    return get_company_news(symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))