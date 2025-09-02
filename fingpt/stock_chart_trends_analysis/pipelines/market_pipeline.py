from helpers.parse_utils import parse_query
from helpers.general_utils import _between_markers, get_curday
from helpers.rag_helpers import ensure_min_ohlc
from helpers.llm_utils import llm_generate, llm_brief_summary, simple_sentiment_from_patterns, build_llm_prompts_for_forecast
from config import logger, OUTPUT_END
from typing import Dict, Any, List
from news_fetcher import news_for_window
import json, pandas as pd, gradio as gr, os

def run_market_pipeline(final_output: Dict[str, Any], do_news: bool):
    ticker = final_output.get("ticker") or final_output.get("company_ticker")
    if not ticker:
        raise gr.Error("Ticker not found for market analysis.")
    anchor_date = final_output.get("date") or get_curday()

    ensure_min_ohlc(final_output)

    sentiment = simple_sentiment_from_patterns(final_output.get("predictions", []))
    try:
        summary_md = llm_brief_summary(final_output, sentiment, mode="image" if final_output.get("source") != "text_query" else "text")
    except Exception as e:
        logger.error("brief summary fallback:", e)
        # minimal deterministic fallback
        pr = final_output.get("price_range") or [None, None]
        ohlc = final_output.get("ohlc") or {}
        summary_md = (
            "### Text Query Summary\n"
            f"- **Ticker**: {ticker}\n"
            f"- **Price Range**: {pr[0]} – {pr[1]}\n"
            f"- **OHLC**: O {ohlc.get('O')}, H {ohlc.get('H')}, L {ohlc.get('L')}, C {ohlc.get('C')}\n"
            f"- **Sentiment**: {sentiment}\n"
        )

    news_items: List[Dict[str, Any]] = news_for_window(ticker, anchor_date, weeks=1) if do_news else []

    sys_f, usr_f = build_llm_prompts_for_forecast(final_output, news_items)
    forecast_md = _between_markers(
        llm_generate(sys_f, usr_f, max_new_tokens=500, temperature=0.7, top_p=0.9, stop=[OUTPUT_END])
    ) or "No forecast generated."

    json_path = os.path.abspath("final_output.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    final_json_str = json.dumps(final_output, indent=2, ensure_ascii=False)

    return (
        summary_md,
        sentiment,
        forecast_md,
        final_json_str,
        pd.DataFrame(news_items) if news_items else pd.DataFrame([{"info": "News omitted or none found."}]),
        json_path,
    )
