import pandas as pd, os, re, json, time
from pathlib import Path
from datetime import date, datetime, timedelta
from StockChart_Trend_Prediction import StockChartTrendPredictor, StockChartMetadataExtractor
from typing import Any, Dict, List, Tuple
import yfinance as yf
import finance_rag_semantic as frs
from helpers.parse_utils import safe_cleanup, parse_query
from helpers.llm_utils import simple_sentiment_from_patterns, llm_generate, llm_brief_summary, build_llm_prompts_for_forecast, OUTPUT_BEGIN, OUTPUT_END
from helpers.rag_helpers import prepare_rag_session_for_file, retrieve_across_sessions, format_sources_for_display
from pipelines.news_fetcher import news_for_window
from pipelines.market_pipeline import run_market_pipeline
from helpers.general_utils import get_curday, _between_markers, finnhub_lookup_ticker_by_name
from config import YOLO_WEIGHTS, logger, finnhub_client, IMG_EXT, DOC_EXT, OUTPUT_END
os.environ["YOLO_MODEL_PATH"] = YOLO_WEIGHTS
os.environ["ULTRALYTICS_VERBOSE"] = "False"

def ensure_final_output_from_image(image_path: str) -> Dict[str, Any]:
    metadata_extractor = StockChartMetadataExtractor(image_path)
    metadata = metadata_extractor.extract_metadata()
    predictor = StockChartTrendPredictor(YOLO_WEIGHTS)
    try:
        predictor.model.to("cuda")
    except Exception:
        pass
    preds, img = predictor.predict(image_path)
    final = predictor.save_predictions_to_json(preds, "final_output.json", img, metadata)
    if not final.get("ticker"):
        title = (final.get("company_name") or "") + " " + (final.get("title") or "")
        m = re.search(r"\(([A-Z.\-]{1,10})\)", title.upper())
        if m: final["ticker"] = m.group(1)
        # --- Fetch latest market trend using yfinance ---
        ticker = final.get("ticker") or final.get("company_name")
        latest_market_trend = None
        latest_market_ohlc = None
        if ticker:
            try:
                # Try yfinance first
                df = yf.download(ticker, period="5d", progress=False)
                if len(df):
                    latest_row = df.iloc[-1]
                    latest_market_ohlc = {
                        "O": float(latest_row["Open"]),
                        "H": float(latest_row["High"]),
                        "L": float(latest_row["Low"]),
                        "C": float(latest_row["Close"]),
                        "V": float(latest_row["Volume"])
                    }
                    # Simple trend: compare last close vs previous close
                    if len(df) > 1:
                        prev_close = df.iloc[-2]["Close"]
                        curr_close = latest_row["Close"]
                        if curr_close > prev_close:
                            latest_market_trend = "Bullish"
                        elif curr_close < prev_close:
                            latest_market_trend = "Bearish"
                        else:
                            latest_market_trend = "Neutral"
            except Exception as e:
                logger.error(f"[yfinance trend fetch] {e}")
        # Fallback to finnhub if yfinance fails
        if not latest_market_trend and ticker:
            try:
                quote = finnhub_client.quote(ticker)
                prev_close = quote.get("pc")
                curr_close = quote.get("c")
                if prev_close is not None and curr_close is not None:
                    if curr_close > prev_close:
                        latest_market_trend = "Bullish"
                    elif curr_close < prev_close:
                        latest_market_trend = "Bearish"
                    else:
                        latest_market_trend = "Neutral"
                    latest_market_ohlc = {
                        "O": quote.get("o"),
                        "H": quote.get("h"),
                        "L": quote.get("l"),
                        "C": quote.get("c"),
                        "V": None
                    }
            except Exception as e:
                logger.error(f"[finnhub trend fetch] {e}")

        # --- Compare chart prediction with latest market trend ---
        chart_sentiment = simple_sentiment_from_patterns(final.get("predictions", []))
        comparison = {
            "chart_sentiment": chart_sentiment,
            "market_trend": latest_market_trend,
            "match": chart_sentiment == latest_market_trend if latest_market_trend else None,
            "chart_ohlc": final.get("ohlc"),
            "market_ohlc": latest_market_ohlc
        }
        final["trend_comparison"] = comparison
    return final

def ensure_final_output_from_text(query: str) -> Dict[str, Any]:
    ticker, day, _ = parse_query(query)
    final = {
        "ticker": ticker, "date": day or get_curday(), "source": "text_query",
        "predictions": [], "sessions": [], "price_range": [None, None], "ohlc": {},
        "exchange": None, "company_name": None
    }
    if ticker:
        end_day = (datetime.strptime(final["date"], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=final["date"], end=end_day, progress=False)
        if len(df):
            row = df.iloc[0]
            final["ohlc"] = {
                "O": float(row["Open"]), "H": float(row["High"]),
                "L": float(row["Low"]), "C": float(row["Close"]),
                "V": float(row["Volume"])
            }
            final["price_range"] = [float(row["Low"]), float(row["High"])]
    with open("final_output.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    return final

# --- Chat handler ---
def handle_user_turn(history_pairs, user_text, user_files, rag_sessions, do_news=True):
    last_summary_md = last_sentiment = last_forecast_md = last_json_str = ""
    last_news_df, last_json_path = pd.DataFrame(), None
    file_paths = [str(f) for f in (user_files or []) if f]
    user_display = (user_text or "").strip() or "(no text)"
    if file_paths: user_display += "\n[attachments: " + ", ".join(Path(p).name for p in file_paths) + "]"
    history_pairs = (history_pairs or []) + [(user_display, "")]
    assistant_reply_sections = []

    for p in file_paths:
        ext = Path(p).suffix.lower()
        if ext in IMG_EXT:
            input_type = "image"
            start_time = time.time()
            final_output = ensure_final_output_from_image(p)
            ticker = final_output.get("ticker")
            if ticker:
                anchor_date = final_output.get("date") or get_curday()
                sentiment = simple_sentiment_from_patterns(final_output.get("predictions", []))
                summary_md = llm_brief_summary(final_output, sentiment, mode="image")
                news_items = news_for_window(ticker, anchor_date, weeks=1)
                sys_f, usr_f = build_llm_prompts_for_forecast(final_output, news_items)
                forecast_md = _between_markers(
                    llm_generate(sys_f, usr_f, max_new_tokens=500, temperature=0.7, top_p=0.9, stop=[OUTPUT_END])
                )
                json_path = os.path.abspath("final_output.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(final_output, f, indent=2, ensure_ascii=False)
                final_json_str = json.dumps(final_output, indent=2, ensure_ascii=False)
                assistant_reply_sections.append(
                    f"**Chart analyzed:** {Path(p).name}\n\n{summary_md}\n\n---\n**Sentiment:** `{sentiment}`\n\n### LLM Forecast\n{forecast_md}"
                )
                last_summary_md, last_sentiment, last_forecast_md = summary_md, sentiment, forecast_md
                last_json_str, last_news_df, last_json_path = final_json_str, pd.DataFrame(news_items), json_path
        elif ext in DOC_EXT:
            input_type = "document"
            start_time = time.time()
            session, meta, summary_text = prepare_rag_session_for_file(p)
            rag_sessions.append(session)
            assistant_reply_sections.append(f"**Document processed:** {Path(p).name}\n\n{summary_text}")


    # --- TEXT QUERY HANDLING ---
    if user_text and not file_paths:
        input_type = "text"
        start_time = time.time()
        explicit_ticker, day, company_hint = parse_query(user_text)
        # 2a) If RAG sessions exist and user did NOT explicitly provide (TICKER), answer from docs
        if rag_sessions and not explicit_ticker:
            ret = retrieve_across_sessions(user_text, rag_sessions, top_k=frs.TOP_K)
            ctx, sources = ret["context"], ret["sources"]
            qa_prompt = (
                "You are a diligent financial analyst. Use ONLY the provided context. "
                "If the answer is not explicitly in the context, say you cannot find it. "
                "Be specific and include figures with units and dates when available.\n\n"
                f"Question:\n{user_text}\n\nContext:\n{ctx}\n\nAnswer:"
            )
            ans = llm_generate(
                "You are a precise financial analyst. Answer factually, using only the given context.",
                qa_prompt,
                max_new_tokens=420,
                temperature=0.2
            )
            srcs = format_sources_for_display(sources)
            if srcs:
                ans += f"\n\n---\n{srcs}"
            assistant_reply_sections.append(f"**Financial Document Answer**\n\n{ans}")

        # 2b) Else if explicit (TICKER) was given, do market pipeline (yfinance + forecast)
        elif explicit_ticker:
            # build final_output from the text, but **force** the explicit ticker & date
            final_output = ensure_final_output_from_text(user_text)
            final_output["ticker"] = explicit_ticker
            if day:
                final_output["date"] = day
            s_md, sent, f_md, j_str, n_df, j_path = run_market_pipeline(final_output, do_news)
            assistant_reply_sections.append(
                f"**Market Query:** {user_text}\n\n{s_md}\n\n---\n**Sentiment:** `{sent}`\n\n### LLM Forecast\n{f_md}"
            )
            last_summary, last_sent, last_forecast = s_md, sent, f_md
            last_json_str, last_news_df, last_json_path = j_str, n_df, j_path

        # 2c) Else (no docs, no explicit ticker): optional name→ticker lookup via Finnhub
        else:
            resolved = finnhub_lookup_ticker_by_name(company_hint or user_text)
            if resolved:
                # run market with resolved ticker
                query_for_builder = f"{resolved} ({day})" if day else resolved
                final_output = ensure_final_output_from_text(query_for_builder)
                final_output["ticker"] = resolved
                if day:
                    final_output["date"] = day
                s_md, sent, f_md, j_str, n_df, j_path = run_market_pipeline(final_output, do_news)
                assistant_reply_sections.append(
                    f"**Market Query (resolved):** {resolved}\n\n{s_md}\n\n---\n**Sentiment:** `{sent}`\n\n### LLM Forecast\n{f_md}"
                )
                last_summary, last_sent, last_forecast = s_md, sent, f_md
                last_json_str, last_news_df, last_json_path = j_str, n_df, j_path
            else:
                assistant_reply_sections.append(
                    "I didn’t find an explicit `(TICKER)` and there’s no document context to search.\n"
                    "Add `(TICKER)` to your question or upload a financial document."
                )
    end_time = time.time()
    latency = end_time - (start_time if 'start_time' in locals() else end_time)
    print(f"{input_type} latency: {latency:.3f} seconds")
    history_pairs[-1] = (history_pairs[-1][0], "\n\n".join(assistant_reply_sections))
    return history_pairs, last_summary_md, last_sentiment, last_forecast_md, last_json_str, last_news_df, last_json_path