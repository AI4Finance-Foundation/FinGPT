import os
import re
import json
import time
import hashlib
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import sys
sys.path.insert(0, r"/content/FinGPT-M/fingpt/stock_chart_trends_analysis")

import torch
import gradio as gr
import pandas as pd
import yfinance as yf
import finnhub
from dotenv import load_dotenv
from datetime import date, datetime, timedelta
import nltk
from nltk.corpus import stopwords
# NEW: OpenAI SDK (pip install openai>=1.0.0)
from openai import OpenAI
from StockChart_Trend_Prediction import StockChartTrendPredictor, StockChartMetadataExtractor
import finance_rag_semantic as frs
from Helpers.helper import safe_cleanup
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # saves logs to file
        logging.StreamHandler()          # prints to console
    ]
)
logger = logging.getLogger("FinGPT-M")

# --- ENV & setup ---
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
if not FINNHUB_KEY:
    raise RuntimeError("FINNHUB_API_KEY not set")

YOLO_WEIGHTS = os.getenv(
    "YOLO_WEIGHTS",
    r"C:\\Users\\rahul\\OneDrive\\Desktop\\FinGPT-M\\fingpt\\stock_chart_trends_analysis\\best.pt"
)
os.environ["YOLO_MODEL_PATH"] = YOLO_WEIGHTS
os.environ["ULTRALYTICS_VERBOSE"] = "False"

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
EN_STOP = set(stopwords.words("english"))
EN_STOP.update({"buy", "sell", "hold", "call", "put", "usd", "nse", "bse", "nyse", "nasdaq", "market", "stock", "shares"})

finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)

# --- OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Constants ---
OUTPUT_BEGIN = "[OUTPUT_BEGIN]"
OUTPUT_END = "[OUTPUT_END]"
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DOC_EXT = {".pdf", ".docx", ".txt", ".md", ".csv", ".html", ".htm", ".pptx"}

# --- Helpers ---
def _between_markers(text: str) -> str:
    if OUTPUT_BEGIN in text:
        text = text.split(OUTPUT_BEGIN, 1)[1]
    if OUTPUT_END in text:
        text = text.split(OUTPUT_END, 1)[0]
    return text.strip()

def get_curday() -> str:
    return date.today().strftime("%Y-%m-%d")

def _clean_terms_for_company(text: str) -> List[str]:
    if not text:
        return []
    words = re.findall(r"[A-Za-z0-9&.'-]+", text)
    keep = []
    for w in words:
        wl = w.lower()
        if wl in EN_STOP:
            continue
        if len(wl) <= 1:
            continue
        keep.append(wl)
    return keep

def finnhub_lookup_ticker_by_name(name_query: str) -> Optional[str]:
    if not name_query:
        return None
    try:
        res = finnhub_client.symbol_lookup(name_query)
    except Exception as e:
        logger.info(f"finnhub symbol_lookup error: {e}")
        return None
    if not res or "result" not in res:
        return None
    items = res.get("result", []) or []
    if not items:
        return None

    q_norm = name_query.strip().lower()

    def score(item):
        sym = (item.get("symbol") or "")
        desc = (item.get("description") or "").lower()
        typ = (item.get("type") or "").lower()
        exch = (item.get("exchange") or "").upper()
        s = 0
        if q_norm == desc: s += 100
        if q_norm in desc: s += 40
        if "common stock" in typ: s += 10
        if exch in {"NASDAQ","NYSE","NYSE ARCA","NYSE MKT"}: s += 6
        s += max(0, 5 - len(sym))
        return s

    items = sorted(items, key=score, reverse=True)
    best = items[0]
    sym = best.get("symbol")
    logger.info(f"[ticker-lookup] '{name_query}' -> '{sym}'")
    return sym

def _prefer_first_uppercase_token(U: str) -> Optional[str]:
    tokens = re.findall(r"\b[A-Z][A-Z.\-]{0,9}\b", U)
    COMMON = {"FOR","AND","THE","A","AN","TO","OF","ON","IN","AT","BY","WITH","FROM","AS","IS","VS"}
    for t in tokens:
        if t.lower() in EN_STOP or t in COMMON:
            continue
        return t
    return None

def finnhub_lookup_ticker_by_name(name_query: str) -> Optional[str]:
    try:
        res = finnhub_client.symbol_lookup(name_query)
    except Exception:
        return None
    if not res or "result" not in res:
        return None
    items = res.get("result", [])
    q_norm = name_query.strip().lower()
    def score(item):
        sym = (item.get("symbol") or "")
        desc = (item.get("description") or "").lower()
        typ = (item.get("type") or "").lower()
        exch = (item.get("exchange") or "").upper()
        s = 0
        if q_norm == desc: s += 100
        if q_norm in desc: s += 40
        if "common stock" in typ: s += 10
        if exch in {"NASDAQ","NYSE","NYSE ARCA","NYSE MKT"}: s += 6
        s += max(0, 5 - len(sym))
        return s
    if not items:
        return None
    best = sorted(items, key=score, reverse=True)[0]
    return best.get("symbol")

def resolve_ticker_from_query(q: str) -> Optional[str]:
    Q, U = q.strip(), q.strip().upper()
    m_sym = re.search(r"\(([A-Z.\-]{1,10})\)", U)
    if m_sym: return m_sym.group(1)
    sym = _prefer_first_uppercase_token(U)
    if sym: return sym
    terms = [t for t in _clean_terms_for_company(Q) if not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", t)]
    terms = [t for t in terms if t not in {"forecast","analyze","analysis","analyse","price","target"}]
    if terms:
        return finnhub_lookup_ticker_by_name(" ".join(terms[:5]))
    return None

def parse_query(q: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract (explicit_ticker, day, company_hint):
      - explicit_ticker: only when provided as (TSLA) in parentheses
      - day: YYYY-MM-DD if present
      - company_hint: cleaned words to help name->ticker lookup if needed
    """
    if not q:
        return None, None, None
    # explicit ticker only if in parentheses
    m_sym = re.search(r"\(([A-Z][A-Z0-9.\-]{0,9})\)", q.upper())
    explicit = m_sym.group(1) if m_sym else None
    # date
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
    day = m.group(1) if m else None
    # company hint (for optional fallback when no docs in session)
    terms = _clean_terms_for_company(q)
    company_hint = " ".join(terms[:5]) if terms else None
    return explicit, day, company_hint

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

def simple_sentiment_from_patterns(preds: List[Dict[str, Any]]) -> str:
    bullish = {"morning_star_rise","hammer","bullish_engulfing","ascending_triangle","golden_cross"}
    bearish = {"evening_star_fall","shooting_star","bearish_engulfing","descending_triangle","death_cross"}
    score = sum(1 for p in preds if any(b in p.get("class","").lower() for b in bullish)) - \
            sum(1 for p in preds if any(b in p.get("class","").lower() for b in bearish))
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"

# --- LLM wrapper (OpenAI Chat Completions) ---
def llm_generate(system_prompt: str,
                 user_prompt: str,
                 max_new_tokens: int = 500,
                 temperature: float = 0.2,
                 top_p: float = 0.95,
                 stop: Optional[List[str]] = None) -> str:
    """
    Thin wrapper over OpenAI Chat Completions returning assistant text.
    """
    stop = stop or None
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=stop,
    )
    return (resp.choices[0].message.content or "").strip()

# --- Chart & text handlers ---
def llm_brief_summary(final_output: Dict[str, Any], sentiment: str, mode: str = "image") -> str:
    ctx = {
        "ticker": final_output.get("ticker") or final_output.get("company_ticker"),
        "company_name": final_output.get("company_name"),
        "exchange": final_output.get("exchange"),
        "price_range": final_output.get("price_range"),
        "ohlc": final_output.get("ohlc"),
        "predictions": final_output.get("predictions"),
        "pattern_sentiment": sentiment,
    }
    title = "### Chart Summary" if mode == "image" else "### Text Query Summary"
    system = (
        "You are a precise market assistant. Using ONLY the provided JSON, "
        "return EXACTLY 5 concise markdown bullets. Each bullet must start with '- ' "
        "and include a bold label followed by a short fact."
    )
    user = (
        f"[JSON]\n{json.dumps(ctx, indent=2)}\n\n"
        f"{OUTPUT_BEGIN}\n"
        f"- **Label**: point\n"
        f"- **Label**: point\n"
        f"- **Label**: point\n"
        f"- **Label**: point\n"
        f"- **Label**: point\n"
        f"{OUTPUT_END}"
    )
    raw = llm_generate(system, user, max_new_tokens=280, temperature=0.2, top_p=0.95, stop=[OUTPUT_END])
    bullets = [ln.strip() for ln in _between_markers(raw).splitlines() if ln.strip().startswith("- ")]
    return title + "\n" + "\n".join(bullets)

def build_llm_prompts_for_forecast(final_output, news_snippets) -> Tuple[str, str]:
    ctx = {
        "ticker": final_output.get("ticker"),
        "exchange": final_output.get("exchange"),
        "ohlc": final_output.get("ohlc"),
        "sessions": final_output.get("sessions"),
        "price_range": final_output.get("price_range"),
        "predictions": final_output.get("predictions"),
    }
    news_text = "\n".join(
        f"- {n['date']} | {n['headline']} :: {n['summary'][:200]}..."
        for n in (news_snippets or [])[:6]
    )
    system = (
        "You are a seasoned stock market analyst. Use ONLY the JSON (if available) and NEWS provided. "
        "Write exactly three markdown sections: Positive Developments, Potential Concerns, Forecast & Analysis."
    )
    user = (
        f"[JSON]\n{json.dumps(ctx, indent=2)}\n\n"
        f"[NEWS]\n{news_text if news_text else 'No recent news'}\n\n"
        f"{OUTPUT_BEGIN}\n<Your answer here>\n{OUTPUT_END}"
    )
    return system, user

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

# --- RAG helpers ---
def _make_unique_index_dir(file_name: str) -> Path:
    h = hashlib.sha1(file_name.encode("utf-8")).hexdigest()[:12]
    d = Path("./faiss_indexes") / f"idx_{h}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def prepare_rag_session_for_file(file_path: str):
    filename = Path(file_path).name
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    unique_dir = _make_unique_index_dir(filename)
    frs.INDEX_DIR = unique_dir
    session = frs.prepare_session_any(file_bytes, filename)
    ctx, meta = frs.get_summary_context(session, max_chunks=8)
    system = (
        "You are a senior equity analyst. Provide a concise executive summary of the document "
        "covering: company, filing/report type, period/fiscal year, revenue/earnings highlights, "
        "segment/geographic notes, liquidity & capital resources, key risks, guidance/outlook, "
        "and any dividends/buybacks. Use bullets."
    )
    head = f"Meta: company={meta.company}, ticker={meta.ticker}, FY={meta.fiscal_year}, period={meta.period}"
    user = f"{head}\n\nContext:\n{ctx}\n\nSummary:"
    summary = llm_generate(system, user, max_new_tokens=450, temperature=0.2)
    return session, meta, f"**Executive Summary — {meta.title or filename}**\n\n{summary}"

def retrieve_across_sessions(question: str, sessions: List[frs.RAGSession], top_k: int = None):
    if not sessions: return {"context": "", "sources": []}
    top_k = top_k or frs.TOP_K
    ctx_parts, sources = [], []
    for s in sessions:
        r = frs.retrieve(s, question, k=top_k)
        ctx_parts.append(r.get("context", ""))
        sources.extend(r.get("sources", []))
    return {"context": "\n\n".join(ctx_parts), "sources": sources}

def format_sources_for_display(sources):
    return "\n".join(
        f"- {Path(s.get('source','unknown')).name} · chunk {s.get('chunk_id','?')}"
        for s in sources
    ) if sources else ""

def generate_text(user_prompt: str,
                  max_new_tokens: int = 400,
                  temperature: float = 0.2,
                  top_p: float = 0.95) -> str:
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise financial analyst. Answer factually, using only the given context."},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    return (resp.choices[0].message.content or "").strip()

def _yf_symbol(ticker: str, exchange: Optional[str] = None) -> str:
    # simple passthrough for now; extend if you need NSE/BSE suffixing
    return (ticker or "").strip()

def ensure_min_ohlc(final_output: Dict[str, Any]) -> None:
    """If OHLC/price_range are missing but we have a ticker, fetch 1 day via yfinance."""
    tkr = final_output.get("ticker") or final_output.get("company_ticker")
    if not tkr:
        return
    if final_output.get("ohlc") and final_output.get("price_range"):
        return
    day = final_output.get("date") or get_curday()
    try:
        end_day = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        yf_tkr = _yf_symbol(tkr)
        df = yf.download(yf_tkr, start=day, end=end_day, progress=False)
        if len(df):
            row = df.iloc[0]
            ohlc = {
                "O": float(row["Open"]),
                "H": float(row["High"]),
                "L": float(row["Low"]),
                "C": float(row["Close"]),
                "V": float(row["Volume"]),
            }
            final_output.setdefault("ohlc", ohlc)
            final_output.setdefault("price_range", [ohlc["L"], ohlc["H"]])
    except Exception as e:
        logger.error(f"[ensure_min_ohlc] skipped: {e}")

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
            final_output = ensure_final_output_from_image(p)
            safe_cleanup(p)
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
            session, meta, summary_text = prepare_rag_session_for_file(p)
            safe_cleanup(p)
            rag_sessions.append(session)
            assistant_reply_sections.append(f"**Document processed:** {Path(p).name}\n\n{summary_text}")

    # --- TEXT QUERY HANDLING ---
    if user_text and not file_paths:
        print('Handling text query...', user_text)
        explicit_ticker, day, company_hint = parse_query(user_text)
        print(f"Parsed query: explicit_ticker={explicit_ticker}, day={day}, company_hint={company_hint}")
        print(f"RAG sessions count: {len(rag_sessions)}")
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

    history_pairs[-1] = (history_pairs[-1][0], "\n\n".join(assistant_reply_sections))
    return history_pairs, last_summary_md, last_sentiment, last_forecast_md, last_json_str, last_news_df, last_json_path

# --- UI ---
with gr.Blocks(title="Unified Finance Chatbot (OpenAI)") as demo:
    chat = gr.Chatbot(label=None, height=520, show_copy_button=True, bubble_full_width=False)
    # Note: depending on your Gradio version, MultimodalTextbox signature may vary.
    mm = gr.MultimodalTextbox(
        placeholder="Ask a question, drop a chart image, or upload a document...",
        show_label=False,
        file_types=list(IMG_EXT | DOC_EXT),
        autofocus=True,
        submit_btn=True
    )
    summary_out = gr.Markdown(visible=False)
    sentiment_out = gr.Textbox(visible=False)
    forecast_out = gr.Markdown(visible=False)
    json_out = gr.Code(language="json", visible=False)
    news_out = gr.Dataframe(visible=False)
    json_download = gr.DownloadButton(label="Download final_output.json", visible=False)
    rag_state = gr.State([])
    chat_state = gr.State([])

    def on_mm_submit_pairs(history_pairs, data, rag_sessions, do_news=True):
        user_text = (data or {}).get("text") or ""
        user_files = (data or {}).get("files") or []
        result = handle_user_turn(history_pairs, user_text, user_files, rag_sessions, do_news)
        return (*result, gr.update(value=None))

    mm.submit(
        fn=on_mm_submit_pairs,
        inputs=[chat_state, mm, rag_state],
        outputs=[chat, summary_out, sentiment_out, forecast_out, json_out, news_out, json_download, mm]
    ).then(
        fn=lambda jo, df, jp: (gr.update(value=jo), df, jp),
        inputs=[json_out, news_out, json_download],
        outputs=[json_out, news_out, json_download]
    ).then(
        fn=lambda h, rs: (h, rs),
        inputs=[chat, rag_state],
        outputs=[chat_state, rag_state]
    )

if __name__ == "__main__":
    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(f"YOLO weights not found at: {YOLO_WEIGHTS}")
    demo.launch(share=True, debug=True)