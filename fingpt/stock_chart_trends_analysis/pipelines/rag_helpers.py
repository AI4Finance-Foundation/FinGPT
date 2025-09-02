from typing import List, Optional, Dict, Any
from pathlib import Path
import hashlib, yfinance as yf
import finance_rag_semantic as frs
from helpers.llm_utils import llm_generate
from config import OPENAI_MODEL, client, logger
from datetime import datetime, timedelta
from general_utils import get_curday

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
