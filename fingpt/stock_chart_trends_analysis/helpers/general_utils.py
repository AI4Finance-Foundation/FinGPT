import re
from datetime import date
from typing import Dict, Any, Optional, Tuple, List
from config import OUTPUT_BEGIN, OUTPUT_END, logger, EN_STOP, finnhub_client
from nltk.corpus import stopwords

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
