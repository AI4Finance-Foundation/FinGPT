import os, re, nltk, string
from typing import Dict, Any, Optional, Tuple, List
from config import EN_STOP, ACTION_TERMS, USE_SPACY, nlp, logger

def safe_cleanup(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("Cleaned up temporary file: %s", file_path)
    except Exception as e:
        logger.warning("Failed to cleanup %s: %s", file_path, e)

def _clean_terms_for_company(text: str) -> List[str]:
    if not text:
        return []
    words = re.findall(r"[A-Za-z0-9&.'-]+", text)
    keep = []
    for w in words:
        wl = w.lower().strip(string.punctuation)
        if wl in EN_STOP:
            continue
        if wl in ACTION_TERMS:
            continue
        if len(wl) <= 1:
            continue
        keep.append(wl)
    return keep

def parse_query(q: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not q:
        return None, None, None

    # explicit ticker
    m_sym = re.search(r"\(([A-Z][A-Z0-9.\-]{0,9})\)", q.upper())
    explicit = m_sym.group(1) if m_sym else None

    # date
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
    day = m.group(1) if m else None

    # initialize
    company_hint = None
    filtered_terms = []

    # --- Try spaCy if available ---
    if USE_SPACY:
        doc = nlp(q)
        companies = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        if companies:
            company_hint = companies[0].lower()

    # --- Fallback logic if spaCy not available OR no ORG detected ---
    if not company_hint:
        terms = _clean_terms_for_company(q)
        for t in terms:
            if explicit and t.lower() == explicit.lower():
                continue
            if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", t):  # filter out dates
                continue
            filtered_terms.append(t)

        # Prefer capitalized word in query
        for word in q.split():
            if word and word[0].isupper() and word.lower() not in ACTION_TERMS:
                company_hint = word.lower().strip(string.punctuation)
                break

        # Final fallback: first filtered term
        if not company_hint and filtered_terms:
            company_hint = filtered_terms[0] 

    return explicit, day, company_hint