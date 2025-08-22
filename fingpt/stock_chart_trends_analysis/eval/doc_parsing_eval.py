# -*- coding: utf-8 -*-
"""
No-GT Document Parsing Evaluation for financial PDFs.

- Runs your PdfExtractor and a baseline (pdfminer.six) for weak-supervision agreement
- Computes text health/density & finance-aware section coverage
- Writes per-file CSV and a summary JSON with aggregate stats and a composite score

Usage (PowerShell/CMD):
  python -m fingpt.stock_chart_trends_analysis.eval.doc_parsing_no_gt_eval ^
    --pdfs "E:/FinGPT-M/fingpt/stock_chart_trends_analysis/eval/docs/pdfs" ^
    --out  "eval_runs/doc_parsing_no_gt"
"""

import os, re, csv, json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# --- Robust IO helpers ---
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def ts_dir(base: Path, run_id: Optional[str] = None) -> Path:
    rid = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = base / rid
    out.mkdir(parents=True, exist_ok=True)
    return out

def safe_write_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)

def safe_write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        safe_write_text(path, "")
        return
    keys = sorted(rows[0].keys())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)

# --- Imports from your repo / baseline ---
import sys
repo_root = Path(__file__).resolve().parents[3]  # .../FinGPT-M
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# your extractor
from fingpt.stock_chart_trends_analysis.PDFExtractor import PdfExtractor

# baseline extractor: pdfminer.six
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None  # allow running without baseline

# --- Text metrics ---
def normalize_text(s: str) -> str:
    return " ".join((s or "").replace("\r", " ").split())

def char_stats(gt: str, pred: str) -> Tuple[int, int, int]:
    gt_n = len(gt); pred_n = len(pred)
    # small DP LCS; fallback to n-gram overlap for large strings
    if gt_n * pred_n <= 2_000_000:
        dp = [0] * (pred_n + 1)
        for i in range(1, gt_n + 1):
            prev = 0
            gi = gt[i-1]
            for j in range(1, pred_n + 1):
                cur = dp[j]
                if gi == pred[j-1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j-1])
                prev = cur
        lcs = dp[-1]
    else:
        def grams(s, k=5):
            return set(s[i:i+k] for i in range(max(0, len(s)-k+1)))
        g1, g2 = grams(gt), grams(pred)
        lcs = len(g1 & g2)
        gt_n = max(len(g1), 1)
        pred_n = max(len(g2), 1)
    return gt_n, pred_n, lcs

def char_prf(a: str, b: str) -> Tuple[float, float, float]:
    a = normalize_text(a); b = normalize_text(b)
    an, bn, inter = char_stats(a, b)
    prec = 0.0 if bn == 0 else inter / bn
    rec  = 0.0 if an == 0 else inter / an
    f1   = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    return prec, rec, f1

def token_set(s: str) -> set:
    toks = re.findall(r"[A-Za-z0-9$%.,:/\-]+", (s or "").lower())
    return set(toks)

def jaccard(a: str, b: str) -> float:
    A, B = token_set(a), token_set(b)
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    return len(A & B) / len(A | B)

def lev_ratio(a: str, b: str) -> float:
    a = normalize_text(a); b = normalize_text(b)
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    n, m = len(a), len(b)
    if n*m > 2_000_000:
        # bigrams proxy
        def grams(s, k=2): return set(s[i:i+k] for i in range(max(0, len(s)-k+1)))
        g1, g2 = grams(a), grams(b)
        if not g1 and not g2: return 1.0
        if not g1 or not g2:  return 0.0
        return len(g1 & g2) / len(g1 | g2)
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev = dp[0]; dp[0] = i
        ai = a[i-1]
        for j in range(1, m+1):
            cur = dp[j]
            cost = 0 if ai == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    dist = dp[m]
    return 1.0 - (dist / max(n, m))

# --- Finance-aware section coverage ---
SECTION_PATTERNS = [
    r"\bmanagement\s+discussion\b",
    r"\bmanagement\s+discussion\s+and\s+analysis\b",
    r"\bboard\s+report\b",
    r"\bfinancial\s+statements?\b",
    r"\bbalance\s+sheet\b",
    r"\bstatement\s+of\s+profit\b",
    r"\bprofit\s+and\s+loss\b",
    r"\bcash\s+flow\b",
    r"\bnotes\s+to\s+accounts?\b",
    r"\baccounting\s+policies\b",
    r"\brisk\s+factors?\b",
    r"\bcorporate\s+governance\b",
    r"\bauditor'?s?\s+report\b",
]

def section_hit_rate(text: str) -> float:
    t = (text or "").lower()
    hits = 0
    for pat in SECTION_PATTERNS:
        if re.search(pat, t):
            hits += 1
    return hits / len(SECTION_PATTERNS)

# --- Health/density metrics ---
def text_health(text: str) -> Dict[str, float]:
    t = text or ""
    n_chars = len(t)
    lines = [ln.strip() for ln in t.splitlines()]
    n_lines = len(lines)
    dup_lines = 0
    seen = set()
    for ln in lines:
        if not ln: continue
        if ln in seen:
            dup_lines += 1
        else:
            seen.add(ln)
    dup_rate = 0.0 if n_lines == 0 else dup_lines / n_lines
    non_ascii = sum(1 for ch in t if ord(ch) > 127)
    non_ascii_rate = 0.0 if n_chars == 0 else non_ascii / n_chars
    digits = sum(1 for ch in t if ch.isdigit())
    letters = sum(1 for ch in t if ch.isalpha())
    digit_letter_ratio = 0.0 if (letters == 0) else digits / letters
    currency_hits = len(re.findall(r"₹|\$|€|£|USD|INR|EUR|GBP", t))
    n_tokens = max(1, len(re.findall(r"[^\s]+", t)))
    currency_rate = currency_hits / n_tokens
    return dict(
        chars=n_chars,
        lines=n_lines,
        dup_line_rate=dup_rate,
        non_ascii_rate=non_ascii_rate,
        digit_letter_ratio=digit_letter_ratio,
        currency_token_rate=currency_rate
    )

# --- Page count (cheap) from pdfminer if available ---
def pdf_pages_via_pdfminer(path: Path) -> int:
    try:
        # quick & dirty: pdfminer layout isn’t cheap; skip full layout, just sniff pages
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfpage import PDFPage
        with open(path, "rb") as f:
            parser = PDFParser(f)
            doc = PDFDocument(parser)
            return sum(1 for _ in PDFPage.create_pages(doc))
    except Exception:
        return 0

# --- One-file evaluation ---
@dataclass
class Row:
    file_id: str
    pages: int
    chars: int
    chars_per_page: float
    empty_page_rate: float
    dup_line_rate: float
    non_ascii_rate: float
    digit_letter_ratio: float
    currency_token_rate: float
    section_hit_rate: float
    agree_char_f1: float
    agree_jaccard: float
    agree_levratio: float
    meta_fill_rate: float
    composite_score: float
    notes: str

META_KEYS = ["Title","Author","Subject","Creator","Producer","CreationDate","ModDate"]

def meta_fill_rate(meta: Dict[str, Any]) -> float:
    if not meta: return 0.0
    filled = 0; total = 0
    for k in META_KEYS:
        total += 1
        v = meta.get(k)
        if v and str(v).strip():
            filled += 1
    return filled / total if total else 0.0

def eval_one(pdf_path: Path, header_pages: int = 2) -> Row:
    # 1) your extractor
    px = PdfExtractor()
    full_text, first_text, meta = px.extract_text_and_header_from_path(str(pdf_path), header_pages=header_pages)
    full_text = full_text or ""

    # 2) baseline via pdfminer (if available)
    if pdfminer_extract_text is not None:
        baseline_text = pdfminer_extract_text(str(pdf_path)) or ""
    else:
        baseline_text = ""

    # 3) agreement (weak supervision)
    cprec, crec, cf1 = char_prf(full_text, baseline_text)
    jac  = jaccard(full_text, baseline_text)
    lrat = lev_ratio(full_text, baseline_text)

    # 4) health/density
    pages = pdf_pages_via_pdfminer(pdf_path)
    th = text_health(full_text)
    chars = th["chars"]
    cpp = 0.0 if pages == 0 else chars / pages

    # empty page rate not directly known; rough proxy by counting page delimiters in text
    # (you can wire a real page-by-page loop inside PdfExtractor to compute accurately)
    empty_rate = 0.0  # keep 0.0 unless you track per-page text lengths

    # 5) section coverage
    sect = section_hit_rate(full_text)

    # 6) metadata presence
    mfill = meta_fill_rate(meta or {})

    # 7) composite (weights are configurable; here is a sensible default)
    # Emphasize agreement + section coverage + density quality, penalize high dup/non-ascii
    composite = (
        0.25 * cf1 +
        0.15 * jac +
        0.10 * lrat +
        0.20 * sect +
        0.10 * (1.0 - th["dup_line_rate"]) +
        0.05 * (1.0 - th["non_ascii_rate"]) +
        0.05 * min(1.0, th["currency_token_rate"] * 10.0) +  # rescale
        0.10 * mfill
    )
    composite = max(0.0, min(1.0, composite))

    return Row(
        file_id=pdf_path.name,
        pages=pages,
        chars=chars,
        chars_per_page=cpp,
        empty_page_rate=empty_rate,
        dup_line_rate=th["dup_line_rate"],
        non_ascii_rate=th["non_ascii_rate"],
        digit_letter_ratio=th["digit_letter_ratio"],
        currency_token_rate=th["currency_token_rate"],
        section_hit_rate=sect,
        agree_char_f1=cf1,
        agree_jaccard=jac,
        agree_levratio=lrat,
        meta_fill_rate=mfill,
        composite_score=composite,
        notes="" if baseline_text else "baseline_missing"
    )

# --- Driver ---
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", required=True, help="Folder with PDFs")
    ap.add_argument("--out", required=True, help="Output folder (a timestamped run dir will be created)")
    ap.add_argument("--header_pages", type=int, default=2)
    args = ap.parse_args()

    pdf_dir = Path(args.pdfs)
    out_base = Path(args.out)
    outdir = ts_dir(out_base)

    pdfs = sorted([p for p in pdf_dir.glob("**/*") if p.suffix.lower() == ".pdf"])
    if not pdfs:
        print(f"[WARN] No PDFs found in {pdf_dir}")
        return

    rows: List[Dict[str, Any]] = []
    for p in pdfs:
        try:
            r = eval_one(p, header_pages=args.header_pages)
            rows.append(asdict(r))
            print(f"[OK] {p.name}")
        except Exception as e:
            rows.append(asdict(Row(
                file_id=p.name, pages=0, chars=0, chars_per_page=0.0, empty_page_rate=1.0,
                dup_line_rate=1.0, non_ascii_rate=1.0, digit_letter_ratio=0.0, currency_token_rate=0.0,
                section_hit_rate=0.0, agree_char_f1=0.0, agree_jaccard=0.0, agree_levratio=0.0,
                meta_fill_rate=0.0, composite_score=0.0, notes=f"error:{e}"
            )))
            print(f"[ERR] {p.name}: {e}")

    # Save artifacts
    per_file_csv = outdir / "per_file.csv"
    safe_write_csv(per_file_csv, rows)

    # Summary
    import numpy as np
    def mean(key): return float(np.mean([r[key] for r in rows])) if rows else 0.0
    summary = {
        "files": len(rows),
        "avg_pages": mean("pages"),
        "avg_chars": mean("chars"),
        "avg_chars_per_page": mean("chars_per_page"),
        "avg_empty_page_rate": mean("empty_page_rate"),
        "avg_dup_line_rate": mean("dup_line_rate"),
        "avg_non_ascii_rate": mean("non_ascii_rate"),
        "avg_digit_letter_ratio": mean("digit_letter_ratio"),
        "avg_currency_token_rate": mean("currency_token_rate"),
        "avg_section_hit_rate": mean("section_hit_rate"),
        "avg_agree_char_f1": mean("agree_char_f1"),
        "avg_agree_jaccard": mean("agree_jaccard"),
        "avg_agree_levratio": mean("agree_levratio"),
        "avg_meta_fill_rate": mean("meta_fill_rate"),
        "avg_composite_score": mean("composite_score"),
        "outdir": str(outdir)
    }
    safe_write_text(outdir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Document Parsing (No-GT) Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()