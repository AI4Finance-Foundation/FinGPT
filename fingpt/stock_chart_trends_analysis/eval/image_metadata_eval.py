import argparse
import csv
import json
import os
import re
import time
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Optional finnhub lookup (only used if try_resolve_ticker:true)
try:
    import finnhub  # pip install finnhub-python
except Exception:
    finnhub = None


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_unique_run_dir(base_outdir: Path, run_id: Optional[str] = None) -> Path:
    """
    Create a unique subfolder under base_outdir (e.g., eval_runs/.../20250822-154210).
    Never deletes older runs -> avoids locked-file headaches on Windows.
    """
    ensure_dir(base_outdir)
    rid = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = base_outdir / rid
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def safe_write_text(path: Path, text: str, retries: int = 6, backoff: float = 0.25):
    """
    Write text atomically with retries:
    - write to temp file next to target
    - os.replace() to avoid partial writes
    Retries handle Windows file locks (Explorer/Excel/AV).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    for i in range(retries):
        try:
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            os.replace(tmp, path)  # atomic on Windows/NTFS
            return
        except PermissionError:
            time.sleep(backoff * (i + 1))
        except Exception:
            # try to cleanup tmp then rethrow on last attempt
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
            if i == retries - 1:
                raise
    # if we somehow got here without returning, raise
    raise PermissionError(f"Could not write to {path}")


def safe_write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str],
                   retries: int = 6, backoff: float = 0.25):
    """
    CSV writer with a temp file + os.replace() and retries.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    for i in range(retries):
        try:
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(backoff * (i + 1))
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
            if i == retries - 1:
                raise
    raise PermissionError(f"Could not write CSV to {path}")


def is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def pct_close(a: float, b: float, pct: float) -> bool:
    base = max(abs(b), 1e-12)
    return abs(a - b) <= pct * base


def normalize_exchange(x: Optional[str]) -> str:
    return (x or "").strip().upper()


def normalize_ticker(x: Optional[str]) -> str:
    return (x or "").strip().upper()


def valid_time_token(t: str) -> bool:
    # Matches 1:05 AM, 11:30 PM, 4:00 AM, etc.
    return bool(re.fullmatch(r"(0?[1-9]|1[0-2]):[0-5][0-9]\s?(AM|PM)", t.strip().upper()))


@dataclass
class SampleReport:
    file: str
    ok: bool
    company_name_ok: bool
    ticker_ok: bool
    exchange_ok: bool
    ohlc_ok: bool
    price_range_ok: bool
    sessions_ok: bool
    dates_ok: bool
    notes: str


def try_lookup_ticker(company_name: str, finnhub_api_key: Optional[str]) -> Optional[str]:
    if not company_name or not finnhub or not finnhub_api_key:
        return None
    try:
        client = finnhub.Client(api_key=finnhub_api_key)
        res = client.symbol_lookup(company_name)
        items = (res or {}).get("result", [])
        if not items:
            return None
        def rank(item):
            exch = (item.get("exchange") or "").upper()
            sym = (item.get("symbol") or "")
            s = 0
            if exch in {"NSE","BSE","NYSE","NASDAQ","NYSE ARCA","NYSE MKT"}:
                s += 5
            s += max(0, 4 - len(sym))
            return s
        best = sorted(items, key=rank, reverse=True)[0]
        return (best.get("symbol") or "").upper()
    except Exception:
        return None


def validate_one(sample: Dict[str, Any], cfg: dict) -> SampleReport:
    notes = []

    # company_name
    company_name = (sample.get("company_name") or "").strip()
    company_name_ok = bool(company_name)
    if not company_name_ok:
        notes.append("missing company_name")

    # ticker
    ticker = normalize_ticker(sample.get("ticker"))
    allow_na_ticker = bool(cfg.get("allow_na_ticker", False))
    ticker_ok = bool(ticker and ticker != "N/A")
    if not ticker_ok:
        if not allow_na_ticker:
            notes.append("ticker is empty or N/A")
        if bool(cfg.get("try_resolve_ticker", False)):
            api_key = os.getenv(str(cfg.get("finnhub_api_key_env", "FINNHUB_API_KEY")))
            resolved = try_lookup_ticker(company_name, api_key)
            if resolved:
                ticker = resolved
                ticker_ok = True
                notes.append(f"ticker resolved -> {resolved}")

    # exchange
    exch = normalize_exchange(sample.get("exchange"))
    allowed = [x.upper() for x in cfg.get("require_exchange_in", [])]
    exchange_ok = (not allowed) or (exch in allowed)
    if not exchange_ok:
        notes.append(f"exchange '{exch}' not in {allowed}")

    # ohlc
    ohlc = sample.get("ohlc") or {}
    has_all = all(k in ohlc for k in ("O","H","L","C","V"))
    numeric = all(is_number(ohlc.get(k)) for k in ("O","H","L","C","V")) if has_all else False
    ohlc_ok = bool(has_all and numeric)
    if not ohlc_ok:
        notes.append("OHLC missing or non-numeric")
    else:
        O = float(ohlc["O"]); H = float(ohlc["H"]); L = float(ohlc["L"]); C = float(ohlc["C"]); V = float(ohlc["V"])
        if cfg.get("ohlc_non_negative", True):
            if min(O,H,L,C) < 0 or V < 0:
                ohlc_ok = False
                notes.append("OHLC has negative values")
        if cfg.get("volume_non_negative", True) and V < 0:
            ohlc_ok = False
            notes.append("Volume negative")
        if cfg.get("ohlc_monotone_check", True):
            lo, hi = min(O, C, L), max(O, C, H)
            if not (L <= lo and hi <= H):
                notes.append("OHLC monotonicity soft check failed (L..H vs O/C)")

    # price_range
    pr = sample.get("price_range") or [None, None]
    price_range_ok = False
    if isinstance(pr, list) and len(pr) >= 2 and is_number(pr[0]) and is_number(pr[1]):
        low, high = float(pr[0]), float(pr[1])
        if low <= high:
            price_range_ok = True
            if cfg.get("check_price_range_vs_ohlc", True) and ohlc_ok:
                tol = float(cfg.get("price_range_tolerance_pct", 0.01))
                if not (pct_close(low, float(ohlc["L"]), tol) and pct_close(high, float(ohlc["H"]), tol)):
                    price_range_ok = False
                    notes.append("price_range deviates from OHLC L/H beyond tolerance")
    if not price_range_ok:
        notes.append("invalid price_range")

    # sessions
    sessions = sample.get("sessions") or []
    sessions_ok = True
    if cfg.get("require_sessions_date", True):
        for s in sessions:
            if not isinstance(s, dict):
                sessions_ok = False; notes.append("session item not a dict"); break
            if not s.get("date"):
                sessions_ok = False; notes.append("session missing date"); break
            if cfg.get("require_sessions_time_list", True):
                times = s.get("time")
                if not isinstance(times, list) or not all(isinstance(t, str) and valid_time_token(t) for t in times):
                    sessions_ok = False; notes.append("session time list invalid"); break

    # dates
    dates = sample.get("dates")
    dates_ok = True
    if cfg.get("require_dates_list", False):
        if not isinstance(dates, list) or not dates:
            dates_ok = False; notes.append("dates missing or not list")
        else:
            if any((not isinstance(d, str) or not d.strip()) for d in dates):
                dates_ok = False; notes.append("dates list has empty items")

    ok = company_name_ok and (ticker_ok or allow_na_ticker) and exchange_ok and ohlc_ok and price_range_ok and sessions_ok and dates_ok
    return SampleReport(
        file=sample.get("__file__", "<in-memory>"),
        ok=ok,
        company_name_ok=company_name_ok,
        ticker_ok=(ticker_ok or allow_na_ticker),
        exchange_ok=exchange_ok,
        ohlc_ok=ohlc_ok,
        price_range_ok=price_range_ok,
        sessions_ok=sessions_ok,
        dates_ok=dates_ok,
        notes="; ".join(notes)
    )


def load_samples(pred_dir: Path) -> List[Dict[str, Any]]:
    if pred_dir.is_file():
        obj = json.loads(pred_dir.read_text(encoding="utf-8"))
        obj["__file__"] = str(pred_dir)
        return [obj]
    items = []
    for p in pred_dir.glob("**/*.json"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            j["__file__"] = str(p)
            items.append(j)
        except Exception:
            pass
    return items


def run_eval(cfg: dict):
    pred_dir = Path(cfg["pred_dir"])

    # Outdir handling - use timestamped subdirs unless disabled
    base_outdir = Path(cfg.get("outdir", "eval_runs/metadata_basic"))
    use_timestamped_dirs = cfg.get("use_timestamped_dirs", True)
    
    if use_timestamped_dirs:
        run_id = cfg.get("run_id")  # optional; else timestamp
        outdir = ensure_unique_run_dir(base_outdir, run_id=run_id)
    else:
        # Use directory directly, with optional cleanup
        cleanup_outdir = cfg.get("cleanup_outdir", False)
        if cleanup_outdir and base_outdir.exists():
            shutil.rmtree(base_outdir)
        outdir = ensure_dir(base_outdir)

    samples = load_samples(pred_dir)
    reports: List[SampleReport] = []
    for s in samples:
        reports.append(validate_one(s, cfg))

    # write per-sample CSV (safe writer)
    rows = [asdict(r) for r in reports]
    per_sample_csv = outdir / cfg.get("per_sample_csv", "per_sample.csv")
    if rows:
        keys = sorted(rows[0].keys())
        safe_write_csv(per_sample_csv, rows, keys)

    # summary
    def mean_bool(field):
        if not reports:
            return 0.0
        return float(np.mean([getattr(r, field) for r in reports]))

    summary = {
        "samples": len(reports),
        "pass_rate": mean_bool("ok"),
        "company_name_ok_rate": mean_bool("company_name_ok"),
        "ticker_ok_rate": mean_bool("ticker_ok"),
        "exchange_ok_rate": mean_bool("exchange_ok"),
        "ohlc_ok_rate": mean_bool("ohlc_ok"),
        "price_range_ok_rate": mean_bool("price_range_ok"),
        "sessions_ok_rate": mean_bool("sessions_ok"),
        "dates_ok_rate": mean_bool("dates_ok"),
        "config": cfg,
        "outdir": str(outdir)
    }
    summary_json = outdir / cfg.get("summary_json", "summary.json")
    safe_write_text(summary_json, json.dumps(summary, indent=2))

    # write failed cases CSV if specified
    errors_csv_name = cfg.get("errors_csv")
    if errors_csv_name:
        failed_reports = [r for r in reports if not r.ok]
        if failed_reports:
            failed_rows = [asdict(r) for r in failed_reports]
            errors_csv = outdir / errors_csv_name
            keys = sorted(failed_rows[0].keys())
            safe_write_csv(errors_csv, failed_rows, keys)
        else:
            # Create empty file if no failures
            errors_csv = outdir / errors_csv_name
            safe_write_text(errors_csv, "")

    print("\n=== Metadata Basic Validation Summary ===")
    print(json.dumps(summary, indent=2))

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to eval/configs/metadata_basic.yaml")
    args = ap.parse_args()
    cfg = load_yaml(Path(args.config))
    run_eval(cfg)

if __name__ == "__main__":
    main()