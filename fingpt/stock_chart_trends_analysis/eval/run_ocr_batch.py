"""
Batch OCR runner for chart images.

- Reads images from --images
- Uses StockChartMetadataExtractor from OCR_Script_for_charts.py to extract METADATA ONLY
- Writes one JSON per image to --out
- Ensures `image_id` is present in every JSON

Usage (Windows):
  python -m fingpt.stock_chart_trends_analysis.eval.run_ocr_batch ^
    --images "E:\FinGPT-M\fingpt\stock_chart_trends_analysis\eval\preds\images" ^
    --out    "E:\FinGPT-M\fingpt\stock_chart_trends_analysis\eval\preds\json"
"""

import os
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure repo root is importable if running from elsewhere
# # (adjust if your working dir already has the package available)
# repo_root = Path(__file__).resolve().parents[3]  # ...\FinGPT-M
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import the metadata extractor class
metadata_extractor_class = None
try:
    from fingpt.stock_chart_trends_analysis.OCR_Script_for_charts import StockChartMetadataExtractor
    metadata_extractor_class = StockChartMetadataExtractor
except Exception as e:
    print(f"Failed to import StockChartMetadataExtractor: {e}")
    metadata_extractor_class = None


def extract_metadata_only(image_path: Path) -> Dict[str, Any]:
    """
    Returns ONLY metadata (no predictions), using StockChartMetadataExtractor.
    """
    # Use the metadata extractor class
    if metadata_extractor_class is not None:
        mex = metadata_extractor_class(str(image_path))
        meta = mex.extract_metadata()
        if not isinstance(meta, dict):
            raise RuntimeError("StockChartMetadataExtractor.extract_metadata() did not return a dict")
        return meta

    raise ImportError(
        "Could not import StockChartMetadataExtractor from OCR_Script_for_charts. "
        "Please ensure your repo imports are correct."
    )


def keep_metadata_fields(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Whittle down to strictly metadata keys we care about.
    (Drop predictions, boxes, etc.)
    """
    out = {}
    if "company_name" in meta: out["company_name"] = meta["company_name"]
    if "ticker" in meta: out["ticker"] = meta["ticker"]
    if "exchange" in meta: out["exchange"] = meta["exchange"]
    if "ohlc" in meta: out["ohlc"] = meta["ohlc"]
    if "price_range" in meta: out["price_range"] = meta["price_range"]
    if "sessions" in meta: out["sessions"] = meta["sessions"]
    if "dates" in meta: out["dates"] = meta["dates"]
    # Accept alternate key names if OCR output differs
    if "date" in meta and "dates" not in out:
        out["dates"] = [meta["date"]] if isinstance(meta["date"], str) else meta["date"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder with chart images")
    ap.add_argument("--out", required=True, help="Folder to write metadata JSONs")
    args = ap.parse_args()

    images_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    imgs = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in exts])

    if not imgs:
        print(f"[WARN] No images found in {images_dir}")
        return

    for img in imgs:
        try:
            meta_full = extract_metadata_only(img)
            meta = keep_metadata_fields(meta_full)

            # Always add image_id for evaluation alignment
            meta["image_id"] = img.name
            meta["source"] = "ocr_only"

            out_path = out_dir / f"{img.stem}.json"
            out_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[OK] {img.name} -> {out_path.name}")
        except Exception as e:
            print(f"[ERR] {img.name}: {e}")


if __name__ == "__main__":
    main()