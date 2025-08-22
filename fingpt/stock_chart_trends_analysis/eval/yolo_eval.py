import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Please install Ultralytics: pip install ultralytics") from e


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_val(model_path: str,
            data_yaml: Optional[str],
            val_images: Optional[str],
            val_labels: Optional[str],
            imgsz: int,
            conf: float,
            iou: float,
            device: str,
            batch: int,
            outdir: Path,
            save_json: bool,
            save_plots: bool):
    """
    Runs Ultralytics .val() and returns (model, det_metrics, names_dict).
    For recent Ultralytics versions, .val() returns a DetMetrics directly.
    """
    model = YOLO(model_path)

    # class names on the model (dict[int, str])
    names = {}
    try:
        names = model.model.names
    except Exception:
        pass

    args = dict(
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        batch=batch,
        save_json=save_json,
        save_hybrid=False,
        plots=save_plots,
        project=str(outdir),
        name="val",
        verbose=False,
    )

    if data_yaml:
        det_metrics = model.val(data=data_yaml, **args)
    else:
        # Build temporary data yaml if only split dirs are given
        tmp_yaml = outdir / "_tmp_data.yaml"
        ds = {"path": ".", "names": names, "val": val_images}
        import yaml
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(ds, f)
        det_metrics = model.val(data=str(tmp_yaml), **args)

    return model, det_metrics, names


def extract_metrics(det_metrics, names: Dict[int, str]) -> Dict:
    """
    Extract standard metrics from Ultralytics DetMetrics.
    """
    m = det_metrics  # alias

    # m.box has: map, map50, map75, maps (list per class), mp (mean precision), mr (mean recall)
    per_class = []
    maps = getattr(m.box, "maps", None)
    if maps is not None:
        for cls_id, ap in enumerate(maps):
            per_class.append({
                "class_id": int(cls_id),
                "class_name": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id),
                "ap": float(ap)
            })

    metrics = {
        "map": float(getattr(m.box, "map", float("nan"))),
        "map50": float(getattr(m.box, "map50", float("nan"))),
        "map75": float(getattr(m.box, "map75", float("nan"))),
        "per_class_ap": per_class,
        "precision": float(getattr(m.box, "mp", float("nan"))),
        "recall": float(getattr(m.box, "mr", float("nan"))),
        "speed": getattr(m, "speed", {}),  # ms/img dict: preprocess, inference, postprocess
        "nc": len(names) if isinstance(names, dict) else None,
    }
    return metrics


def write_per_class_csv(det_metrics, names: Dict[int, str], out_csv: Path):
    """
    Writes per-class AP@[.50:.95] to CSV (Ultralytics DetMetrics).
    """
    maps = getattr(det_metrics.box, "maps", None)
    if maps is None:
        return

    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "ap_[.50:.95]"])
        for cls_id, ap in enumerate(maps):
            cname = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            w.writerow([cls_id, cname, float(ap)])


def collect_slice_files(root_img_dir: Path, patterns: List[str]) -> List[Path]:
    """
    Collect images in root_img_dir that match any of the patterns (glob or regex).
    """
    files: List[Path] = []
    all_imgs = [p for p in root_img_dir.glob("**/*") if p.is_file()]
    for p in patterns:
        is_regex = any(ch in p for ch in "[]().*+?|^$\\")
        if is_regex:
            rx = re.compile(p)
            files.extend([f for f in all_imgs if rx.search(str(f))])
        else:
            files.extend(list(root_img_dir.glob(p)))
    # unique
    seen = set()
    unique: List[Path] = []
    for f in files:
        if f not in seen and f.is_file():
            unique.append(f)
            seen.add(f)
    return unique


@torch.inference_mode()
def probe_latency(model, image_dir: Path, imgsz: int, batch: int,
                  warmup_batches: int, timed_batches: int, device="cpu") -> Dict:
    """
    Simple throughput/latency probe using raw predict() on a directory of images.
    """
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in image_dir.glob("**/*") if p.suffix.lower() in suffixes]
    if not imgs:
        return {"error": f"No images found under {image_dir}"}

    # Cycle through images
    def image_stream():
        i = 0
        L = len(imgs)
        while True:
            yield str(imgs[i % L])
            i += 1

    stream = image_stream()

    # warmup
    for _ in range(warmup_batches):
        batch_paths = [next(stream) for _ in range(batch)]
        _ = model.predict(batch_paths, imgsz=imgsz, device=device, verbose=False)

    # timed
    t0 = time.perf_counter()
    for _ in range(timed_batches):
        batch_paths = [next(stream) for _ in range(batch)]
        _ = model.predict(batch_paths, imgsz=imgsz, device=device, verbose=False)
    t1 = time.perf_counter()

    total_images = timed_batches * batch
    elapsed = t1 - t0
    ips = total_images / max(1e-9, elapsed)
    out = {
        "timed_batches": timed_batches,
        "batch_size": batch,
        "total_images": total_images,
        "elapsed_sec": elapsed,
        "images_per_sec": ips,
        "avg_latency_ms_per_image": 1000.0 * elapsed / total_images
    }

    if torch.cuda.is_available() and str(device) != "cpu":
        torch.cuda.synchronize()
        out["max_cuda_mem_mb"] = round(torch.cuda.max_memory_allocated() / (1024**2), 2)
        torch.cuda.reset_peak_memory_stats()

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to eval/configs/yolo.yaml")
    ap.add_argument("--outdir", type=str, default=None, help="Override output dir")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    outdir = ensure_dir(Path(args.outdir) if args.outdir else Path(cfg.get("outdir", "eval_runs/yolo")))
    save_json_flag = bool(cfg.get("save_json", True))
    save_plots = bool(cfg.get("save_plots", True))

    dataset_yaml = cfg.get("dataset_yaml") or None
    splits = cfg.get("splits", {})
    val_images = splits.get("val_images")
    val_labels = splits.get("val_labels")  # currently unused; Ultralytics finds labels by convention

    model_path = cfg["weights"]
    imgsz = int(cfg.get("imgsz", 640))
    conf = float(cfg.get("conf", 0.25))
    iou = float(cfg.get("iou", 0.5))
    device = str(cfg.get("device", 0))
    batch = int(cfg.get("batch", 16))

    # 1) Standard validation
    model, det_metrics, names = run_val(
        model_path=model_path,
        data_yaml=dataset_yaml,
        val_images=val_images,
        val_labels=val_labels,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        batch=batch,
        outdir=outdir,
        save_json=save_json_flag,
        save_plots=save_plots,
    )

    metrics = extract_metrics(det_metrics, names)
    save_json(metrics, outdir / "metrics_summary.json")
    write_per_class_csv(det_metrics, names, outdir / "per_class_ap.csv")

    # 2) Latency / throughput on the validation images folder if available
    latency_cfg = cfg.get("latency", {})
    warmup_batches = int(latency_cfg.get("warmup_batches", 5))
    timed_batches = int(latency_cfg.get("timed_batches", 20))

    # Decide the image directory to probe
    probe_dir = None
    if dataset_yaml:
        # Try to parse the dataset yaml to find val path
        try:
            ds_yaml = load_yaml(Path(dataset_yaml))
            vp = ds_yaml.get("val", None)
            if vp:
                probe_dir = Path(vp)
        except Exception:
            pass
    if not probe_dir and val_images:
        probe_dir = Path(val_images)

    latency_report = {}
    if probe_dir and probe_dir.exists():
        latency_report = probe_latency(
            model=model,
            image_dir=probe_dir,
            imgsz=imgsz,
            batch=batch,
            warmup_batches=warmup_batches,
            timed_batches=timed_batches,
            device=device
        )
        save_json(latency_report, outdir / "latency.json")

    # 3) Robustness slices
    slices = cfg.get("slices", []) or []
    slice_reports = []
    if probe_dir and probe_dir.exists():
        for s in slices:
            sname = s.get("name", "slice")
            include = s.get("include", [])
            if not include:
                continue
            files = []
            for pat in include:
                files.extend(collect_slice_files(probe_dir, [pat]))
            files = sorted(set(files))
            if not files:
                slice_reports.append({"name": sname, "images": 0, "note": "no matches"})
                continue

            t0 = time.perf_counter()
            preds = model.predict([str(p) for p in files], imgsz=imgsz, device=device, verbose=False)
            t1 = time.perf_counter()
            confs = []
            for pr in preds:
                if pr is None or pr.boxes is None:
                    continue
                if hasattr(pr.boxes, "conf") and pr.boxes.conf is not None:
                    confs.extend(pr.boxes.conf.detach().cpu().numpy().tolist())
            slice_reports.append({
                "name": sname,
                "images": len(files),
                "avg_confidence": float(np.mean(confs)) if confs else None,
                "elapsed_sec": (t1 - t0)
            })

    if slice_reports:
        save_json(slice_reports, outdir / "robustness_slices.json")

    # 4) Console summary
    print("\n=== YOLO Evaluation Summary ===")
    print(json.dumps({
        "mAP@.50:.95": metrics.get("map"),
        "mAP@.50": metrics.get("map50"),
        "mAP@.75": metrics.get("map75"),
        "mean_precision": metrics.get("precision"),
        "mean_recall": metrics.get("recall"),
    }, indent=2))


if __name__ == "__main__":
    main()