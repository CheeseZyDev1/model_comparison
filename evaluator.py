from __future__ import annotations

import csv
import html
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def validate_ground_truth(data: dict) -> None:
    required = {"images", "annotations", "categories"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Ground truth is not COCO JSON; missing: {sorted(missing)}")


def validate_predictions(rows: list[dict], path: Path) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"{path}: predictions must be a JSON list")
    required = {"image_id", "category_id", "bbox", "score"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing or not isinstance(row.get("bbox"), list) or len(row.get("bbox", [])) != 4:
            raise ValueError(f"{path}: invalid detection at index {index}; missing={sorted(missing)}")
        if not 0.0 <= float(row["score"]) <= 1.0:
            raise ValueError(f"{path}: score at index {index} must be between 0 and 1")
        if float(row["bbox"][2]) < 0 or float(row["bbox"][3]) < 0:
            raise ValueError(f"{path}: bbox width/height at index {index} cannot be negative")


def validate_compatibility(gt: dict, rows: list[dict], path: Path) -> None:
    image_ids = {int(item["id"]) for item in gt["images"]}
    category_ids = {int(item["id"]) for item in gt["categories"]}
    unknown_images = {int(row["image_id"]) for row in rows} - image_ids
    unknown_categories = {int(row["category_id"]) for row in rows} - category_ids
    if unknown_images or unknown_categories:
        raise ValueError(
            f"{path}: IDs do not match the selected ground truth. "
            f"Unknown image_id examples={sorted(unknown_images)[:5]}, "
            f"unknown category_id={sorted(unknown_categories)}"
        )


def discover_models(config_path: str | Path) -> tuple[dict, Path, list[dict]]:
    config_path = Path(config_path).resolve()
    base = config_path.parent
    config = load_json(config_path)
    gt_path = resolve(base, config["ground_truth"])
    if not gt_path.is_file():
        raise FileNotFoundError(
            f"Ground truth not found: {gt_path}\n"
            "Edit ground_truth in config.json to point to the shared COCO test annotation."
        )

    found: dict[Path, dict] = {}
    explicit_runs = config.get("model_runs", [])
    if explicit_runs:
        model_root = resolve(base, config.get("model_root", "../xrayfsod_outputs"))
        missing = []
        for entry in explicit_runs:
            item = {"run_id": entry} if isinstance(entry, str) else dict(entry)
            run_id = str(item["run_id"])
            run_dir = model_root / run_id
            pred_path = run_dir / item.get("prediction_file", "test_predictions.json")
            weight_path = run_dir / item.get("weight_file", "weights/best.pt")
            if not pred_path.is_file():
                weight_status = f"พบ weight: {weight_path}" if weight_path.is_file() else f"ไม่พบ weight: {weight_path}"
                missing.append(f"{run_id}: ไม่พบ {pred_path.name} ({weight_status})")
                continue
            manifest_path = run_dir / "model.json"
            manifest = load_json(manifest_path) if manifest_path.is_file() else {}
            if weight_path.is_file():
                manifest.setdefault("weight_file", str(weight_path))
            pred_path = pred_path.resolve()
            found[pred_path] = {
                "name": item.get("display_name", manifest.get("display_name", run_id)),
                "run_id": run_id,
                "prediction_path": pred_path,
                "weight_path": weight_path.resolve() if weight_path.is_file() else None,
                "manifest": manifest,
            }
        if missing:
            details = "\n  - ".join(missing)
            raise FileNotFoundError(
                "โฟลเดอร์โมเดลถูกพบได้ แต่ยังขาดผลทำนายที่ใช้คำนวณ metric:\n"
                f"  - {details}\n"
                "ไฟล์ best.pt ของแต่ละ framework โหลดต่างกัน จึงต้อง export ผลบน test set "
                "เป็น test_predictions.json ก่อน (notebook เทรน XrayFSOD สร้างไฟล์นี้ให้อัตโนมัติ)"
            )
    else:
        for pattern in config.get("prediction_globs", []):
            pattern_path = Path(pattern).expanduser()
            root = pattern_path.anchor if pattern_path.is_absolute() else str(base)
            relative_pattern = str(pattern_path)[len(pattern_path.anchor):].lstrip("\\/") if pattern_path.is_absolute() else pattern
            for pred_path in Path(root).glob(relative_pattern):
                pred_path = pred_path.resolve()
                if not pred_path.is_file():
                    continue
                manifest_path = pred_path.parent / "model.json"
                manifest = load_json(manifest_path) if manifest_path.is_file() else {}
                default_name = pred_path.parent.name
                weight_path = pred_path.parent / "weights" / "best.pt"
                found[pred_path] = {
                    "name": manifest.get("display_name", default_name),
                    "run_id": default_name,
                    "prediction_path": pred_path,
                    "weight_path": weight_path.resolve() if weight_path.is_file() else None,
                    "manifest": manifest,
                }

    models = sorted(found.values(), key=lambda item: (item["name"].lower(), str(item["prediction_path"])))
    if not models:
        patterns = "\n".join(f"  - {p}" for p in config.get("prediction_globs", [])) or "  - ไม่มี model_runs หรือ prediction_globs"
        raise FileNotFoundError(f"No prediction files found. Checked:\n{patterns}")
    return config, gt_path, models


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, width, height = map(float, box)
    return x, y, x + max(0.0, width), y + max(0.0, height)


def box_iou(left: list[float], right: list[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(left)
    bx1, by1, bx2, by2 = xywh_to_xyxy(right)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(
    gt: dict,
    predictions: list[dict],
    confidence: float,
    iou_threshold: float,
    max_detections: int,
) -> tuple[dict, list[dict]]:
    image_ids = {int(item["id"]) for item in gt["images"]}
    categories = {int(item["id"]): str(item["name"]) for item in gt["categories"]}
    gt_boxes: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ann in gt["annotations"]:
        key = (int(ann["image_id"]), int(ann["category_id"]))
        gt_boxes[key].append({"bbox": ann["bbox"], "matched": False})

    per_image: dict[int, list[dict]] = defaultdict(list)
    for pred in predictions:
        if float(pred["score"]) >= confidence:
            per_image[int(pred["image_id"])].append(pred)

    filtered = []
    for image_id, rows in per_image.items():
        filtered.extend(sorted(rows, key=lambda row: float(row["score"]), reverse=True)[:max_detections])
    filtered.sort(key=lambda row: float(row["score"]), reverse=True)

    counts = {category_id: {"tp": 0, "fp": 0, "fn": 0} for category_id in categories}
    for pred in filtered:
        image_id, category_id = int(pred["image_id"]), int(pred["category_id"])
        if image_id not in image_ids or category_id not in categories:
            continue
        candidates = gt_boxes.get((image_id, category_id), [])
        best_iou, best_index = 0.0, -1
        for index, candidate in enumerate(candidates):
            if candidate["matched"]:
                continue
            value = box_iou(pred["bbox"], candidate["bbox"])
            if value > best_iou:
                best_iou, best_index = value, index
        if best_index >= 0 and best_iou >= iou_threshold:
            candidates[best_index]["matched"] = True
            counts[category_id]["tp"] += 1
        else:
            counts[category_id]["fp"] += 1

    for (_, category_id), rows in gt_boxes.items():
        counts[category_id]["fn"] += sum(not row["matched"] for row in rows)

    per_class = []
    for category_id, class_name in categories.items():
        tp, fp, fn = (counts[category_id][key] for key in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({
            "category_id": category_id, "class_name": class_name,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        })

    total_tp = sum(row["tp"] for row in per_class)
    total_fp = sum(row["fp"] for row in per_class)
    total_fn = sum(row["fn"] for row in per_class)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": precision, "recall": recall, "f1": f1,
        "macro_f1": float(np.mean([row["f1"] for row in per_class])) if per_class else 0.0,
    }
    return summary, per_class


def coco_metrics(gt_path: Path, predictions: list[dict], max_detections: int) -> dict:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        warnings.warn(
            "pycocotools is not installed; mAP/AP/AR will be NaN. "
            "Run: pip install -r requirements.txt",
            RuntimeWarning,
        )
        return {"mAP_50_95": math.nan, "AP_50": math.nan, "AP_75": math.nan, "AR_100": math.nan}
    if not predictions:
        return {"mAP_50_95": 0.0, "AP_50": 0.0, "AP_75": 0.0, "AR_100": 0.0}
    from contextlib import redirect_stdout
    from io import StringIO
    with redirect_stdout(StringIO()):
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.maxDets = [1, 10, max_detections]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "mAP_50_95": float(evaluator.stats[0]), "AP_50": float(evaluator.stats[1]),
        "AP_75": float(evaluator.stats[2]), "AR_100": float(evaluator.stats[8]),
    }


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value).strip("_")
    return cleaned or "model"


def save_model_plots(model_dir: Path, model_name: str, sweep: pd.DataFrame, per_class: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for column, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
        axis.plot(sweep["confidence"], sweep[column], marker="o", label=label)
    axis.set(xlabel="Confidence threshold", ylabel="Score", ylim=(0, 1.03), title=f"Threshold sweep — {model_name}")
    axis.legend()
    fig.tight_layout(); fig.savefig(model_dir / "threshold_sweep.png", dpi=160); plt.close(fig)

    ordered = per_class.sort_values("f1", ascending=True)
    fig, axis = plt.subplots(figsize=(9, max(4.5, len(ordered) * 0.32)))
    axis.barh(ordered["class_name"], ordered["f1"])
    axis.set(xlabel="F1", xlim=(0, 1.03), title=f"Per-class F1 — {model_name}")
    fig.tight_layout(); fig.savefig(model_dir / "per_class_f1.png", dpi=160); plt.close(fig)


def write_dashboard(output_dir: Path, summary: pd.DataFrame, config: dict, gt_path: Path) -> None:
    table = summary.to_html(index=False, float_format=lambda value: f"{value:.4f}", classes="metrics")
    cards = "".join(
        f'<div class="card"><h3>{html.escape(row.model)}</h3><strong>F1 {row.f1:.3f}</strong>'
        f'<span>Precision {row.precision:.3f} · Recall {row.recall:.3f}</span></div>'
        for row in summary.itertuples()
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Model comparison</title>
<style>body{{font-family:system-ui;margin:32px;background:#f5f7fb;color:#172033}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{background:white;padding:18px;border-radius:12px;box-shadow:0 2px 12px #0001;min-width:220px}}.card strong,.card span{{display:block;margin-top:8px}}img{{max-width:100%;background:white;border-radius:12px;margin-top:20px}}table{{border-collapse:collapse;background:white;width:100%;margin-top:20px}}th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style></head>
<body><h1>Model comparison</h1><p>Ground truth: {html.escape(str(gt_path))}<br>IoU {config['iou_threshold']} · confidence {config['confidence_threshold']}</p>
<div class="cards">{cards}</div>{table}<img src="model_comparison.png"><img src="best_f1_by_threshold.png"></body></html>"""
    (output_dir / "dashboard.html").write_text(document, encoding="utf-8")


def evaluate_all(config_path: str | Path = "config.json") -> pd.DataFrame:
    config, gt_path, models = discover_models(config_path)
    base = Path(config_path).resolve().parent
    output_dir = resolve(base, config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    gt = load_json(gt_path)
    validate_ground_truth(gt)
    confidence = float(config["confidence_threshold"])
    iou_threshold = float(config["iou_threshold"])
    max_detections = int(config["max_detections_per_image"])
    thresholds = sorted({float(value) for value in config["threshold_sweep"]} | {confidence})

    summaries = []
    all_per_class = []
    all_sweeps = []
    used_names: set[str] = set()
    for model in models:
        predictions = load_json(model["prediction_path"])
        validate_predictions(predictions, model["prediction_path"])
        validate_compatibility(gt, predictions, model["prediction_path"])
        run_id = safe_name(model["run_id"])
        if run_id in used_names:
            raise ValueError(f"Duplicate model folder name: {run_id}")
        used_names.add(run_id)
        model_dir = output_dir / run_id
        model_dir.mkdir(parents=True, exist_ok=True)

        summary, per_class_rows = match_detections(gt, predictions, confidence, iou_threshold, max_detections)
        summary.update(coco_metrics(gt_path, predictions, max_detections))
        summary.update({
            "model": model["name"], "run_id": run_id,
            "confidence": confidence, "iou": iou_threshold,
            "inference_ms": model["manifest"].get("inference_ms_per_image"),
            "prediction_file": str(model["prediction_path"]),
            "weight_file": str(model["weight_path"]) if model.get("weight_path") else None,
        })

        sweep_rows = []
        for threshold in thresholds:
            row, _ = match_detections(gt, predictions, threshold, iou_threshold, max_detections)
            sweep_rows.append({"model": model["name"], "confidence": threshold, **row})
        best = max(sweep_rows, key=lambda row: row["f1"])
        summary["best_f1"] = best["f1"]
        summary["best_confidence"] = best["confidence"]

        per_class_df = pd.DataFrame(per_class_rows)
        sweep_df = pd.DataFrame(sweep_rows)
        per_class_df.to_csv(model_dir / "per_class_metrics.csv", index=False)
        sweep_df.to_csv(model_dir / "threshold_sweep.csv", index=False)
        (model_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        save_model_plots(model_dir, model["name"], sweep_df, per_class_df)
        summaries.append(summary)
        all_per_class.extend({"model": model["name"], **row} for row in per_class_rows)
        all_sweeps.extend(sweep_rows)

    summary_df = pd.DataFrame(summaries).sort_values("f1", ascending=False).reset_index(drop=True)
    summary_df.insert(0, "rank", range(1, len(summary_df) + 1))
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(all_per_class).to_csv(output_dir / "all_per_class.csv", index=False)
    pd.DataFrame(all_sweeps).to_csv(output_dir / "all_threshold_sweeps.csv", index=False)

    sns.set_theme(style="whitegrid")
    metric_columns = ["precision", "recall", "f1", "mAP_50_95", "AP_50"]
    chart_df = summary_df.melt(id_vars="model", value_vars=metric_columns, var_name="metric", value_name="score")
    fig, axis = plt.subplots(figsize=(max(9, len(summary_df) * 1.5), 5))
    sns.barplot(data=chart_df, x="model", y="score", hue="metric", ax=axis)
    axis.set(ylim=(0, 1.03), xlabel="", ylabel="Score", title="Model comparison on the same test set")
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(output_dir / "model_comparison.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(max(8, len(summary_df) * 1.2), 4.5))
    sns.barplot(data=summary_df, x="model", y="best_f1", ax=axis)
    axis.set(ylim=(0, 1.03), xlabel="", ylabel="Best micro F1", title="Best F1 from confidence sweep")
    axis.tick_params(axis="x", rotation=25)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", padding=3)
    fig.tight_layout(); fig.savefig(output_dir / "best_f1_by_threshold.png", dpi=180); plt.close(fig)
    write_dashboard(output_dir, summary_df, config, gt_path)
    return summary_df


def display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rank", "model", "precision", "recall", "f1", "macro_f1",
        "mAP_50_95", "AP_50", "AR_100", "best_f1", "best_confidence", "inference_ms",
    ]
    return frame[[column for column in columns if column in frame]].round(4)


def load_learning_curve(model_root: str | Path, run_id: str) -> tuple[pd.DataFrame | None, Path | None]:
    """Normalize Torchvision history.csv or Ultralytics results.csv for plotting."""
    run_dir = Path(model_root).expanduser().resolve() / run_id
    candidates = [run_dir / "history.csv", run_dir / "results.csv"]
    history_path = next((path for path in candidates if path.is_file()), None)
    if history_path is None:
        history_path = next(iter(run_dir.glob("**/results.csv")), None) if run_dir.is_dir() else None
    if history_path is None:
        return None, None

    raw = pd.read_csv(history_path)
    raw.columns = [str(column).strip() for column in raw.columns]
    if raw.empty:
        return None, history_path
    curve = pd.DataFrame()
    curve["epoch"] = pd.to_numeric(raw["epoch"], errors="coerce") if "epoch" in raw else np.arange(1, len(raw) + 1)
    if curve["epoch"].min() == 0:
        curve["epoch"] += 1

    direct = {
        "train_loss": ["train_loss"],
        "val_loss": ["val_loss"],
        "mAP_50_95": ["val_AP_50_95", "metrics/mAP50-95(B)", "metrics/mAP50-95(M)"],
        "mAP_50": ["val_AP_50", "metrics/mAP50(B)", "metrics/mAP50(M)"],
        "precision": ["metrics/precision(B)", "metrics/precision(M)"],
        "recall": ["metrics/recall(B)", "metrics/recall(M)"],
    }
    for target, options in direct.items():
        source = next((column for column in options if column in raw), None)
        if source:
            curve[target] = pd.to_numeric(raw[source], errors="coerce")

    for prefix, target in (("train/", "train_loss"), ("val/", "val_loss")):
        if target not in curve:
            loss_columns = [column for column in raw if column.startswith(prefix) and "loss" in column.lower()]
            if loss_columns:
                curve[target] = raw[loss_columns].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    curve = curve.dropna(axis=1, how="all").dropna(subset=["epoch"])
    return curve, history_path


def load_training_settings(model_root: str | Path, run_id: str) -> tuple[dict | None, dict]:
    run_dir = Path(model_root).expanduser().resolve() / run_id
    config_path, metrics_path = run_dir / "config.json", run_dir / "metrics.json"
    if not config_path.is_file():
        return None, {}
    config = load_json(config_path)
    metrics = load_json(metrics_path) if metrics_path.is_file() else {}
    is_torchvision = "detector" in config
    row = {
        "model": config.get("run_name", run_id),
        "framework/model": config.get("detector") or config.get("weights") or metrics.get("framework", "-"),
        "image size": config.get("image_size"),
        "epochs": config.get("epochs"),
        "batch": config.get("batch_size"),
        "optimizer": "SGD" if is_torchvision else "Ultralytics auto",
        "learning rate": str(config.get("learning_rate", "auto")),
        "pretrained": config.get("use_pretrained"),
        "AMP": config.get("amp"),
        "seed": config.get("seed"),
        "workers": config.get("workers"),
        "classes": len(config.get("classes", [])),
        "best epoch": metrics.get("best_epoch"),
        "parameters (M)": round(float(metrics["parameters"]) / 1_000_000, 2) if metrics.get("parameters") else None,
        "training (hours)": round(float(metrics["training_seconds"]) / 3600, 2) if metrics.get("training_seconds") else None,
        "test FPS": round(float(metrics["test_fps_end_to_end"]), 2) if metrics.get("test_fps_end_to_end") else None,
    }
    return row, config
