from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from evaluator import load_learning_curve, load_training_settings


APP_DIR = Path(__file__).resolve().parent


def copy_if_present(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build(model_root: Path) -> None:
    source = APP_DIR / "results"
    target = APP_DIR / "public_report"
    summary = pd.read_csv(source / "summary.csv")
    summary = summary.drop(columns=["prediction_file", "weight_file"], errors="ignore")
    target.mkdir(parents=True, exist_ok=True)
    summary.to_csv(target / "summary.csv", index=False)

    for name in ["all_per_class.csv", "all_threshold_sweeps.csv", "model_comparison.png", "best_f1_by_threshold.png"]:
        copy_if_present(source / name, target / name)

    settings_rows = []
    for row in summary.to_dict("records"):
        run_id = str(row["run_id"])
        run_target = target / "runs" / run_id
        for name in ["per_class_metrics.csv", "threshold_sweep.csv", "per_class_f1.png", "threshold_sweep.png"]:
            copy_if_present(source / run_id / name, run_target / name)
        curve, _ = load_learning_curve(model_root, run_id)
        if curve is not None:
            run_target.mkdir(parents=True, exist_ok=True)
            curve.to_csv(run_target / "learning_curve.csv", index=False)
        settings, _ = load_training_settings(model_root, run_id)
        if settings:
            settings_rows.append(settings)
    pd.DataFrame(settings_rows).to_csv(target / "training_settings.csv", index=False)
    print(f"Public report ready: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a sanitized, read-only report for Streamlit Cloud")
    parser.add_argument("model_root", nargs="?", default=APP_DIR.parent / "xrayfsod_outputs", type=Path)
    build(parser.parse_args().model_root.expanduser().resolve())

