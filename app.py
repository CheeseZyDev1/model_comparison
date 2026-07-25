from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
BASE_CONFIG = json.loads((APP_DIR / "config.json").read_text(encoding="utf-8"))
DEFAULT_RUNS = BASE_CONFIG["model_runs"]
DEFAULT_MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", APP_DIR / BASE_CONFIG["model_root"])).expanduser().resolve()
report_mode = os.environ.get("REPORT_MODE")
PUBLIC_MODE = report_mode.lower() == "true" if report_mode is not None else not DEFAULT_MODEL_ROOT.is_dir()
PUBLIC_REPORT = APP_DIR / "public_report"
THESIS_PDF = APP_DIR / "thesis" / "thesis_latest.pdf"
THESIS_PAGES = APP_DIR / "thesis" / "pages"


def display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rank", "model", "precision", "recall", "f1", "macro_f1",
        "mAP_50_95", "AP_50", "AR_100", "best_f1", "best_confidence", "inference_ms",
    ]
    return frame[[column for column in columns if column in frame]].round(4)


if not PUBLIC_MODE:
    from evaluator import evaluate_all, load_learning_curve, load_training_settings

def show_thesis() -> None:
    st.title("เล่มปริญญานิพนธ์")
    st.caption("ระบบตรวจจับวัตถุต้องห้ามในภาพเอกซเรย์ - ฉบับปัจจุบันสำหรับอ่านตรวจเนื้อหาและรูปแบบ")
    if not THESIS_PDF.is_file():
        st.error("ไม่พบไฟล์ PDF ของเล่มปริญญานิพนธ์")
        return

    pdf_bytes = THESIS_PDF.read_bytes()
    st.download_button(
        "ดาวน์โหลด PDF",
        data=pdf_bytes,
        file_name="Napat_Nueyen_XrayFSOD_Thesis.pdf",
        mime="application/pdf",
        type="primary",
    )

    page_files = sorted(THESIS_PAGES.glob("page-*.png"))
    if not page_files:
        st.warning("ไม่พบภาพตัวอย่างแต่ละหน้า กรุณาใช้ปุ่มดาวน์โหลด PDF")
        return

    page_number = st.slider("เลือกหน้า", 1, len(page_files), 1)
    st.image(
        str(page_files[page_number - 1]),
        caption=f"หน้า {page_number} จาก {len(page_files)}",
        use_container_width=True,
    )


st.set_page_config(page_title="X-ray FSOD Project", page_icon="📊", layout="wide")
page = st.sidebar.radio(
    "เมนู",
    ["ผลเปรียบเทียบโมเดล", "อ่านเล่มปริญญานิพนธ์"],
)
if page == "อ่านเล่มปริญญานิพนธ์":
    show_thesis()
    st.stop()

st.title("X-ray Model Comparison")
st.caption("เปรียบเทียบทุกโมเดลด้วย test set และเกณฑ์เดียวกัน")

model_root = str(DEFAULT_MODEL_ROOT)
if PUBLIC_MODE:
    st.sidebar.success("Public report · read-only")
    st.sidebar.caption("ไม่มี model weights, dataset หรือ filesystem access บนเว็บนี้")
    run_clicked = False
else:
    with st.sidebar:
        st.header("ตั้งค่า")
        model_root = st.text_input("Model root", str(DEFAULT_MODEL_ROOT))
        ground_truth = st.text_input(
            "Ground-truth COCO JSON",
            os.environ.get("GT_JSON", str((APP_DIR / BASE_CONFIG["ground_truth"]).resolve())),
        )
        runs_text = st.text_area("Model runs (หนึ่งชื่อต่อบรรทัด)", "\n".join(DEFAULT_RUNS), height=150)
        confidence = st.slider("Confidence", 0.0, 1.0, float(BASE_CONFIG["confidence_threshold"]), 0.05)
        iou = st.slider("IoU", 0.1, 0.95, float(BASE_CONFIG["iou_threshold"]), 0.05)
        run_clicked = st.button("ประเมินโมเดล", type="primary", use_container_width=True)

runtime_path = APP_DIR / "results" / "runtime_config.json"
results_dir = PUBLIC_REPORT if PUBLIC_MODE else APP_DIR / "results"

if run_clicked:
    runs = [line.strip() for line in runs_text.splitlines() if line.strip()]
    if not runs:
        st.error("กรุณาระบุอย่างน้อยหนึ่ง model run")
        st.stop()
    runtime = {
        **BASE_CONFIG,
        "ground_truth": str(Path(ground_truth).expanduser().resolve()),
        "model_root": str(Path(model_root).expanduser().resolve()),
        "model_runs": runs,
        "output_dir": str(results_dir),
        "confidence_threshold": confidence,
        "iou_threshold": iou,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        with st.spinner("กำลังประเมินและสร้างกราฟ..."):
            st.session_state["summary"] = evaluate_all(runtime_path)
        st.success("ประเมินเสร็จแล้ว")
    except Exception as error:
        st.error(str(error))
        st.stop()

summary_path = results_dir / "summary.csv"
summary = st.session_state.get("summary")
if summary is None and summary_path.is_file():
    summary = pd.read_csv(summary_path)

if summary is None:
    st.info("ยังไม่มี public report" if PUBLIC_MODE else "ตั้งค่า path ด้านซ้าย แล้วกด **ประเมินโมเดล**")
    st.stop()

st.subheader("อันดับรวม")
st.dataframe(display_columns(summary), use_container_width=True, hide_index=True)
st.download_button(
    "ดาวน์โหลด summary.csv",
    summary.to_csv(index=False).encode("utf-8-sig"),
    "model_comparison_summary.csv",
    "text/csv",
)

left, right = st.columns(2)
with left:
    if (results_dir / "model_comparison.png").is_file():
        st.image(str(results_dir / "model_comparison.png"), use_container_width=True)
with right:
    if (results_dir / "best_f1_by_threshold.png").is_file():
        st.image(str(results_dir / "best_f1_by_threshold.png"), use_container_width=True)

st.subheader("Training settings")
settings_rows = []
training_configs = {}
if PUBLIC_MODE and (results_dir / "training_settings.csv").is_file():
    settings_rows = pd.read_csv(results_dir / "training_settings.csv").to_dict("records")
else:
    for row in summary.to_dict("records"):
        settings, full_config = load_training_settings(model_root, str(row["run_id"]))
        if settings:
            settings_rows.append(settings)
            training_configs[str(row["model"])] = full_config
if settings_rows:
    st.dataframe(pd.DataFrame(settings_rows), use_container_width=True, hide_index=True)
    st.caption(
        "ค่าร่วมจาก notebook: validation split 10% · pretrained · AMP · deterministic · "
        "ทดสอบด้วย confidence floor 0.001 และไม่เกิน 100 detections/ภาพ"
    )
    with st.expander("รายละเอียดวิธีเทรน"):
        st.markdown(
            """
- **EfficientNet / ResNet50 / VGG16:** Faster R-CNN + FPN, SGD (`momentum=0.9`, `weight_decay=0.0005`),
  learning-rate scheduler ลด LR 10 เท่าประมาณ epoch 34 และ 45, horizontal flip 50%, validation ทุก epoch
- **YOLOv11:** pretrained `yolo11l.pt`, ใช้ optimizer/scheduler/augmentation ค่าอัตโนมัติของ Ultralytics
- **RT-DETR:** pretrained `rtdetr-l.pt`, ใช้ optimizer/scheduler/augmentation ค่าอัตโนมัติของ Ultralytics
- ทุกโมเดลเทรนสูงสุด **50 epochs**, target image size **640**, ใช้ **20 classes** และบันทึก `best.pt`
"""
        )
else:
    st.warning("ไม่พบ config.json ในโฟลเดอร์ run จึงแสดง training settings ไม่ได้")

st.subheader("Learning curves")
learning_curves = {}
missing_histories = []
for row in summary.to_dict("records"):
    run_id = str(row["run_id"])
    if PUBLIC_MODE:
        curve_path = results_dir / "runs" / run_id / "learning_curve.csv"
        curve = pd.read_csv(curve_path) if curve_path.is_file() else None
        source = curve_path.name
    else:
        curve, source = load_learning_curve(model_root, run_id)
    if curve is None:
        missing_histories.append(str(row["run_id"]))
    else:
        learning_curves[str(row["model"])] = (curve, source)

if learning_curves:
    loss_series, map_series = [], []
    for model_name, (curve, _) in learning_curves.items():
        if "train_loss" in curve:
            loss_series.append(curve[["epoch", "train_loss"]].rename(columns={"train_loss": model_name}).set_index("epoch"))
        if "mAP_50_95" in curve:
            map_series.append(curve[["epoch", "mAP_50_95"]].rename(columns={"mAP_50_95": model_name}).set_index("epoch"))
    curve_left, curve_right = st.columns(2)
    with curve_left:
        st.caption("Training loss (ยิ่งต่ำยิ่งดี)")
        if loss_series:
            st.line_chart(pd.concat(loss_series, axis=1), use_container_width=True)
        else:
            st.info("ไฟล์ history ไม่มีข้อมูล training loss")
    with curve_right:
        st.caption("Validation mAP@0.50:0.95 (ยิ่งสูงยิ่งดี)")
        if map_series:
            st.line_chart(pd.concat(map_series, axis=1), use_container_width=True)
        else:
            st.info("ไฟล์ history ไม่มีข้อมูล validation mAP")
if missing_histories:
    st.warning("ไม่พบ history.csv/results.csv: " + ", ".join(missing_histories))

st.subheader("ผลรายโมเดล")
tabs = st.tabs(summary["model"].astype(str).tolist())
for tab, row in zip(tabs, summary.to_dict("records")):
    with tab:
        run_dir = results_dir / "runs" / str(row["run_id"]) if PUBLIC_MODE else results_dir / str(row["run_id"])
        a, b, c = st.columns(3)
        a.metric("F1", f"{float(row['f1']):.4f}")
        b.metric("Precision", f"{float(row['precision']):.4f}")
        c.metric("Recall", f"{float(row['recall']):.4f}")
        graph_left, graph_right = st.columns(2)
        if (run_dir / "threshold_sweep.png").is_file():
            graph_left.image(str(run_dir / "threshold_sweep.png"), use_container_width=True)
        if (run_dir / "per_class_f1.png").is_file():
            graph_right.image(str(run_dir / "per_class_f1.png"), use_container_width=True)
        per_class_path = run_dir / "per_class_metrics.csv"
        if per_class_path.is_file():
            st.dataframe(pd.read_csv(per_class_path).round(4), use_container_width=True, hide_index=True)
        curve_info = learning_curves.get(str(row["model"]))
        if curve_info:
            curve, source = curve_info
            st.caption(f"Learning curve จาก {source}")
            available = [column for column in ["train_loss", "val_loss", "mAP_50_95", "mAP_50", "precision", "recall"] if column in curve]
            st.line_chart(curve.set_index("epoch")[available], use_container_width=True)
        config_info = training_configs.get(str(row["model"]))
        if config_info:
            with st.expander("ดู config.json ของโมเดลนี้"):
                st.json(config_info)
