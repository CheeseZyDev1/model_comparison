from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "yolov11_xrayfsod_best.pt"
MAX_IMAGE_PIXELS = 25_000_000


@st.cache_resource(show_spinner=False)
def load_demo_model() -> YOLO:
    return YOLO(str(MODEL_PATH))


def detection_rows(result) -> list[dict]:
    rows = []
    if result.boxes is None:
        return rows

    names = result.names if isinstance(result.names, dict) else dict(enumerate(result.names))
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
        rows.append(
            {
                "class": str(names.get(class_id, class_id)),
                "confidence": round(float(box.conf[0].item()), 3),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            }
        )
    return rows


def show_model_demo() -> None:
    st.title("ทดสอบโมเดลจริง")
    st.caption("YOLOv11 ที่คัดเลือกจากการเปรียบเทียบโมเดลในโครงงาน")
    st.warning("ผลลัพธ์เป็นต้นแบบเพื่อการทดลองและต้องใช้การพิจารณาของมนุษย์ ไม่ใช่การตัดสินด้านความปลอดภัย")

    if not MODEL_PATH.is_file():
        st.error("ไม่พบไฟล์โมเดลสำหรับการทดสอบ")
        return

    threshold_column, iou_column = st.columns(2)
    with threshold_column:
        confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    with iou_column:
        iou = st.slider("IoU threshold", 0.10, 0.90, 0.50, 0.05)

    uploaded = st.file_uploader(
        "ภาพเอกซเรย์",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=False,
    )
    if uploaded is None:
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("ไฟล์ที่อัปโหลดไม่ใช่ภาพที่รองรับ")
        return

    if image.width * image.height > MAX_IMAGE_PIXELS:
        st.error("ภาพมีความละเอียดสูงเกินไป กรุณาใช้ภาพที่ไม่เกิน 25 ล้านพิกเซล")
        return

    try:
        with st.spinner("กำลังประมวลผลด้วย YOLOv11..."):
            model = load_demo_model()
            started = perf_counter()
            result = model.predict(
                image,
                conf=confidence,
                iou=iou,
                imgsz=640,
                device="cpu",
                verbose=False,
            )[0]
            elapsed_ms = (perf_counter() - started) * 1000
    except Exception as error:
        st.error(f"ไม่สามารถประมวลผลภาพได้: {error}")
        return

    rows = detection_rows(result)
    annotated = result.plot()

    original_column, result_column = st.columns(2)
    with original_column:
        st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)
    with result_column:
        st.image(
            annotated,
            caption="ผลการตรวจจับจาก YOLOv11",
            channels="BGR",
            use_container_width=True,
        )

    model_inference_ms = float(result.speed.get("inference", 0.0))
    metrics = st.columns(3)
    metrics[0].metric("วัตถุที่ตรวจพบ", len(rows))
    metrics[1].metric("Model inference", f"{model_inference_ms:.0f} ms")
    metrics[2].metric("เวลารวม", f"{elapsed_ms:.0f} ms")

    st.subheader("รายละเอียดผลการตรวจจับ")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("ไม่พบวัตถุที่มี confidence สูงกว่าค่าที่กำหนด")

    export = {
        "model": "YOLOv11-1",
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "model_inference_ms": round(model_inference_ms, 2),
        "elapsed_ms": round(elapsed_ms, 2),
        "detections": rows,
        "notice": "Prototype output for experimental review; human interpretation is required.",
    }
    st.download_button(
        "ดาวน์โหลดผลตรวจจับ JSON",
        data=json.dumps(export, ensure_ascii=False, indent=2),
        file_name="xrayfsod_yolov11_prediction.json",
        mime="application/json",
    )
