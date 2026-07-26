from __future__ import annotations

import json
import threading
from pathlib import Path
from queue import Empty, Full, Queue
from time import monotonic, perf_counter

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "yolov11_xrayfsod_best.pt"
MAX_IMAGE_PIXELS = 25_000_000
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


@st.cache_resource(show_spinner=False)
def load_demo_model():
    from ultralytics import YOLO

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


def read_image(uploaded) -> Image.Image | None:
    if uploaded is None:
        return None
    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("ไฟล์มีขนาดเกิน 10 MB")
        return None

    try:
        image = Image.open(uploaded)
        if image.width * image.height > MAX_IMAGE_PIXELS:
            st.error("ภาพมีความละเอียดสูงเกินไป กรุณาใช้ภาพที่ไม่เกิน 25 ล้านพิกเซล")
            return None
        return image.convert("RGB")
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError):
        st.error("ไฟล์ที่เลือกไม่ใช่ภาพที่รองรับ")
        return None


def run_detection(image: Image.Image, confidence: float, iou: float):
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
    return result, (perf_counter() - started) * 1000


def show_detection_result(image: Image.Image, result, elapsed_ms: float, source: str) -> None:
    rows = detection_rows(result)
    annotated = result.plot()

    original_column, result_column = st.columns(2)
    with original_column:
        st.image(image, caption=source, use_container_width=True)
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
        "source": source,
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


class LiveYoloProcessor:
    """Keeps the webcam stream responsive while one worker runs YOLO inference."""

    def __init__(self, model, confidence: float, iou: float, interval_seconds: float) -> None:
        self._model = model
        self._confidence = confidence
        self._iou = iou
        self._interval_seconds = interval_seconds
        self._queue: Queue[np.ndarray] = Queue(maxsize=1)
        self._lock = threading.Lock()
        self._detections: list[dict] = []
        self._inference_ms = 0.0
        self._last_submitted = 0.0
        self._busy = False
        self._worker = threading.Thread(target=self._run, daemon=True, name="yolo-live-inference")
        self._worker.start()

    def _run(self) -> None:
        while True:
            try:
                image = self._queue.get(timeout=1)
            except Empty:
                continue

            try:
                started = perf_counter()
                result = self._model.predict(
                    image,
                    conf=self._confidence,
                    iou=self._iou,
                    imgsz=640,
                    device="cpu",
                    verbose=False,
                )[0]
                rows = detection_rows(result)
                elapsed_ms = (perf_counter() - started) * 1000
                with self._lock:
                    self._detections = rows
                    self._inference_ms = elapsed_ms
            except Exception:
                # Keep the camera stream available even if one frame cannot be processed.
                with self._lock:
                    self._detections = []
                    self._inference_ms = 0.0
            finally:
                with self._lock:
                    self._busy = False
                self._queue.task_done()

    def recv(self, frame):
        import av
        import cv2

        image = frame.to_ndarray(format="bgr24")
        now = monotonic()
        with self._lock:
            should_submit = not self._busy and now - self._last_submitted >= self._interval_seconds
            detections = list(self._detections)
            inference_ms = self._inference_ms

            if should_submit:
                self._busy = True
                self._last_submitted = now
                try:
                    self._queue.put_nowait(image.copy())
                except Full:
                    self._busy = False

            busy = self._busy

        annotated = image.copy()
        for row in detections:
            x1, y1, x2, y2 = (int(row[key]) for key in ("x1", "y1", "x2", "y2"))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (38, 207, 108), 2)
            label = f"{row['class']} {row['confidence']:.2f}"
            label_y = max(y1 - 8, 20)
            cv2.putText(
                annotated,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (38, 207, 108),
                2,
                cv2.LINE_AA,
            )

        status = "Detecting..." if busy else f"Latest: {len(detections)} objects | {inference_ms:.0f} ms"
        cv2.rectangle(annotated, (0, 0), (min(annotated.shape[1], 430), 34), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            status,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def show_live_camera(confidence: float, iou: float) -> None:
    try:
        from streamlit_webrtc import webrtc_streamer
    except ImportError:
        st.error("ยังไม่ได้ติดตั้งส่วนรองรับกล้องวิดีโอ")
        return

    st.caption("เปิดกล้องแล้วกด START ภาพวิดีโอจะไหลต่อเนื่อง และระบบจะตรวจจับเป็นช่วง ๆ พร้อมกรอบของหลายวัตถุในเฟรมเดียว")
    interval_seconds = st.slider(
        "ช่วงห่างการตรวจจับ (วินาที)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.5,
        help="เพิ่มค่านี้หากเครื่องช้าเพื่อให้วิดีโอไหลลื่นขึ้น",
    )

    model = load_demo_model()

    def processor_factory():
        return LiveYoloProcessor(model, confidence, iou, interval_seconds)

    webrtc_streamer(
        key=f"yolov11-live-{confidence:.2f}-{iou:.2f}-{interval_seconds:.1f}",
        video_processor_factory=processor_factory,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_html_attrs={"autoPlay": True, "controls": False, "muted": True},
    )
    st.info("หากเครือข่ายไม่อนุญาตการส่งวิดีโอสด ให้ใช้โหมดถ่ายภาพจากกล้องแทนได้ทันที")


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

    source = st.radio(
        "แหล่งภาพ",
        ["อัปโหลดภาพ", "ถ่ายภาพจากกล้อง", "กล้องวิดีโอสด"],
        horizontal=True,
    )

    if source == "กล้องวิดีโอสด":
        show_live_camera(confidence, iou)
        return

    form_key = "upload-image-form" if source == "อัปโหลดภาพ" else "camera-image-form"
    with st.form(form_key):
        if source == "อัปโหลดภาพ":
            uploaded = st.file_uploader(
                "ภาพเอกซเรย์",
                type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                accept_multiple_files=False,
            )
        else:
            uploaded = st.camera_input("ถ่ายภาพจากกล้อง")
        st.caption("รองรับไฟล์ไม่เกิน 10 MB และความละเอียดไม่เกิน 25 ล้านพิกเซล")
        submitted = st.form_submit_button("เริ่มตรวจจับ", type="primary", use_container_width=True)

    if not submitted:
        return

    image = read_image(uploaded)
    if image is None:
        st.info("กรุณาเลือกหรือถ่ายภาพก่อนเริ่มตรวจจับ")
        return

    try:
        with st.spinner("กำลังประมวลผลด้วย YOLOv11..."):
            result, elapsed_ms = run_detection(image, confidence, iou)
    except Exception:
        st.error("ไม่สามารถประมวลผลภาพได้ กรุณาลองภาพอื่นหรือลองใหม่ภายหลัง")
        return

    show_detection_result(image, result, elapsed_ms, source)
