from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path
from queue import Empty, Full, Queue
from time import monotonic, perf_counter

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "yolov11_xrayfsod_best.pt"
MAX_IMAGE_PIXELS = 25_000_000
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

CLASS_DISPLAY_NAMES = {
    "laptop": "โน้ตบุ๊ก",
    "lighter": "ไฟแช็ก",
    "portable_charger_2": "พาวเวอร์แบงก์",
    "iron_shoe": "เตารีด",
    "straight_knife": "มีดปลายตรง",
    "folding_knife": "มีดพับ",
    "scissor": "กรรไกร",
    "multi-tool_knife": "มีดอเนกประสงค์",
    "umbrella": "ร่ม",
    "glass_bottle": "ขวดแก้ว",
    "battery": "แบตเตอรี่",
    "metal_cup": "แก้วโลหะ",
    "nail_clippers": "กรรไกรตัดเล็บ",
    "pressure_tank": "ถังแรงดัน",
    "spray_alcohol": "สเปรย์แอลกอฮอล์",
    "portable_charger_1": "พาวเวอร์แบงก์",
    "utility_knife": "มีดคัตเตอร์",
    "mobile_phone": "โทรศัพท์มือถือ",
    "metal_can": "กระป๋องโลหะ",
    "drink_bottle": "ขวดเครื่องดื่ม",
}
LABEL_FONT_PATHS = (
    Path("C:/Windows/Fonts/LeelawUI.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansThai-Regular.ttf"),
)
LATIN_FONT_PATHS = (
    Path("C:/Windows/Fonts/LeelawUI.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
)


@st.cache_resource(show_spinner=False)
def load_demo_model():
    from ultralytics import YOLO

    return YOLO(str(MODEL_PATH))


def display_name(class_name: str) -> str:
    return CLASS_DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").title())


@lru_cache(maxsize=16)
def label_font(size: int):
    for path in LABEL_FONT_PATHS:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


@lru_cache(maxsize=16)
def latin_font(size: int):
    for path in LATIN_FONT_PATHS:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def detection_rows(result) -> list[dict]:
    rows = []
    if result.boxes is None:
        return rows

    names = result.names if isinstance(result.names, dict) else dict(enumerate(result.names))
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
        raw_name = str(names.get(class_id, class_id))
        rows.append(
            {
                "class": display_name(raw_name),
                "confidence": round(float(box.conf[0].item()), 3),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            }
        )
    return rows


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def annotate_detections(
    image: Image.Image,
    rows: list[dict],
    status_segments: list[tuple[str, object]] | None = None,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = label_font(max(16, min(28, canvas.width // 36)))
    value_font = latin_font(max(16, min(28, canvas.width // 36)))

    for row in rows:
        x1, y1, x2, y2 = (int(row[key]) for key in ("x1", "y1", "x2", "y2"))
        class_label = row["class"]
        value_label = f" {row['confidence']:.2f}"
        class_width, class_height = text_size(draw, class_label, font)
        value_width, value_height = text_size(draw, value_label, value_font)
        text_width = class_width + value_width
        text_height = max(class_height, value_height)
        label_top = max(y1 - text_height - 12, 0)
        label_right = min(x1 + text_width + 12, canvas.width)

        draw.rectangle((x1, y1, x2, y2), outline="#22c55e", width=3)
        draw.rounded_rectangle((x1, label_top, label_right, label_top + text_height + 10), radius=4, fill="#14532d")
        draw.text((x1 + 6, label_top + 3), class_label, font=font, fill="white")
        draw.text((x1 + 6 + class_width, label_top + 3), value_label, font=value_font, fill="white")

    if status_segments:
        segment_sizes = [(text, *text_size(draw, text, segment_font)) for text, segment_font in status_segments]
        status_width = sum(width for _, width, _ in segment_sizes)
        status_height = max(height for _, _, height in segment_sizes)
        draw.rounded_rectangle((8, 8, status_width + 24, status_height + 22), radius=4, fill="#1f2937")
        cursor_x = 16
        for (text, segment_font), (_, segment_width, _) in zip(status_segments, segment_sizes):
            draw.text((cursor_x, 12), text, font=segment_font, fill="white")
            cursor_x += segment_width

    return canvas


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
    annotated = annotate_detections(image, rows)

    original_column, result_column = st.columns(2)
    with original_column:
        st.image(image, caption=source, use_container_width=True)
    with result_column:
        st.image(
            annotated,
            caption="ผลการตรวจจับจาก YOLOv11",
            use_container_width=True,
        )

    model_inference_ms = float(result.speed.get("inference", 0.0))
    metrics = st.columns(3)
    metrics[0].metric("วัตถุที่ตรวจพบ", len(rows))
    metrics[1].metric("Model inference", f"{model_inference_ms:.0f} ms")
    metrics[2].metric("เวลารวม", f"{elapsed_ms:.0f} ms")

    st.subheader("รายละเอียดผลการตรวจจับ")
    if rows:
        display_rows = pd.DataFrame(rows).rename(
            columns={
                "class": "วัตถุ",
                "confidence": "ความมั่นใจ",
                "x1": "x1",
                "y1": "y1",
                "x2": "x2",
                "y2": "y2",
            }
        )
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
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

        status_font = label_font(max(15, min(24, image.shape[1] // 42)))
        if busy:
            status_segments = [("กำลังตรวจจับ...", status_font)]
        else:
            status_segments = [
                ("ผลล่าสุด: ", status_font),
                (f"{len(detections)}", latin_font(max(15, min(24, image.shape[1] // 42)))),
                (" วัตถุ | ", status_font),
                (f"{inference_ms:.0f} ms", latin_font(max(15, min(24, image.shape[1] // 42)))),
            ]
        annotated = annotate_detections(Image.fromarray(image[:, :, ::-1].copy()), detections, status_segments)
        return av.VideoFrame.from_ndarray(
            np.ascontiguousarray(np.asarray(annotated)[:, :, ::-1]),
            format="bgr24",
        )


def live_rtc_configuration() -> dict:
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    try:
        turn_url = str(st.secrets.get("TURN_URL", "")).strip()
        turn_username = str(st.secrets.get("TURN_USERNAME", "")).strip()
        turn_credential = str(st.secrets.get("TURN_CREDENTIAL", "")).strip()
    except Exception:
        return {"iceServers": ice_servers}
    if turn_url and turn_username and turn_credential:
        ice_servers.append(
            {
                "urls": [turn_url],
                "username": turn_username,
                "credential": turn_credential,
            }
        )
    return {"iceServers": ice_servers}


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
        rtc_configuration=live_rtc_configuration(),
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
        confidence = st.slider("Confidence threshold (ความมั่นใจขั้นต่ำ)", 0.05, 0.95, 0.25, 0.05)
    with iou_column:
        iou = st.slider("IoU threshold (การจัดการกรอบซ้อน)", 0.10, 0.90, 0.50, 0.05)

    with st.expander("ความหมายของค่า Confidence และ IoU"):
        st.markdown(
            """
**Confidence threshold** คือระดับความมั่นใจขั้นต่ำที่โมเดลต้องมีจึงจะแสดงวัตถุนั้นบนภาพ
เมื่อปรับให้ต่ำลง จะเห็นวัตถุที่โมเดลยังไม่มั่นใจมากขึ้น แต่ผลที่คลาดเคลื่อนอาจเพิ่มขึ้น
เมื่อปรับให้สูงขึ้น จะแสดงเฉพาะผลที่โมเดลมั่นใจมากขึ้น แต่อาจพลาดวัตถุบางรายการได้

**IoU threshold** ใช้จัดการกรอบตรวจจับที่ซ้อนทับกัน
เมื่อปรับให้ต่ำลง ระบบจะตัดกรอบที่ทับซ้อนกันออกมากขึ้น
เมื่อปรับให้สูงขึ้น ระบบจะเก็บกรอบที่ซ้อนกันไว้มากขึ้น ซึ่งอาจเห็นกรอบซ้ำสำหรับวัตถุเดียวกัน
"""
        )

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
