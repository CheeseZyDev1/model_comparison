# ชุดเปรียบเทียบโมเดล Object Detection

ชุดนี้ใช้เปรียบเทียบโมเดลหลายสถาปัตยกรรมด้วย test set เดียวกัน โดยไม่ผูกกับวิธีโหลด weight ของ framework ใด framework หนึ่ง

## อัปขึ้น GitHub และ Deploy ผ่าน Streamlit Community Cloud

ให้อัปโหลด **เฉพาะไฟล์ภายในโฟลเดอร์ `model_comparison`** เป็น root ของ GitHub repository อย่าอัปโหลดโฟลเดอร์ `PJ1` ทั้งก้อน และห้ามเพิ่ม `xrayfsod_outputs`, Dataset, `.pt`, `test_predictions.json`, `.env` หรือ `.streamlit/secrets.toml`

ก่อนอัปโหลด หากต้องการสร้าง public report ใหม่ ให้รันจาก `model_comparison`:

```powershell
python build_public_report.py "..\xrayfsod_outputs"
```

`public_report/` มีเฉพาะ CSV และกราฟที่ลบ path ภายในเครื่องแล้ว เมื่อรันบน Cloud เว็บจะเข้าโหมด read-only อัตโนมัติ จึงไม่มีช่องกรอก path และไม่มีปุ่มสั่ง evaluator ที่คนภายนอกใช้กินทรัพยากรได้

ขั้นตอน deploy:

1. สร้าง GitHub repository ใหม่
2. อัปโหลดเนื้อหาภายใน `model_comparison` โดยให้ `app.py` และ `requirements.txt` อยู่ที่ root ของ repository
3. เข้า Streamlit Community Cloud แล้วเลือก **Create app**
4. เลือก repository, branch และกำหนด entrypoint เป็น `app.py`
5. กด Deploy — เวอร์ชัน public report นี้ไม่ต้องใส่ Secrets

หากภายหลังมี API key หรือรหัสผ่าน ห้ามเขียนลงโค้ดหรือ GitHub ให้ใส่ผ่านหน้า Secrets ของ Streamlit Cloud เท่านั้น โดย `.gitignore` ได้กัน `.streamlit/secrets.toml` ไว้แล้ว

## วิธีใช้แบบเร็วที่สุด

โฟลเดอร์บนเซิร์ฟเวอร์ควรเป็นแบบนี้:

```text
project/
├── model_comparison/
└── xrayfsod_outputs/
    ├── EfficientNet-1/
    ├── ResNet50-1/
    ├── RT-DETR-1/
    ├── VGG16-4/
    └── YOLOv11-1/
```

เปิด terminal ที่เซิร์ฟเวอร์แล้วรัน:

```bash
cd project/model_comparison
python -m pip install -r requirements-local.txt
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

จากเครื่องของผู้ใช้เปิด:

```text
http://<IP-ของเซิร์ฟเวอร์>:8501
```

บนหน้าเว็บ:

1. `Model root` เลือกโฟลเดอร์ `xrayfsod_outputs`
2. `Ground-truth COCO JSON` เลือก `_prepared_full20/annotations/test.json`
3. ตรวจรายชื่อโมเดล โดยไม่ใส่ VGG16-1/2/3
4. เลือก Confidence และ IoU หรือใช้ค่าเริ่มต้น 0.25/0.50
5. กด **ประเมินโมเดล** แล้วรอจนขึ้นข้อความสำเร็จ
6. ดูอันดับรวม, Learning curves, กราฟเปรียบเทียบ และแท็บรายโมเดล หรือดาวน์โหลด `summary.csv`

หน้า **Training settings** จะแสดงค่าจาก `config.json`/`metrics.json` ของแต่ละ run เช่น model/weights, image size, epochs, batch size, optimizer, learning rate, AMP, seed, จำนวนพารามิเตอร์, เวลาเทรน และ test FPS พร้อมรายละเอียด protocol ที่ใช้ใน notebook

## โครงสร้าง

```text
model_comparison/
├── compare_models.ipynb       # เปิดและ Run All
├── config.json                # ตั้ง path และ threshold ที่จุดเดียว
├── evaluator.py               # ตัวคำนวณ metric และสร้างกราฟ
├── requirements.txt
├── model_inputs/              # ทางเลือก: วางผลของแต่ละโมเดลแยกโฟลเดอร์
│   ├── README.md
│   └── <ชื่อโมเดล>/
│       ├── predictions.json   # จำเป็น
│       ├── model.json         # ไม่จำเป็น
│       └── weights/best.pt    # ไม่จำเป็น ใช้จัดระเบียบเท่านั้น
└── results/                   # สร้างอัตโนมัติเมื่อรัน
    ├── dashboard.html
    ├── summary.csv
    ├── model_comparison.png
    └── <ชื่อโมเดล>/
        ├── metrics.json
        ├── per_class_metrics.csv
        ├── threshold_sweep.csv
        └── กราฟรายโมเดล
```

## วิธีใช้บนเซิร์ฟเวอร์

1. คัดลอกโฟลเดอร์ `model_comparison` ไปวางข้าง `xrayfsod_outputs`
2. ติดตั้ง dependency ด้วย `pip install -r requirements-local.txt`
3. ตรวจ `config.json` โดยเฉพาะ `ground_truth` และ `model_root`
4. เปิด `compare_models.ipynb` แล้วกด **Run All**
5. เปิด `http://localhost:8501` หรือกดลิงก์จาก cell สุดท้าย

หรือเปิดเว็บโดยไม่ใช้ notebook:

```bash
cd model_comparison
streamlit run app.py
```

บนหน้าเว็บตั้ง `Model root` และ ground-truth JSON แล้วกด **ประเมินโมเดล** ได้ทันที

## Deploy ด้วย Docker

สร้าง image:

```bash
cd model_comparison
docker build -t xray-model-comparison .
```

รันบนเซิร์ฟเวอร์ โดยเปลี่ยน `/server/xrayfsod_outputs` เป็น path จริง:

```bash
docker run --rm -p 8501:8501 \
  -v /server/xrayfsod_outputs:/data/xrayfsod_outputs:ro \
  -e MODEL_ROOT=/data/xrayfsod_outputs \
  -e GT_JSON=/data/xrayfsod_outputs/_prepared_full20/annotations/test.json \
  xray-model-comparison
```

จากนั้นเปิด `http://<IP-เซิร์ฟเวอร์>:8501` และอนุญาต port 8501 ใน firewall เฉพาะเครือข่ายที่ต้องการใช้งาน

ค่าเริ่มต้นค้นหาไฟล์ที่ notebook เทรนเดิมสร้างไว้:

```text
xrayfsod_outputs/<ชื่อโมเดล>/test_predictions.json
```

จึงไม่ต้องย้าย weight หรือรัน inference ซ้ำ ถ้ามี `test_predictions.json` อยู่แล้ว

ชุดที่เลือกไว้ใน `model_runs` ตอนนี้มีเฉพาะ:

```text
EfficientNet-1
ResNet50-1
RT-DETR-1
VGG16-4
YOLOv11-1
```

`VGG16-1`, `VGG16-2` และ `VGG16-3` จะไม่ถูกนำมาคำนวณ แม้จะอยู่ใน `model_root` ก็ตาม หากต้องการเพิ่มหรือลดโมเดล ให้แก้เฉพาะรายการ `model_runs`

ในแต่ละ run ตัวโปรแกรมจะตรวจไฟล์ดังนี้:

```text
<model_root>/<ชื่อรัน>/test_predictions.json  # ใช้คำนวณคะแนน
<model_root>/<ชื่อรัน>/weights/best.pt        # ตรวจและแสดง path ประกอบรายงาน
```

ถ้ามีเฉพาะ `best.pt` แต่ไม่มี `test_predictions.json` โปรแกรมจะหยุดและแจ้งชื่อ run ที่ขาด เพราะ EfficientNet/ResNet/VGG, YOLO และ RT-DETR ต้องใช้ model builder และ preprocessing ต่างกัน โดย notebook เทรน `XrayFSOD_5Models_Sequential.ipynb` จะสร้าง `test_predictions.json` ให้หลังประเมิน test set

## Metric ที่แสดง

- Precision, Recall และ Micro F1 ที่ confidence/IoU เดียวกัน
- Macro F1 เพื่อให้เห็นผลเฉลี่ยระหว่างคลาส
- mAP@0.50:0.95, AP@0.50 และ AR@100 ตาม COCO
- Best F1 จาก confidence threshold sweep พร้อม threshold ที่เหมาะกับแต่ละโมเดล
- F1 รายคลาส และกราฟ Precision/Recall/F1 ตาม confidence
- เวลา inference ต่อภาพ หากระบุ `inference_ms_per_image` ใน `model.json`
- Learning curve ของ training loss, validation loss, mAP, precision และ recall เท่าที่มีในไฟล์เทรน

## ไฟล์สำหรับ Learning curve

ระบบหาไฟล์ที่ระดับเดียวกับโฟลเดอร์ `weights` โดยอัตโนมัติ:

```text
EfficientNet-1/history.csv  # Torchvision: EfficientNet, ResNet50, VGG16
ResNet50-1/history.csv
VGG16-4/history.csv
RT-DETR-1/results.csv       # Ultralytics
YOLOv11-1/results.csv
```

สำหรับ `history.csv` รองรับคอลัมน์ `epoch`, `train_loss`, `val_loss`, `val_AP_50_95`, `val_AP_50` ส่วน `results.csv` จะอ่านคอลัมน์มาตรฐานของ Ultralytics และรวม loss components ให้เอง หากไม่พบไฟล์ เว็บยังแสดง F1/mAP ได้ตามปกติ แต่จะแจ้งว่าโมเดลใดไม่มี Learning curve

ไม่ใช้ accuracy เป็น metric หลัก เพราะ object detection มี true-negative/background จำนวนไม่แน่นอน ทำให้ accuracy ดูดีเกินจริงได้

## เงื่อนไขเพื่อให้เปรียบเทียบยุติธรรม

- ทุกโมเดลต้องทำนายรูปชุดเดียวกัน และใช้ ground-truth JSON ไฟล์เดียวกัน
- `image_id`, `category_id` และลำดับคลาสต้องตรงกัน
- กล่องใน `predictions.json` ต้องเป็น COCO `xywh`
- ใช้ IoU, confidence และ max detections ชุดเดียวกัน
- หากเทียบความเร็ว ต้องวัดบน hardware, batch size และ image size เดียวกัน

รายละเอียดรูปแบบไฟล์อยู่ที่ `model_inputs/README.md`
