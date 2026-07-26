# X-ray FSOD Model Report

Streamlit dashboard comparing EfficientNet, ResNet50, RT-DETR, VGG16, and YOLOv11. It includes:

- a read-only comparison report with headline winners, COCO/F1 metrics, learning curves, training settings, and per-class results;
- an explicit-submit YOLOv11 inference demo with upload-size and pixel-count limits; and
- an in-browser thesis reader with a PDF download.

## Safe GitHub deployment

Upload **this `model_comparison` folder only** as the repository root. Do not upload the parent project, `xrayfsod_outputs`, datasets, additional model weights, prediction JSON, `.env`, or `secrets.toml`.

The committed `public_report/` contains only sanitized CSV/PNG report artifacts. The selected YOLOv11 weight is stored at `models/yolov11_xrayfsod_best.pt`, and the current thesis PDF is stored at `thesis/thesis_latest.pdf`. On Streamlit Community Cloud the report remains read-only while the model demo accepts one uploaded X-ray image at a time (maximum 10 MB and 25 million pixels).

Deploy with Streamlit Community Cloud:

1. Create a GitHub repository and put the contents of this folder at its root.
2. In Streamlit Community Cloud, create an app from that repository.
3. Select the branch and use `app.py` as the entrypoint.
4. Deploy. No secrets are required for this public report.

To refresh the published data after evaluating locally:

```bash
python -m pip install -r requirements-local.txt
python build_public_report.py ../xrayfsod_outputs
```

Then commit only the updated `public_report/` files. See [README_TH.md](README_TH.md) for full Thai instructions.
