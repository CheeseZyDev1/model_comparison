# X-ray FSOD Model Report

Read-only Streamlit dashboard comparing EfficientNet, ResNet50, RT-DETR, VGG16, and YOLOv11. The app also includes an in-browser reader and download link for the current thesis PDF.

## Safe GitHub deployment

Upload **this `model_comparison` folder only** as the repository root. Do not upload the parent project, `xrayfsod_outputs`, datasets, model weights, prediction JSON, `.env`, or `secrets.toml`.

The committed `public_report/` contains only sanitized CSV/PNG report artifacts. The current thesis PDF is stored at `thesis/thesis_latest.pdf`. On Streamlit Community Cloud the app detects that local model files are absent and automatically runs in read-only mode: no evaluation button and no filesystem path inputs.

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
