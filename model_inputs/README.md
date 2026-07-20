# Model input contract

Create one subfolder per model. Folder names must be unique and become the model names in reports.

```text
model_inputs/
  YOLOv11-1/
    predictions.json       # required: COCO detection result list
    model.json             # optional: display name and metadata
    weights/
      best.pt              # optional: kept here for organization only
  RT-DETR-1/
    predictions.json
    model.json
    weights/
      best.pt
```

`predictions.json` must be a JSON list. Each detection must use this shape:

```json
[
  {
    "image_id": 1,
    "category_id": 3,
    "bbox": [120.5, 80.0, 50.0, 70.0],
    "score": 0.92
  }
]
```

The bounding box format is COCO `xywh`: `[left, top, width, height]`. `image_id` and `category_id` must match the ground-truth COCO JSON exactly.

Optional `model.json`:

```json
{
  "display_name": "YOLOv11-L run 1",
  "framework": "ultralytics",
  "weight_file": "weights/best.pt",
  "inference_ms_per_image": 12.4,
  "notes": "640 px, 50 epochs"
}
```

Weights are not loaded by the comparison notebook. Different frameworks require different model-building code, preprocessing and class mappings. Generate `predictions.json` once with the original training/inference code, then compare every model using the same test set and evaluator.

