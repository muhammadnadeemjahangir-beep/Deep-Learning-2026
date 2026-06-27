---
title: EyeCare AI - Eye Disease Detection
emoji: 👁️
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 👁️ EyeCare AI — Smart Eye Disease Detection

EyeCare AI is a YOLOv8-based object detection app that screens eye images for
multiple eye conditions. It was trained on a Roboflow eye-disease dataset
(100 epochs) and is served here with a Gradio interface.

## How it works

1. Upload an eye / retinal image (JPG, PNG, WEBP, or BMP).
2. The app runs YOLOv8 inference and draws bounding boxes for any detected
   conditions, along with a confidence score for each.
3. Adjust the **Confidence Threshold** and **IoU Threshold (NMS)** sliders
   under "Detection Settings" to tune sensitivity.

## ⚠️ Required: add your trained weights before deploying

This Space ships **without** a trained model checkpoint. You must add your
own `best.pt` (the file your training notebook produces and downloads via
`files.download("runs/detect/.../weights/best.pt")`) at:

```
model/best.pt
```

If `model/best.pt` is not present, the app silently falls back to the
untrained base `yolov8n.pt`, which will **not** detect eye diseases
correctly — it will only run generic COCO object detection. To deploy with
your trained weights:

1. Train the model using the included notebook
   (`100_epochs_with_14_classes_Eye_Disease_Detection_model.ipynb`).
2. Download the resulting `best.pt`.
3. Create a `model/` folder in this Space's repo and upload `best.pt` into it
   (drag-and-drop via the Hugging Face web UI, or `git add model/best.pt`
   if pushing via git — note `best.pt` files are typically tens of MB, so
   make sure Git LFS is enabled for `*.pt` files on this repo).

The app automatically reads the class names from whatever model you load
(`model.names`), so it adapts to however many classes your dataset has —
no code changes needed even if you retrain with a different class set.

## Optional: example images

Drop sample eye images into an `examples/` folder (as `.jpg` or `.png`) to
have them appear as clickable examples in the UI.

## Files in this Space

| File              | Purpose                                            |
| ------------------ | --------------------------------------------------- |
| `app.py`           | Gradio app — UI + YOLOv8 inference                 |
| `requirements.txt` | Python dependencies                                |
| `packages.txt`     | System (apt) dependencies required by OpenCV       |
| `model/best.pt`    | **You must add this** — your trained YOLOv8 weights |
| `examples/`        | Optional sample images shown in the UI             |

## Medical disclaimer

EyeCare AI is for research and screening purposes only and is not a
substitute for professional ophthalmological diagnosis. Always consult a
licensed eye-care specialist.
