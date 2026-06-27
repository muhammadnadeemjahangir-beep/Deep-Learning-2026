---
title: EyeCare AI - Eye Disease Detection
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: cc-by-4.0
---

# 👁️ EyeCare AI — Smart Eye Disease Detection

EyeCare AI is a YOLOv8-based object detection app that screens eye images for
common eye conditions. Upload a close-up eye/retinal photo and the model will
draw bounding boxes around any detected conditions along with a confidence
score for each.

## Model details

- **Architecture:** YOLOv8n (nano), fine-tuned with [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Training data:** [Eye Disease dataset](https://universe.roboflow.com/fabricdefect-g0o4o/eye-disease-yg5lp/dataset/4) (Roboflow, v4), licensed CC BY 4.0
- **Epochs:** 100, image size 640×640
- **Classes (14):** Bacterial Conjunctivitis, Cataract, Chalazion, Conjunctivitis,
  Foreign Object, Hordeolum, Katarak, Normal, Pinguecula, Ptrigio-Pinguecula,
  Viral Conjunctivitis, cataract, normal, normal eye
- **Validation mAP@50:** ≈ 0.36 (mAP@50-95 ≈ 0.23)

> Some classes overlap in naming (e.g. `Cataract` / `cataract` / `Katarak`) because
> the source dataset merges multiple sub-collections. The app displays whatever
> label the model predicts as-is.

## How it works

1. Upload an eye image (JPG / PNG / WEBP / BMP).
2. Optionally tweak the **Confidence Threshold** and **IoU Threshold** under
   "Detection Settings".
3. Click **Analyse Image** (or it runs automatically after upload).
4. View the annotated image and a text report of detected conditions with
   confidence scores.

## Files

| File | Purpose |
|---|---|
| `app.py` | Gradio app (UI + inference) |
| `model/best.pt` | Trained YOLOv8 weights — **you must add this file** (see below) |
| `requirements.txt` | Python dependencies for the Space |
| `examples/` | Optional sample images shown in the UI |

## Deploying your trained weights

This repo does **not** include the trained `best.pt` weights (binary model
weights aren't stored inside the training notebook). After training in your
notebook, download `best.pt` from
`runs/detect/eye_disease_detection/weights/best.pt` and place it at:

```
model/best.pt
```

If `model/best.pt` is missing, the app automatically falls back to the
un-trained base `yolov8n.pt` so it doesn't crash — but it won't recognize eye
diseases until you add your real weights.

Because `best.pt` is a binary file >5MB, track it with Git LFS before pushing
to your Space (see `.gitattributes`, already configured for `*.pt`).

## ⚕️ Disclaimer

This tool is for **research and screening purposes only**. It is **not** a
substitute for professional ophthalmological diagnosis. Always consult a
licensed eye-care specialist for medical advice.
