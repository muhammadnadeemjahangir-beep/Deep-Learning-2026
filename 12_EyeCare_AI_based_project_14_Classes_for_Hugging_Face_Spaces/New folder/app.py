import gradio as gr
from ultralytics import YOLO
from PIL import Image
import os
import glob

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "model/best.pt"
FALLBACK   = "yolov8n.pt"

def load_model():
    if os.path.exists(MODEL_PATH):
        print(f"✅  Loaded trained model: {MODEL_PATH}")
        return YOLO(MODEL_PATH)
    print(f"⚠️  best.pt not found – using base YOLOv8n (untrained for eye disease).")
    return YOLO(FALLBACK)

model = load_model()

# ── Disease classes from training ──────────────────────────────────────────────
# Trained on Roboflow Eye-Disease Dataset v5
# Classes: Cataract | Jaundice | Normal_Eye | Pterygium
CLASS_INFO = {
    "Cataract":    {"emoji": "🔵", "color": "#4B9FFF", "desc": "Clouding of the eye lens"},
    "Jaundice":    {"emoji": "🟡", "color": "#FFD54B", "desc": "Yellowing of the sclera (whites)"},
    "Normal_Eye":  {"emoji": "🟢", "color": "#4BFF91", "desc": "No condition detected"},
    "Pterygium":   {"emoji": "🔴", "color": "#FF4B4B", "desc": "Growth of tissue on the cornea"},
}
DEFAULT_COLOR = "#BBBBBB"

# ── Inference ──────────────────────────────────────────────────────────────────
def detect_eye_disease(image, conf_threshold, iou_threshold):
    if image is None:
        return None, "⚠️  No image uploaded. Please upload a retinal / eye image."

    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        save=False,
        verbose=False,
    )

    detected = []
    output_image = results[0].plot()

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = results[0].names[cls_id]
        info   = CLASS_INFO.get(label, {})
        emoji  = info.get("emoji", "🔵")
        desc   = info.get("desc",  "")
        bar    = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        detected.append(
            f"  {emoji} {label}"
            + (f"  —  {desc}" if desc else "")
            + f"\n     Confidence: {bar} {conf:.1%}"
        )

    if not detected:
        text = (
            "✅  NO CONDITION DETECTED\n"
            "──────────────────────────────\n"
            f"No findings at {conf_threshold:.0%} confidence.\n"
            "Try lowering the threshold if\n"
            "you expect a finding.\n\n"
            "⚕️  Always consult a specialist."
        )
    else:
        text = (
            f"⚠️  {len(detected)} CONDITION(S) DETECTED\n"
            "──────────────────────────────\n"
            + "\n\n".join(detected)
            + "\n\n──────────────────────────────\n"
            "⚕️  Screening only.\n"
            "    See an ophthalmologist."
        )

    return Image.fromarray(output_image), text


# ── CSS ────────────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body {
    min-height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}
*, body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box !important;
}
.gradio-container, .gradio-container * {
    color: #cfe3fa;
}
.gradio-container label span,
.gradio-container .label-wrap span,
.gradio-container [data-testid="block-info"],
.gradio-container .block-info,
.gradio-container span.svelte-1gfkn6j,
.gradio-container .info {
    color: #8fb8e6 !important;
}
.gradio-container {
    background: linear-gradient(135deg, #060d1a 0%, #0a1628 60%, #081520 100%) !important;
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}
footer, .svelte-1ax1toq, #footer { display: none !important; }
.main { padding: 0 !important; }
.contain { max-width: 100% !important; padding: 0 12px !important; }

/* ── App wrapper ── */
#app-wrapper {
    display: flex;
    flex-direction: column;
    padding: 10px 16px 16px;
    gap: 8px;
}
#app-wrapper > * { flex-shrink: 0; }
#app-wrapper > #main-grid { flex: 1 1 auto; min-height: 480px; }

/* ── Header ── */
#app-header {
    background: linear-gradient(90deg, #0d3b6e, #1a5276, #0d3b6e);
    border: 1px solid rgba(33,150,243,0.5);
    border-radius: 12px;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
    box-shadow: 0 0 30px rgba(33,150,243,0.15);
}
#app-header .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
#app-header .eye-icon { font-size: 2rem; line-height: 1; }
#app-header h1 {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #fff !important;
    margin: 0 !important;
}
#app-header .subtitle {
    font-size: 0.72rem;
    color: #90caf9;
    margin: 2px 0 0;
}
.pulse-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #4caf50;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(76,175,80,0.7); }
    70%  { box-shadow: 0 0 0 6px rgba(76,175,80,0); }
    100% { box-shadow: 0 0 0 0 rgba(76,175,80,0); }
}
#app-header .header-right { display: flex; align-items: center; gap: 8px; }

/* ── Share button ── */
#share-btn {
    background: linear-gradient(90deg, #00695c, #00897b) !important;
    border: 1px solid #4db6ac !important;
    border-radius: 8px !important;
    color: white !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 7px 14px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    white-space: nowrap;
}
#share-btn:hover {
    background: linear-gradient(90deg, #00796b, #00897b) !important;
    box-shadow: 0 4px 14px rgba(0,137,123,0.4) !important;
    transform: translateY(-1px) !important;
}
#share-toast {
    display: none;
    position: fixed;
    bottom: 24px; left: 50%;
    transform: translateX(-50%);
    background: #1b5e20;
    border: 1px solid #4caf50;
    color: #a5d6a7;
    padding: 10px 22px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    z-index: 9999;
}

/* ── Stats strip ── */
#stats-strip {
    display: flex;
    gap: 8px;
    flex-shrink: 0;
}
.stat-chip {
    flex: 1;
    background: rgba(13,59,110,0.45);
    border: 1px solid rgba(33,150,243,0.2);
    border-radius: 9px;
    padding: 7px 10px;
    text-align: center;
}
.stat-num { font-size: 1rem; font-weight: 700; color: #64b5f6 !important; line-height: 1.2; }
.stat-lbl { font-size: 0.62rem; color: #c3d7f0 !important; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }

/* ── Main grid ── */
#main-grid {
    display: flex !important;
    gap: 10px;
    flex: 1 1 auto;
    min-height: 480px;
    flex-wrap: wrap;
}
#main-grid > * {
    min-width: 320px !important;
    flex: 1 1 380px !important;
}

/* ── Panels ── */
.panel {
    background: rgba(10,22,40,0.7) !important;
    border: 1px solid rgba(33,150,243,0.25) !important;
    border-radius: 12px !important;
    padding: 10px !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 480px !important;
    transition: border-color 0.3s;
    backdrop-filter: blur(8px);
}
.panel:hover { border-color: rgba(33,150,243,0.5) !important; }
#left-panel  { flex: 1 1 380px !important; }
#right-panel { flex: 1 1 380px !important; }
.panel > * { flex-shrink: 0; }

.panel-title {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    color: #5c8fbe !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    margin: 0 0 6px !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid rgba(33,150,243,0.15) !important;
}

/* ── Image components ── */
#img-input, #img-output {
    flex: 1 1 auto !important;
    min-height: 260px !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}
#img-input > *, #img-output > * {
    flex: 1 1 auto !important;
    min-height: 260px !important;
}
#img-input [data-testid="image"], #img-output [data-testid="image"] {
    min-height: 260px !important;
}
#img-input [data-testid="image"] {
    background: rgba(5,14,28,0.8) !important;
    border: 2px dashed rgba(33,150,243,0.4) !important;
    border-radius: 8px !important;
    height: 100% !important;
}
#img-input [data-testid="image"]:hover {
    border-color: #2196f3 !important;
    background: rgba(33,150,243,0.05) !important;
}
#img-output [data-testid="image"] {
    background: rgba(25,45,75,0.55) !important;
    border: 2px solid rgba(33,150,243,0.4) !important;
    border-radius: 8px !important;
    height: 100% !important;
    box-shadow: 0 0 12px rgba(33,150,243,0.12) !important;
}
.image-frame img, #img-input img, #img-output img {
    object-fit: contain !important;
    max-height: 100% !important;
    width: 100% !important;
}

/* ── Settings accordion ── */
.settings-acc {
    flex-shrink: 0 !important;
    border: 1px solid rgba(33,150,243,0.2) !important;
    border-radius: 8px !important;
    background: rgba(5,14,28,0.6) !important;
    margin-top: 6px !important;
}
.settings-acc button, .settings-acc .label-wrap span {
    color: #b3d9ff !important;
}
.settings-acc span, .settings-acc label, .settings-acc p {
    color: #b3d9ff !important;
}
.settings-acc input[type="number"] {
    color: #b3d9ff !important;
    background: rgba(5,14,28,0.8) !important;
}
.settings-acc .info, .settings-acc [data-testid="block-info"] {
    color: #7a9bc2 !important;
}

/* ── Analyse button ── */
#analyse-btn {
    background: linear-gradient(90deg, #1565c0, #0d47a1) !important;
    border: 1px solid #2196f3 !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 3px 12px rgba(21,101,192,0.35) !important;
    width: 100% !important;
    flex-shrink: 0 !important;
    margin-top: 8px !important;
}
#analyse-btn:hover {
    background: linear-gradient(90deg, #1976d2, #1565c0) !important;
    box-shadow: 0 5px 18px rgba(33,150,243,0.45) !important;
    transform: translateY(-1px) !important;
}

.file-hint {
    text-align: center;
    font-size: 0.67rem;
    color: #9fb8d6 !important;
    margin: 4px 0 0;
    flex-shrink: 0;
}

/* ── Prediction box ── */
#pred-box {
    flex: 1 1 auto !important;
    min-height: 160px !important;
    margin-top: 8px !important;
}
#pred-box textarea {
    background: rgba(3,10,22,0.85) !important;
    color: #b3d9ff !important;
    border: 1px solid rgba(33,150,243,0.25) !important;
    border-radius: 8px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.8rem !important;
    line-height: 1.65 !important;
    min-height: 160px !important;
    resize: vertical !important;
    padding: 10px 12px !important;
}

/* ── Bottom bar ── */
#bottom-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255,152,0,0.07);
    border: 1px solid rgba(255,152,0,0.25);
    border-radius: 9px;
    padding: 7px 14px;
    flex-shrink: 0;
}
#bottom-bar .disclaimer-text {
    font-size: 0.71rem;
    color: #ffb74d;
    line-height: 1.4;
}
#bottom-bar .footer-text {
    font-size: 0.66rem;
    color: #aabfdc !important;
    white-space: nowrap;
    margin-left: 16px;
    text-align: right;
}

/* ── Legend chips ── */
.legend-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 6px;
    flex-shrink: 0;
}
.legend-chip {
    font-size: 0.65rem;
    padding: 3px 8px;
    border-radius: 20px;
    font-weight: 500;
    color: #000;
    opacity: 0.85;
}
"""

# ── Theme ──────────────────────────────────────────────────────────────────────
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="transparent",
    block_background_fill="transparent",
    block_border_color="transparent",
    block_label_text_color="#5c8fbe",
    input_background_fill="rgba(5,14,28,0.8)",
    input_border_color="rgba(33,150,243,0.3)",
    button_primary_background_fill="linear-gradient(90deg,#1565c0,#0d47a1)",
    button_primary_text_color="#ffffff",
)

# ── Example images ─────────────────────────────────────────────────────────────
example_paths = sorted(glob.glob("examples/*.jpg")) + sorted(glob.glob("examples/*.png"))
examples = [[p, 0.40, 0.45] for p in example_paths] if example_paths else None

# ── UI ─────────────────────────────────────────────────────────────────────────
with gr.Blocks(theme=theme, css=css, title="EyeCare AI") as demo:

    gr.HTML('<div id="share-toast">✅ Link copied to clipboard!</div>')

    with gr.Column(elem_id="app-wrapper"):

        # Header
        gr.HTML("""
        <div id="app-header">
            <div class="header-left">
                <span class="eye-icon">👁️</span>
                <div>
                    <h1>EyeCare AI</h1>
                    <p class="subtitle">
                        <span class="pulse-dot"></span>
                        Smart Eye Disease Detection &nbsp;·&nbsp; YOLOv8 Deep Learning &nbsp;·&nbsp; NED University of Engineering & Technology
                    </p>
                </div>
            </div>
            <div class="header-right">
                <button id="share-btn" onclick="
                    (function(){
                        var url = window.location.href;
                        try { if (window.top && window.top.location.href) { url = window.top.location.href; } } catch(e) {}
                        function showToast(){
                            const t = document.getElementById('share-toast');
                            t.style.display = 'block';
                            setTimeout(()=>{ t.style.display='none'; }, 2500);
                        }
                        try {
                            if (navigator.clipboard && navigator.clipboard.writeText) {
                                navigator.clipboard.writeText(url).then(showToast).catch(()=>{ prompt('Copy this link:', url); });
                            } else {
                                var ta = document.createElement('textarea');
                                ta.value = url;
                                ta.style.position = 'fixed';
                                ta.style.opacity = '0';
                                document.body.appendChild(ta);
                                ta.focus(); ta.select();
                                var ok = false;
                                try { ok = document.execCommand('copy'); } catch(e) {}
                                document.body.removeChild(ta);
                                if (ok) { showToast(); } else { prompt('Copy this link:', url); }
                            }
                        } catch(err) {
                            prompt('Copy this link:', url);
                        }
                    })();
                ">🔗 Share App</button>
            </div>
        </div>
        """)

        # Stats strip
        gr.HTML("""
        <div id="stats-strip">
            <div class="stat-chip">
                <div class="stat-num">YOLOv8n</div>
                <div class="stat-lbl">Architecture</div>
            </div>
            <div class="stat-chip">
                <div class="stat-num">88.4%</div>
                <div class="stat-lbl">mAP@50</div>
            </div>
            <div class="stat-chip">
                <div class="stat-num">4</div>
                <div class="stat-lbl">Eye Conditions</div>
            </div>
            <div class="stat-chip">
                <div class="stat-num">100</div>
                <div class="stat-lbl">Epochs Trained</div>
            </div>
            <div class="stat-chip">
                <div class="stat-num">3 988</div>
                <div class="stat-lbl">Training Images</div>
            </div>
        </div>
        """)

        # Main grid
        with gr.Row(elem_id="main-grid", equal_height=True):

            # Left – Input
            with gr.Column(elem_id="left-panel", elem_classes="panel"):
                gr.HTML('<p class="panel-title">📤 Upload Eye Image</p>')
                image_input = gr.Image(
                    type="pil",
                    label="",
                    show_label=False,
                    elem_id="img-input",
                )
                gr.HTML('<p class="file-hint">JPG · PNG · WEBP · BMP</p>')

                with gr.Accordion("⚙️ Detection Settings", open=False, elem_classes="settings-acc"):
                    conf_slider = gr.Slider(
                        minimum=0.10, maximum=0.95, value=0.40, step=0.05,
                        label="Confidence Threshold",
                        info="Min score to show a detection",
                    )
                    iou_slider = gr.Slider(
                        minimum=0.10, maximum=0.95, value=0.45, step=0.05,
                        label="IoU Threshold (NMS)",
                        info="Controls overlap suppression",
                    )

                analyse_btn = gr.Button(
                    "🔬  Analyse Image",
                    elem_id="analyse-btn",
                    variant="primary",
                )

                # Class legend
                gr.HTML("""
                <div class="legend-row">
                    <span class="legend-chip" style="background:#4BFF91;">🟢 Normal Eye</span>
                    <span class="legend-chip" style="background:#FF4B4B;">🔴 Pterygium</span>
                    <span class="legend-chip" style="background:#4B9FFF;">🔵 Cataract</span>
                    <span class="legend-chip" style="background:#FFD54B;">🟡 Jaundice</span>
                </div>
                """)

            # Right – Output
            with gr.Column(elem_id="right-panel", elem_classes="panel"):
                gr.HTML('<p class="panel-title">🔍 Detection Result</p>')
                image_output = gr.Image(
                    label="",
                    show_label=False,
                    elem_id="img-output",
                )
                gr.HTML('<p class="panel-title" style="margin-top:8px;">📋 Prediction Report</p>')
                prediction_output = gr.Textbox(
                    label="",
                    lines=8,
                    show_label=False,
                    elem_id="pred-box",
                    placeholder="Results will appear here after analysis…",
                )

        # Bottom bar
        gr.HTML("""
        <div id="bottom-bar">
            <span class="disclaimer-text">
                ⚕️ Medical Disclaimer:
                EyeCare AI is for research &amp; screening purposes only.
                Not a substitute for professional ophthalmological diagnosis.
                Always consult a licensed eye-care specialist. Developed by Bareera Rehan & Muhammad Nadeem
            </span>
            <span class="footer-text">
                EyeCare AI &nbsp;·&nbsp; YOLOv8 &amp; Gradio<br>
                Hugging Face Spaces
            </span>
        </div>
        """)

    # Events
    analyse_btn.click(
        fn=detect_eye_disease,
        inputs=[image_input, conf_slider, iou_slider],
        outputs=[image_output, prediction_output],
    )
    image_input.change(
        fn=detect_eye_disease,
        inputs=[image_input, conf_slider, iou_slider],
        outputs=[image_output, prediction_output],
    )

    if examples:
        gr.Examples(
            examples=examples,
            inputs=[image_input, conf_slider, iou_slider],
            outputs=[image_output, prediction_output],
            fn=detect_eye_disease,
            cache_examples=True,
            label="📸 Sample Eye Images",
        )

if __name__ == "__main__":
    demo.launch(show_error=True)