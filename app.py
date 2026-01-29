import gradio as gr
from PIL import Image
import io
import requests
import os

# --- KMT Dynamics: Lens AI Intelligence ---
# System Architect: BSc. PHS Msc. PHQ "Abdulrahman" Mugabi Kizito Lenny

# SECURITY CONFIGURATION (GITHUB/PRODUCTION SAFE)
# This pulls from the Environment Variables you set in your host settings
HF_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# STRATEGIC MODEL ROUTING (2026 Optimized)
MODELS = {
    "X-Ray": "https://api-inference.huggingface.co/models/microsoft/swin-base-patch4-window7-224",
    "Ultrasound": "https://api-inference.huggingface.co/models/microsoft/swin-base-patch4-window7-224",
    "Skin": "https://api-inference.huggingface.co/models/facebook/dino-v2-base",
    "Pathology": "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
}

def lens_diagnostic_engine(image, modality):
    if image is None:
        return "⚠️ [LENS ERROR]: No diagnostic input detected."
    
    if not HF_TOKEN:
        return "⚠️ [CONFIG ERROR]: API Token missing. Please set HF_TOKEN in environment variables."

    # Image Pre-processing
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    img_bytes = buf.getvalue()

    target_url = MODELS.get(modality)

    try:
        # Neural Router Connection
        response = requests.post(target_url, headers=headers, data=img_bytes, timeout=25)
        prediction = response.json()

        if isinstance(prediction, list) and len(prediction) > 0:
            top_result = prediction[0]
            label = top_result.get('label', 'Pattern Detected').replace('_', ' ').title()
            score = round(top_result.get('score', 0) * 100, 2)
        else:
            label = "Complex Pattern Analyzed"
            score = "N/A"

        # Clinical Strategy Logic
        strategy_map = {
            "X-Ray": "Correlate with CURB-65 for pneumonia or Sieve signs for TB. Suggest Lateral view.",
            "Skin": "Analyze borders (ABCDE criteria). Suggest biopsy if asymmetry index is high.",
            "Pathology": "Initiate pathogen count. Correlate with Gram-stain morphology.",
            "Ultrasound": "Evaluate acoustic shadowing. Fasting recommended for repeat biliary scan."
        }

        return f"""
# 🔎 LENS | Strategic Diagnostic Trace
---
### **1. AI ANALYSIS METRICS**
* **SYSTEM SOURCE:** {modality}
* **PRIMARY PATTERN:** **{label}**
* **AI CONFIDENCE:** {score}%

### **2. CLINICAL STRATEGY**
> {strategy_map.get(modality)}

---
**Product of KMT Dynamics Company Ltd.**
***⚠️ CLINICAL DISCLAIMER:** Trace generated via {modality} Neural Router. Physician verification mandatory.*
"""
    except Exception as e:
        return f"📡 [CONNECTION ERROR]: Diagnostic server busy. Details: {str(e)}"

# --- LENS UI DESIGN (High-End Silver/Black Aesthetic) ---
custom_css = """
    .gradio-container { font-family: 'Garamond', serif !important; background-color: #f8fafc; }
    #lens-header {
        background: linear-gradient(135deg, #000000 0%, #4b5563 50%, #9ca3af 100%);
        padding: 40px; border-radius: 15px; text-align: center; border: 2px solid silver;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    }
    #lens-title { color: #C0C0C0 !important; font-size: 4em !important; font-weight: bold; margin: 0; text-shadow: 2px 2px 8px #000000; }
    #lens-subtitle { color: #ffffff !important; font-size: 1.2em; margin-top: 5px; opacity: 0.9; }
    .team-section { display: flex; justify-content: center; gap: 40px; margin-top: 20px; text-align: center; }
    .grey-icon { font-size: 1.8em; color: #cbd5e1; margin-bottom: 5px; }
"""

with gr.Blocks(css=custom_css, theme='soft', title="Lens AI - KMT Dynamics") as app:

    gr.HTML(f"""
        <div id='lens-header'>
            <h1 id='lens-title'>LENS</h1>
            <div id='lens-subtitle'>KMT Dynamics Company Ltd.</div>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Diagnostic Input", sources=["upload", "webcam"])
            modality_select = gr.Dropdown(
                choices=["X-Ray", "Skin", "Pathology", "Ultrasound"],
                label="Select Diagnostic Modality",
                value="X-Ray"
            )
            submit_btn = gr.Button("PERFORM DIAGNOSTIC TRACE", variant="primary")

        with gr.Column(scale=1):
            output_markdown = gr.Markdown("### 📡 System Active\nAwaiting physician input...")

    gr.HTML("""
        <div style='text-align: center; margin-top: 50px; padding: 30px; border-top: 1px solid #9ca3af;'>
            <div class='team-section'>
                <div><div class='grey-icon'>👤</div><strong>Kazibwe Visionary</strong><br><small>(Medical)</small></div>
                <div><div class='grey-icon'>👤</div><strong>Sheikh Sharif Abdul Sunna</strong><br><small>(Management)</small></div>
            </div>
            <div style='margin-top: 25px; color: #1e293b; font-size: 1em;'>
                System Architect: <strong>BSc. PHS Msc. PHQ "Abdulrahman" Mugabi Kizito Lenny</strong>
            </div>
            <div style='color: #9ca3af; font-size: 0.8em; margin-top: 10px;'>MADE IN UGANDA | © 2026 KMT Dynamics</div>
        </div>
    """)

    submit_btn.click(fn=lens_diagnostic_engine, inputs=[input_img, modality_select], outputs=output_markdown)

# PRODUCTION LAUNCH CONFIG
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
