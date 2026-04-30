import streamlit as st
import numpy as np
from PIL import Image
import io
import base64

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --card: #1a2236;
    --accent: #00e5ff;
    --accent2: #7c3aed;
    --danger: #ef4444;
    --success: #22c55e;
    --warn: #f59e0b;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: rgba(0,229,255,0.15);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg) !important; }

/* Hide default streamlit header */
header[data-testid="stHeader"] { background: transparent !important; }

/* Title */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 4px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Upload box */
.upload-section {
    border: 1.5px dashed var(--border);
    border-radius: 16px;
    padding: 2rem;
    background: var(--card);
    margin: 1.5rem 0;
}

/* Result cards */
.result-card {
    border-radius: 14px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid var(--border);
    background: var(--card);
}
.result-card.tumor {
    border-color: rgba(239,68,68,0.4);
    background: linear-gradient(135deg, rgba(239,68,68,0.08), var(--card));
}
.result-card.no-tumor {
    border-color: rgba(34,197,94,0.4);
    background: linear-gradient(135deg, rgba(34,197,94,0.08), var(--card));
}
.result-card.uncertain {
    border-color: rgba(245,158,11,0.4);
    background: linear-gradient(135deg, rgba(245,158,11,0.08), var(--card));
}

.verdict {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.verdict.tumor { color: var(--danger); }
.verdict.no-tumor { color: var(--success); }
.verdict.uncertain { color: var(--warn); }

/* Confidence bar */
.conf-label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
.conf-value { font-family: 'Space Mono', monospace; font-size: 1.1rem; }
.conf-bar-bg { background: rgba(255,255,255,0.07); border-radius: 99px; height: 8px; margin: 6px 0 14px; }
.conf-bar-fill { height: 8px; border-radius: 99px; }

/* Disclaimer */
.disclaimer {
    font-size: 0.75rem;
    color: var(--muted);
    border: 1px solid rgba(100,116,139,0.2);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-top: 1.5rem;
    background: rgba(100,116,139,0.05);
}

/* Streamlit button */
.stButton > button {
    background: linear-gradient(135deg, var(--accent2), #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stFileUploader {
    background: var(--card) !important;
    border-radius: 14px !important;
}

section[data-testid="stFileUploadDropzone"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
}

div[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid var(--border);
}

.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}
.metric-box {
    flex: 1;
    min-width: 120px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-box .m-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
.metric-box .m-value { font-family: 'Space Mono', monospace; font-size: 1.1rem; color: var(--accent); margin-top: 2px; }

hr.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load a pre-trained brain tumor classification model.
    Tries to load a local model; falls back to a demo mode if unavailable.
    """
    try:
        import tensorflow as tf  # noqa: F401
        # If you have a saved model, load it here:
        # model = tf.keras.models.load_model("brain_tumor_model.h5")
        # return model, "tensorflow"
        raise ImportError("No local model file found")
    except Exception:
        pass

    try:
        import torch  # noqa: F401
        # model = torch.load("brain_tumor_model.pt")
        # return model, "pytorch"
        raise ImportError("No local model file found")
    except Exception:
        pass

    # Demo mode: returns None, UI will use mock inference
    return None, "demo"


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    img = img.convert("RGB").resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, framework: str, arr: np.ndarray):
    """
    Run the model. In demo mode, derive a reproducible mock result from
    image statistics so different images give different outcomes.
    """
    if framework == "demo" or model is None:
        # Use image statistics to create a varied (but deterministic) mock result
        seed = int(arr.mean() * 1000 + arr.std() * 500) % 100
        rng = np.random.default_rng(seed)
        raw = rng.dirichlet([1.5, 1.5])  # [no_tumor_prob, tumor_prob]
        return raw[1], raw[0]  # tumor_conf, no_tumor_conf

    if framework == "tensorflow":
        import tensorflow as tf  # noqa: F401
        inp = np.expand_dims(arr, 0)
        preds = model.predict(inp, verbose=0)[0]
        # Assume model output: [no_tumor, tumor]
        return float(preds[1]), float(preds[0])

    if framework == "pytorch":
        import torch
        inp = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            logits = model(inp)[0]
            probs = torch.softmax(logits, dim=0).numpy()
        return float(probs[1]), float(probs[0])

    return 0.5, 0.5


# ── Image stats ───────────────────────────────────────────────────────────────
def image_stats(img: Image.Image):
    gray = np.array(img.convert("L"), dtype=np.float32)
    return {
        "Size": f"{img.width} × {img.height}",
        "Mode": img.mode,
        "Brightness": f"{gray.mean():.1f}",
        "Contrast": f"{gray.std():.1f}",
    }


# ── Confidence bar HTML ───────────────────────────────────────────────────────
def conf_bar(label: str, value: float, color: str):
    pct = value * 100
    return f"""
<div>
  <div style="display:flex;justify-content:space-between;align-items:baseline;">
    <span class="conf-label">{label}</span>
    <span class="conf-value">{pct:.1f}%</span>
  </div>
  <div class="conf-bar-bg">
    <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
  </div>
</div>"""


# ── Main UI ───────────────────────────────────────────────────────────────────
def main():
    # Hero
    st.markdown('<div class="hero-title">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">MRI scan analysis · Deep learning classifier</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading model…"):
        model, framework = load_model()

    if framework == "demo":
        st.info(
            "**Demo mode** – No trained model found. Upload any MRI image to see the interface in action. "
            "To use a real model, add your `brain_tumor_model.h5` (TensorFlow/Keras) or "
            "`brain_tumor_model.pt` (PyTorch) file and update the `load_model()` function.",
            icon="ℹ️",
        )

    # Upload
    uploaded = st.file_uploader(
        "Upload an MRI scan (JPG / PNG / TIFF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        label_visibility="visible",
    )

    if uploaded is None:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#475569;">
            <div style="font-size:3rem;margin-bottom:0.5rem;">📂</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.9rem;">
                Drop an MRI image above to begin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
        _show_disclaimer()
        return

    # Display image + stats
    img = Image.open(io.BytesIO(uploaded.read()))
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(img, caption="Uploaded MRI scan", use_container_width=True)

    with col2:
        stats = image_stats(img)
        st.markdown('<div class="metric-row">' + "".join(
            f'<div class="metric-box"><div class="m-label">{k}</div>'
            f'<div class="m-value">{v}</div></div>'
            for k, v in stats.items()
        ) + "</div>", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Run analysis button
    if st.button("🔍  Analyse Scan"):
        with st.spinner("Running inference…"):
            arr = preprocess_image(img)
            tumor_conf, no_tumor_conf = run_inference(model, framework, arr)

        # Verdict logic
        THRESHOLD = 0.60
        if tumor_conf >= THRESHOLD:
            verdict, card_cls, v_cls, emoji = "TUMOR DETECTED", "tumor", "tumor", "🔴"
        elif no_tumor_conf >= THRESHOLD:
            verdict, card_cls, v_cls, emoji = "NO TUMOR DETECTED", "no-tumor", "no-tumor", "🟢"
        else:
            verdict, card_cls, v_cls, emoji = "INCONCLUSIVE", "uncertain", "uncertain", "🟡"

        # Result card
        bars = (
            conf_bar("Tumor probability", tumor_conf, "#ef4444") +
            conf_bar("No-tumor probability", no_tumor_conf, "#22c55e")
        )

        st.markdown(f"""
        <div class="result-card {card_cls}">
            <div class="verdict {v_cls}">{emoji} {verdict}</div>
            <div style="color:var(--muted);font-size:0.85rem;margin-bottom:1rem;">
                Model confidence · {framework.upper()} backend
            </div>
            {bars}
        </div>
        """, unsafe_allow_html=True)

        # Extra guidance
        if card_cls == "tumor":
            st.error(
                "⚠️ The model flagged a potential anomaly. This result must be reviewed "
                "by a qualified radiologist before any clinical decision is made."
            )
        elif card_cls == "no-tumor":
            st.success("✅ No significant anomaly detected by the model.")
        else:
            st.warning(
                "🔁 The model was not confident enough to make a clear prediction. "
                "Consider re-uploading a higher-quality scan or consulting a specialist."
            )

    _show_disclaimer()


def _show_disclaimer():
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for <em>research and educational purposes only</em>.
        It is <strong>not</strong> a certified medical device and must not be used for clinical diagnosis or
        treatment decisions. Always consult a licensed medical professional for any health concerns.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
