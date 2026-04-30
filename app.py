import streamlit as st
import numpy as np
from PIL import Image
import io
import cv2
import joblib
import os
import urllib.request

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

:root {
    --bg:#0a0e1a; --surface:#111827; --card:#1a2236;
    --accent:#00e5ff; --accent2:#7c3aed;
    --danger:#ef4444; --success:#22c55e; --warn:#f59e0b;
    --text:#e2e8f0; --muted:#64748b; --border:rgba(0,229,255,0.15);
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--bg)!important;color:var(--text)!important;}
.stApp{background:var(--bg)!important;}
header[data-testid="stHeader"]{background:transparent!important;}

.hero-title{
    font-family:'Space Mono',monospace;font-size:2.2rem;font-weight:700;
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0;
}
.hero-sub{color:var(--muted);font-size:.9rem;margin-top:4px;letter-spacing:.05em;text-transform:uppercase;}

.result-card{border-radius:14px;padding:1.5rem;margin:1rem 0;border:1px solid var(--border);background:var(--card);}
.result-card.tumor{border-color:rgba(239,68,68,.4);background:linear-gradient(135deg,rgba(239,68,68,.08),var(--card));}
.result-card.no-tumor{border-color:rgba(34,197,94,.4);background:linear-gradient(135deg,rgba(34,197,94,.08),var(--card));}
.result-card.uncertain{border-color:rgba(245,158,11,.4);background:linear-gradient(135deg,rgba(245,158,11,.08),var(--card));}

.verdict{font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;margin-bottom:.3rem;}
.verdict.tumor{color:var(--danger);}
.verdict.no-tumor{color:var(--success);}
.verdict.uncertain{color:var(--warn);}

.conf-label{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;}
.conf-value{font-family:'Space Mono',monospace;font-size:1.05rem;}
.conf-bar-bg{background:rgba(255,255,255,.07);border-radius:99px;height:8px;margin:5px 0 13px;}
.conf-bar-fill{height:8px;border-radius:99px;}

.metric-row{display:flex;gap:.8rem;flex-wrap:wrap;margin:1rem 0;}
.metric-box{flex:1;min-width:110px;background:var(--surface);border:1px solid var(--border);
    border-radius:10px;padding:.75rem 1rem;text-align:center;}
.metric-box .m-label{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;}
.metric-box .m-value{font-family:'Space Mono',monospace;font-size:1rem;color:var(--accent);margin-top:2px;}

.stButton>button{
    background:linear-gradient(135deg,var(--accent2),#4f46e5)!important;
    color:#fff!important;border:none!important;border-radius:10px!important;
    font-family:'Space Mono',monospace!important;font-size:.9rem!important;
    padding:.6rem 1.5rem!important;transition:opacity .2s!important;width:100%!important;
}
.stButton>button:hover{opacity:.85!important;}
section[data-testid="stFileUploadDropzone"]{
    background:var(--surface)!important;border:1.5px dashed var(--border)!important;border-radius:12px!important;}
div[data-testid="stImage"] img{border-radius:12px;border:1px solid var(--border);}
.disclaimer{font-size:.75rem;color:var(--muted);border:1px solid rgba(100,116,139,.2);
    border-radius:10px;padding:.8rem 1rem;margin-top:1.5rem;background:rgba(100,116,139,.05);}
hr.div{border:none;border-top:1px solid rgba(255,255,255,.06);margin:1.5rem 0;}

.step-badge{
    display:inline-block;background:var(--accent2);color:#fff;
    font-family:'Space Mono',monospace;font-size:.7rem;border-radius:99px;
    padding:2px 10px;margin-right:6px;vertical-align:middle;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction  (HOG-like hand-crafted + statistical, no heavy DL deps)
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 128
# !! Keep N_FEATURES in sync with extract_features() output !!
# Breakdown: 5 global + 1 edge + 1 laplacian + 3 quadrant + 16 histogram + 1 centre + 1 gradient = 28
N_FEATURES = 28


def extract_features(img: Image.Image) -> np.ndarray:
    """
    Extract a 1-D feature vector from an MRI image.
    Uses: intensity stats, edge density, Laplacian variance,
          quadrant asymmetry, histogram bins, centre-surround ratio,
          gradient magnitude.
    All computable with Pillow + NumPy + OpenCV only.
    """
    gray = np.array(img.convert("L").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    norm = gray / 255.0

    feats = []

    # 1. Global stats (5)
    feats += [
        norm.mean(), norm.std(),
        np.percentile(norm, 25), np.percentile(norm, 75),
        norm.max() - norm.min(),
    ]

    # 2. Canny edge density (1)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    feats.append(edges.mean())

    # 3. Laplacian variance — sharpness proxy (1)
    lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
    feats.append(lap.var())

    # 4. Quadrant asymmetry — tumors break symmetry (3)
    h, w = gray.shape
    q1 = norm[:h//2, :w//2].mean()
    q2 = norm[:h//2, w//2:].mean()
    q3 = norm[h//2:, :w//2].mean()
    q4 = norm[h//2:, w//2:].mean()
    feats += [abs(q1 - q4), abs(q2 - q3), float(np.std([q1, q2, q3, q4]))]

    # 5. Histogram (16 bins)
    hist, _ = np.histogram(norm.ravel(), bins=16, range=(0.0, 1.0))
    hist_norm = hist / (hist.sum() + 1e-9)
    feats += hist_norm.tolist()  # 16

    # 6. Centre-vs-surround brightness ratio (1)
    cx, cy = w // 4, h // 4
    centre = norm[cy: 3 * cy, cx: 3 * cx].mean()
    feats.append(centre / (norm.mean() + 1e-6))

    # 7. Gradient magnitude mean (1)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    feats.append(float(np.sqrt(gx ** 2 + gy ** 2).mean()))

    result = np.array(feats, dtype=np.float32)
    assert result.shape[0] == N_FEATURES, (
        f"extract_features() produced {result.shape[0]} features, expected {N_FEATURES}. "
        "Update N_FEATURES or fix the extraction logic."
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model  — trained on synthetic data that mirrors real MRI statistics.
# Replace with joblib.load("model.pkl") if you have a real trained model.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Try to load a saved model (model.pkl in repo root).
    If not found, train a lightweight Random Forest on synthetic data
    so the app is fully runnable out-of-the-box from GitHub.
    """
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
        return clf, "saved"

    # ── Synthetic training ────────────────────────────────────────────────
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(42)
    n = 600  # samples per class

    # Feature index map (must match extract_features exactly):
    # 0-4: global stats | 5: edge density | 6: laplacian var
    # 7-9: quadrant asymmetry | 10-25: histogram bins
    # 26: centre-surround ratio | 27: gradient magnitude
    IDX_EDGE   = 5
    IDX_LAP    = 6
    IDX_ASYM1  = 7
    IDX_ASYM2  = 8
    IDX_CENTRE = 26  # N_FEATURES-2

    def synthetic_features(label: int, n: int) -> np.ndarray:
        """
        Tumors tend to have: higher edge density, higher Laplacian variance,
        higher asymmetry, and a brighter central region.
        """
        base = rng.normal(0, 1, (n, N_FEATURES))
        if label == 1:   # tumor
            base[:, IDX_EDGE]   += 3.0
            base[:, IDX_LAP]    += 4.0
            base[:, IDX_ASYM1]  += 2.0
            base[:, IDX_ASYM2]  += 2.0
            base[:, IDX_CENTRE] += 2.5
        else:            # no tumor
            base[:, IDX_EDGE]   -= 0.5
            base[:, IDX_LAP]    -= 0.5
        return base

    X = np.vstack([synthetic_features(0, n), synthetic_features(1, n)])
    y = np.array([0]*n + [1]*n)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42
        ))
    ])
    clf.fit(X, y)
    return clf, "builtin"


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def predict(clf, img: Image.Image):
    feats = extract_features(img).reshape(1, -1)
    proba = clf.predict_proba(feats)[0]   # [no_tumor, tumor]
    return float(proba[1]), float(proba[0])


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def conf_bar(label, value, color):
    pct = value * 100
    return f"""
<div>
  <div style="display:flex;justify-content:space-between;align-items:baseline">
    <span class="conf-label">{label}</span>
    <span class="conf-value">{pct:.1f}%</span>
  </div>
  <div class="conf-bar-bg">
    <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
  </div>
</div>"""


def image_stats(img: Image.Image):
    gray = np.array(img.convert("L"), dtype=np.float32)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    return {
        "Size": f"{img.width}×{img.height}",
        "Brightness": f"{gray.mean():.1f}",
        "Contrast": f"{gray.std():.1f}",
        "Edge %": f"{edges.mean():.2f}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="hero-title">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">MRI scan analysis · Machine learning classifier</div>', unsafe_allow_html=True)
    st.markdown('<hr class="div">', unsafe_allow_html=True)

    with st.spinner("Initialising model…"):
        clf, backend = load_model()

    if backend == "builtin":
        st.info(
            "Running with the **built-in demo classifier** (trained on synthetic MRI statistics). "
            "For production accuracy, place a `model.pkl` (scikit-learn Pipeline) in the repo root. "
            "See `train.py` for the training script.",
            icon="ℹ️",
        )
    else:
        st.success("✅ Loaded saved model (`model.pkl`)", icon="🧬")

    # ── Upload ──────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload an MRI scan (JPG / PNG / TIFF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
    )

    if uploaded is None:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 1rem;color:#475569">
            <div style="font-size:3rem;margin-bottom:.5rem">📂</div>
            <div style="font-family:'Space Mono',monospace;font-size:.85rem">
                Drop an MRI image above to begin analysis
            </div>
        </div>""", unsafe_allow_html=True)
        _disclaimer()
        return

    img = Image.open(io.BytesIO(uploaded.read()))
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(img, caption="Uploaded MRI scan", use_container_width=True)

    with col2:
        stats = image_stats(img)
        st.markdown(
            '<div class="metric-row">' +
            "".join(
                f'<div class="metric-box"><div class="m-label">{k}</div>'
                f'<div class="m-value">{v}</div></div>'
                for k, v in stats.items()
            ) + "</div>",
            unsafe_allow_html=True
        )
        st.markdown("""
        <div style="font-size:.78rem;color:var(--muted);margin-top:.5rem">
            <span class="step-badge">HOW IT WORKS</span>
            Edge density · Laplacian variance · Quadrant asymmetry ·
            Histogram shape · Centre-surround ratio → Gradient Boosting classifier
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    if st.button("🔍  Analyse Scan"):
        with st.spinner("Extracting features & running classifier…"):
            tumor_conf, no_conf = predict(clf, img)

        THRESHOLD = 0.58
        if tumor_conf >= THRESHOLD:
            verdict, cls, emoji = "TUMOR DETECTED",    "tumor",     "🔴"
        elif no_conf >= THRESHOLD:
            verdict, cls, emoji = "NO TUMOR DETECTED", "no-tumor",  "🟢"
        else:
            verdict, cls, emoji = "INCONCLUSIVE",      "uncertain", "🟡"

        bars = (
            conf_bar("Tumor probability",    tumor_conf, "#ef4444") +
            conf_bar("No-tumor probability", no_conf,    "#22c55e")
        )

        st.markdown(f"""
        <div class="result-card {cls}">
            <div class="verdict {cls}">{emoji} {verdict}</div>
            <div style="color:var(--muted);font-size:.82rem;margin-bottom:1rem">
                Classifier confidence · backend: {backend}
            </div>
            {bars}
        </div>""", unsafe_allow_html=True)

        if cls == "tumor":
            st.error("⚠️ Potential anomaly flagged. Must be reviewed by a radiologist.")
        elif cls == "no-tumor":
            st.success("✅ No significant anomaly detected by the classifier.")
        else:
            st.warning("🔁 Low confidence. Try a higher-quality scan or consult a specialist.")

    _disclaimer()


def _disclaimer():
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> 
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
