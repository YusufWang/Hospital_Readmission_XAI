# English note:
# Streamlit XAI Dashboard (stable paths + full-page background + robust image loading)
# Key fixes:
# 1) Use absolute paths relative to this file (no working-dir issues).
# 2) Use st.radio(horizontal=True) for navigation to enable full-page background.
# 3) Robust image loading with PIL + file-size debug (avoid "blank image" mystery).
# 4) Fuzzy-match image filenames if the exact names do not exist.

import os
import glob
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

APP_TITLE = "Hospital Readmission XAI Dashboard"
SUBTITLE = "System Prototype: Prediction + Performance + XAI (SHAP + Counterfactuals)"

TARGET_COL = "readmitted"  # yes/no in your CSV

# -----------------------------
# Paths (absolute, stable)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "dashboard_assets")

DATA_PATH = os.path.join(BASE_DIR, "hospital_readmissions.csv")

PIPE_PATH = os.path.join(ASSET_DIR, "model_pipeline.joblib")
MODEL_ONLY_PATH = os.path.join(ASSET_DIR, "model_only.joblib")
PREPROC_PATH = os.path.join(ASSET_DIR, "preprocessor.joblib")
TRAIN_COLS_PATH = os.path.join(ASSET_DIR, "train_columns.joblib")

# Preferred artifact names (but we will fuzzy-match if missing)
SHAP_GLOBAL_PREFERRED = os.path.join(ASSET_DIR, "shap_global_summary.png")
SHAP_LOCAL_PREFERRED  = os.path.join(ASSET_DIR, "shap_local_waterfall.png")
DICE_CSV_PREFERRED    = os.path.join(ASSET_DIR, "dice_counterfactuals.csv")


# -----------------------------
# CSS (full-page background)
# -----------------------------
def apply_page_bg(page: str):
    if page == "Overview":
        bg = "rgba(46, 204, 113, 0.06)"
        bd = "rgba(39, 174, 96, 0.18)"
    elif page == "Model Performance":
        bg = "rgba(52, 152, 219, 0.06)"
        bd = "rgba(41, 128, 185, 0.18)"
    else:
        bg = "rgba(230, 126, 34, 0.06)"
        bd = "rgba(211, 84, 0, 0.18)"

    st.markdown(
        f"""
        <style>
        /* Make the top header transparent to avoid covering text */
        header[data-testid="stHeader"] {{
            background: rgba(255,255,255,0.0);
        }}

        /* Give enough top padding so title/navigation won't be overlapped */
        .block-container {{
            padding-top: 3.8rem;
            padding-bottom: 2.0rem;
        }}

        .stApp {{
            background: {bg};
        }}

        .panel {{
            background: rgba(255,255,255,0.85);
            border: 1px solid {bd};
            border-radius: 16px;
            padding: 16px 18px;
        }}

        /* Slightly lift radio nav and make it readable */
        div[role="radiogroup"] {{
            background: rgba(255,255,255,0.75);
            border: 1px solid rgba(0,0,0,0.08);
            padding: 8px 12px;
            border-radius: 999px;
            display: inline-flex;
            gap: 8px;
        }}

        .stButton>button {{
            border-radius: 12px;
            font-weight: 800;
            padding: 0.6rem 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# Helpers
# -----------------------------
def ensure_fe_columns(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "total_visits" not in X.columns:
        X["total_visits"] = (
            X.get("n_outpatient", 0).astype(float)
            + X.get("n_inpatient", 0).astype(float)
            + X.get("n_emergency", 0).astype(float)
        )
    if "any_inpatient" not in X.columns:
        X["any_inpatient"] = (X.get("n_inpatient", 0).astype(float) > 0).astype(int)
    return X


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            f"Tip: copy your Kaggle-downloaded CSV to this location."
        )
    return pd.read_csv(DATA_PATH)


def load_artifacts():
    pipe = joblib.load(PIPE_PATH) if os.path.exists(PIPE_PATH) else None
    model = joblib.load(MODEL_ONLY_PATH) if os.path.exists(MODEL_ONLY_PATH) else None
    preproc = joblib.load(PREPROC_PATH) if os.path.exists(PREPROC_PATH) else None
    train_cols = joblib.load(TRAIN_COLS_PATH) if os.path.exists(TRAIN_COLS_PATH) else None
    return pipe, model, preproc, train_cols


def predict_proba_with_fallback(X_in_raw: pd.DataFrame, pipe, model, preproc, train_cols) -> float:
    X_in = ensure_fe_columns(X_in_raw)

    if pipe is not None:
        return float(pipe.predict_proba(X_in)[:, 1][0])

    if (model is not None) and (preproc is not None):
        X_trans = preproc.transform(X_in)
        return float(model.predict_proba(X_trans)[:, 1][0])

    if (model is not None) and (train_cols is not None):
        X_enc = pd.get_dummies(X_in)
        X_enc = X_enc.reindex(columns=train_cols, fill_value=0)
        return float(model.predict_proba(X_enc)[:, 1][0])

    raise RuntimeError("No deployable model found. Export model_pipeline.joblib is recommended.")


def int_input(label: str, value, min_value: int = 0, max_value: int | None = None, step: int = 1):
    try:
        v = int(round(float(value)))
    except Exception:
        v = int(min_value)
    kwargs = dict(value=v, step=step, format="%d", min_value=int(min_value))
    if max_value is not None:
        kwargs["max_value"] = int(max_value)
        if v > max_value:
            kwargs["value"] = int(max_value)
    return st.number_input(label, **kwargs)


def cat_select(df: pd.DataFrame, col: str, current_value):
    if col not in df.columns:
        return None
    options = sorted(df[col].dropna().astype(str).unique().tolist())
    cur = str(current_value) if current_value is not None else (options[0] if options else "")
    if options and cur not in options:
        cur = options[0]
    return st.selectbox(col, options=options if options else [cur], index=(options.index(cur) if cur in options else 0))


def find_best_file(preferred_path: str, patterns: list[str]) -> str | None:
    """Return an existing file path. Try preferred_path first, then glob patterns."""
    if os.path.exists(preferred_path) and os.path.getsize(preferred_path) > 0:
        return preferred_path
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(ASSET_DIR, pat)))
        # pick first non-empty
        for h in hits:
            if os.path.exists(h) and os.path.getsize(h) > 0:
                return h
    return None


def show_image_block(title: str, preferred: str, patterns: list[str]):
    st.markdown(f"**{title}**")
    path = find_best_file(preferred, patterns)

    if path is None:
        st.warning(
            "Image not found (or file is empty). "
            "Please export the PNG into dashboard_assets/."
        )
        # quick listing for debug
        with st.expander("Debug: list dashboard_assets", expanded=False):
            files = sorted(glob.glob(os.path.join(ASSET_DIR, "*")))
            if not files:
                st.write("(dashboard_assets is empty)")
            else:
                st.write(pd.DataFrame(
                    [{"file": os.path.basename(f), "size_KB": round(os.path.getsize(f)/1024, 2)} for f in files]
                ))
        return

    size_kb = round(os.path.getsize(path)/1024, 2)
    st.caption(f"Loaded: {os.path.basename(path)}  |  size: {size_kb} KB")

    try:
        img = Image.open(path)
        st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to open image via PIL: {e}")
        st.stop()


def show_artifacts_status():
    items = [
        ("DATA", DATA_PATH),
        ("model_pipeline.joblib", PIPE_PATH),
        ("model_only.joblib", MODEL_ONLY_PATH),
        ("preprocessor.joblib", PREPROC_PATH),
        ("train_columns.joblib", TRAIN_COLS_PATH),
        ("shap_global_summary.png (preferred)", SHAP_GLOBAL_PREFERRED),
        ("shap_local_waterfall.png (preferred)", SHAP_LOCAL_PREFERRED),
        ("dice_counterfactuals.csv (preferred)", DICE_CSV_PREFERRED),
    ]
    rows = []
    for name, p in items:
        exists = os.path.exists(p)
        size = round(os.path.getsize(p)/1024, 2) if exists else None
        rows.append({"Artifact": name, "Exists": "Yes" if exists else "No", "Size(KB)": size})
    st.table(pd.DataFrame(rows))


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(SUBTITLE)

page = st.radio(
    "Navigation",
    ["Overview", "Model Performance", "Prediction + XAI"],
    horizontal=True,
    label_visibility="collapsed"
)

apply_page_bg(page)

try:
    df = load_dataset()
except Exception as e:
    st.error(str(e))
    st.stop()

pipe, model, preproc, train_cols = load_artifacts()

with st.expander("Artifacts status (debug)", expanded=False):
    show_artifacts_status()


# -----------------------------
# Pages
# -----------------------------
if page == "Overview":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Overview")
    st.markdown(
        "- Use case: Readmission risk screening and decision support.\n"
        "- Key idea: choose an operating point (threshold) based on clinical goal (e.g., high recall).\n"
        "- XAI outputs: Global SHAP, Local SHAP, and counterfactual examples (DiCE).\n"
    )
    st.markdown("**Dataset preview**")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Model Performance":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Model Performance")
    st.info(
        "This page is intentionally lightweight. "
        "If you want automatic metric rendering here, export metrics.json from your notebook."
    )
    st.markdown(
        "- Recommended reporting: show two operating points (thr=0.50 baseline vs thr=0.35 high-recall).\n"
        "- Emphasize the trade-off: recall(yes) ↑ but accuracy ↓ and FP ↑ when threshold decreases.\n"
    )
    st.markdown("</div>", unsafe_allow_html=True)


else:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Prediction + XAI")

    st.markdown("### A) Select one patient row as a template (auto-loaded from Kaggle)")
    max_idx = max(len(df) - 1, 0)
    idx = int_input("Row index", value=50, min_value=0, max_value=max_idx, step=1)

    row = df.iloc[int(idx)].copy()
    row_features = row.drop(labels=[TARGET_COL]) if TARGET_COL in row.index else row.copy()
    st.dataframe(row_features.to_frame().T, use_container_width=True)

    st.markdown("### B) Edit key inputs (integer-only form)")
    row_features_df = ensure_fe_columns(pd.DataFrame([row_features]))
    row_features = row_features_df.iloc[0]

    c1, c2 = st.columns(2)

    with c1:
        time_in_hospital = int_input("time_in_hospital", row_features.get("time_in_hospital", 1), min_value=0)
        n_inpatient = int_input("n_inpatient", row_features.get("n_inpatient", 0), min_value=0)
        n_emergency = int_input("n_emergency", row_features.get("n_emergency", 0), min_value=0)
        n_outpatient = int_input("n_outpatient", row_features.get("n_outpatient", 0), min_value=0)
        n_medications = int_input("n_medications", row_features.get("n_medications", 0), min_value=0)

    with c2:
        age = cat_select(df, "age", row_features.get("age", None))
        medical_specialty = cat_select(df, "medical_specialty", row_features.get("medical_specialty", None))
        diag_1 = cat_select(df, "diag_1", row_features.get("diag_1", None))
        diag_2 = cat_select(df, "diag_2", row_features.get("diag_2", None))
        diag_3 = cat_select(df, "diag_3", row_features.get("diag_3", None))

    X_in = pd.DataFrame([row_features.to_dict()])
    X_in["time_in_hospital"] = int(time_in_hospital)
    X_in["n_inpatient"] = int(n_inpatient)
    X_in["n_emergency"] = int(n_emergency)
    X_in["n_outpatient"] = int(n_outpatient)
    X_in["n_medications"] = int(n_medications)

    if age is not None: X_in["age"] = age
    if medical_specialty is not None: X_in["medical_specialty"] = medical_specialty
    if diag_1 is not None: X_in["diag_1"] = diag_1
    if diag_2 is not None: X_in["diag_2"] = diag_2
    if diag_3 is not None: X_in["diag_3"] = diag_3

    X_in = ensure_fe_columns(X_in)

    st.markdown("### C) Prediction")
    thr = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01)

    if st.button("Predict"):
        try:
            proba = predict_proba_with_fallback(X_in, pipe, model, preproc, train_cols)
            pred = int(proba >= thr)

            st.success(f"Predicted probability (readmitted = 1): **{proba:.4f}**")
            st.info(f"Decision @ threshold {thr:.2f}: **{'YES (readmitted)' if pred==1 else 'NO'}**")
        except Exception as e:
            st.error(str(e))
            st.stop()

    st.markdown("### D) XAI Visualizations")
    x1, x2 = st.columns(2)

    with x1:
        show_image_block(
            "Global SHAP summary",
            SHAP_GLOBAL_PREFERRED,
            patterns=["shap_global*.png", "*global*shap*.png", "*.png"]
        )
    with x2:
        show_image_block(
            "Local SHAP (waterfall)",
            SHAP_LOCAL_PREFERRED,
            patterns=["shap_local*.png", "*local*shap*.png", "*waterfall*.png", "*.png"]
        )

    st.markdown("### E) Counterfactual Examples (DiCE)")
    dice_csv = None
    if os.path.exists(DICE_CSV_PREFERRED) and os.path.getsize(DICE_CSV_PREFERRED) > 0:
        dice_csv = DICE_CSV_PREFERRED
    else:
        # fuzzy match
        hits = sorted(glob.glob(os.path.join(ASSET_DIR, "*dice*.csv"))) + sorted(glob.glob(os.path.join(ASSET_DIR, "*counterfactual*.csv")))
        for h in hits:
            if os.path.exists(h) and os.path.getsize(h) > 0:
                dice_csv = h
                break

    if dice_csv is None:
        st.warning("Counterfactual CSV not found (or empty). Export dice_counterfactuals.csv into dashboard_assets/.")
    else:
        st.caption(f"Loaded: {os.path.basename(dice_csv)}  |  size: {round(os.path.getsize(dice_csv)/1024, 2)} KB")
        try:
            cf_df = pd.read_csv(dice_csv)
            st.dataframe(cf_df, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
