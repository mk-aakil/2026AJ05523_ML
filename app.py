"""Shopper Insights ML — inference-only evaluation dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DEFAULT_MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}
TARGET = "Revenue"
NUMERIC = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
           "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
CATEGORICAL = ["OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Month", "Weekend"]
FEATURES = NUMERIC + CATEGORICAL

st.set_page_config(page_title="Shopper Insights ML", page_icon="🛍️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp {background:#f7f9fe;color:#18243c;}
.block-container {padding-top:1.7rem;max-width:1450px;}
.hero {padding:1.6rem 1.9rem;border-radius:20px;background:linear-gradient(120deg,#182f70,#5d48ad);color:white;margin-bottom:1.3rem;}
.hero h1 {color:white;font-size:2.1rem;margin:0 0 .4rem 0;}
.hero p {color:#e0e7ff;margin:0;}
.section {font-size:1.2rem;font-weight:750;color:#1f3263;margin:1.2rem 0 .6rem;}
[data-testid="stMetric"] {background:white;padding:1rem;border:1px solid #e5eaf5;border-radius:14px;box-shadow:0 3px 14px #192f6410;}
[data-testid="stSidebar"] {background:#f0f3fd;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading pretrained model...")
def load_pipeline(path_str: str):
    # Only load artifacts that you created/trust. joblib/pickle can execute code on load.
    return joblib.load(path_str)

@st.cache_data(show_spinner=False)
def load_metadata():
    p = MODEL_DIR / "metadata.json"
    if p.exists():
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def as_binary_target(series: pd.Series) -> pd.Series:
    """Accept 0/1, True/False and their common CSV string representations."""
    if series.isna().any():
        raise ValueError("The Revenue column contains missing labels.")
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
    values = series.astype(str).str.strip().str.lower().map(mapping)
    if values.isna().any():
        examples = series[values.isna()].astype(str).unique()[:4].tolist()
        raise ValueError(f"Revenue must contain True/False or 1/0. Unexpected labels: {examples}")
    return values.astype(int)


def validate_upload(frame: pd.DataFrame, feature_names: list[str]):
    missing = [c for c in feature_names + [TARGET] if c not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("The uploaded CSV contains no rows.")
    if frame.columns.duplicated().any():
        raise ValueError("The uploaded CSV contains duplicate column names.")
    X = frame[feature_names].copy()
    y = as_binary_target(frame[TARGET])
    # Preserve the types expected by the fitted ColumnTransformer.
    for col in NUMERIC:
        if col in X:
            X[col] = pd.to_numeric(X[col], errors="raise")
    for col in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        if col in X:
            X[col] = pd.to_numeric(X[col], errors="raise")
    if "Weekend" in X:
        normalized = X["Weekend"].astype(str).str.strip().str.lower()
        invalid = ~normalized.isin(["true", "false", "1", "0"])
        if invalid.any():
            raise ValueError("Weekend must contain True/False or 1/0.")
        X["Weekend"] = normalized.isin(["true", "1"])
    return X, y


def evaluate(model, X, y):
    pred = model.predict(X)
    score = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        idx = classes.index(1) if 1 in classes else classes.index(True)
        score = model.predict_proba(X)[:, idx]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "AUC": roc_auc_score(y, score) if score is not None and y.nunique() == 2 else np.nan,
    }
    return metrics, pred, score

metadata = load_metadata()
model_files = metadata.get("model_files", DEFAULT_MODELS)
feature_names = metadata.get("feature_columns", FEATURES)
available = {name: MODEL_DIR / filename for name, filename in model_files.items() if (MODEL_DIR / filename).is_file()}

st.markdown('<div class="hero"><h1>🛍️ Shopper Insights ML</h1><p>Business Analytics Pro · Purchase intention intelligence · Five pretrained classifiers</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Evaluation settings")
    st.caption("Inference only · No training on uploaded data")
    if not available:
        st.error("No trained models found in model/. Run notebook 02 first and copy its artifacts into this project.")
        st.stop()
    selected = st.selectbox("Select classifier", list(available))
    uploaded = st.file_uploader("Upload labeled test CSV", type=["csv"], help="Include all predictor columns plus Revenue.")
    st.markdown("---")
    st.caption(f"Ready models: {len(available)} / {len(model_files)}")
    st.caption("Use the untouched test_data.csv exported by notebook 02 for comparable scores.")

if uploaded is None:
    st.info("Upload **test_data.csv** to evaluate the pretrained classifiers. The CSV must contain the Revenue label and all predictor columns.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Models available", len(available))
    col2.metric("Required predictors", len(feature_names))
    col3.metric("Target", "Revenue")
    st.markdown('<div class="section">Expected data columns</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Column": feature_names + [TARGET], "Role": ["Predictor"] * len(feature_names) + ["Target"]}), hide_index=True, use_container_width=True)
    st.stop()

try:
    uploaded.seek(0)
    frame = pd.read_csv(uploaded)
    X, y = validate_upload(frame, feature_names)
except Exception as exc:
    st.error(f"CSV validation failed: {exc}")
    st.stop()

st.markdown('<div class="section">Dataset overview</div>', unsafe_allow_html=True)
a, b, c, d = st.columns(4)
a.metric("Sessions", f"{len(frame):,}")
b.metric("Predictors", len(feature_names))
c.metric("Purchase rate", f"{y.mean():.1%}")
d.metric("Non-purchase rate", f"{1-y.mean():.1%}")
with st.expander("Preview uploaded data"):
    st.dataframe(frame.head(30), use_container_width=True)

try:
    pipeline = load_pipeline(str(available[selected]))
    metrics, pred, scores = evaluate(pipeline, X, y)
except Exception as exc:
    st.error(f"Model evaluation failed: {exc}")
    st.stop()

st.markdown(f'<div class="section">Performance · {selected}</div>', unsafe_allow_html=True)
metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
cols = st.columns(3)
for i, name in enumerate(metric_names):
    val = metrics[name]
    cols[i % 3].metric(name, "N/A" if pd.isna(val) else f"{val:.4f}")

left, right = st.columns(2)
with left:
    st.markdown('<div class="section">Confusion matrix</div>', unsafe_allow_html=True)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig = px.imshow(cm, text_auto=True, x=["No Purchase", "Purchase"], y=["No Purchase", "Purchase"], color_continuous_scale="Blues", labels={"x": "Predicted", "y": "Actual", "color": "Count"})
    fig.update_layout(height=370, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.markdown('<div class="section">ROC curve</div>', unsafe_allow_html=True)
    if scores is not None and y.nunique() == 2:
        fpr, tpr, _ = roc_curve(y, scores)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{selected} (AUC {metrics['AUC']:.3f})", line=dict(color="#5d48ad", width=3)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random baseline", line=dict(color="#9ca3af", dash="dash")))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate", height=370, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("ROC-AUC needs both target classes and a model score.")

st.markdown('<div class="section">Classification report</div>', unsafe_allow_html=True)
report = classification_report(y, pred, labels=[0,1], target_names=["No Purchase", "Purchase"], output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).T
st.dataframe(report_df.style.format(precision=4), use_container_width=True)

st.markdown('<div class="section">Compare all available models on this CSV</div>', unsafe_allow_html=True)
if st.button("Evaluate all five pretrained models", type="primary"):
    rows = []
    with st.spinner("Running inference across saved models..."):
        for name, path in available.items():
            try:
                model = load_pipeline(str(path))
                vals, _, _ = evaluate(model, X, y)
                rows.append({"Model": name, **vals})
            except Exception as exc:
                st.warning(f"{name}: {exc}")
    if rows:
        table = pd.DataFrame(rows).set_index("Model")
        st.dataframe(table.style.format(precision=4).highlight_max(axis=0), use_container_width=True)
        plot_df = table[["Accuracy", "AUC", "Precision", "Recall", "F1"]].reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(plot_df, x="Model", y="Score", color="Metric", barmode="group", color_discrete_sequence=["#2563eb", "#7c3aed", "#14b8a6", "#f59e0b", "#64748b"])
        fig.update_layout(yaxis_range=[0,1], height=430)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download evaluation results", data=table.to_csv().encode("utf-8"), file_name="uploaded_test_model_comparison.csv", mime="text/csv")

with st.expander("How to interpret the metrics"):
    st.markdown("**Accuracy** measures overall correctness. **Precision** measures how many predicted purchases were real purchases. **Recall** measures how many actual purchases were found. **F1** balances precision and recall. **AUC** measures ranking quality. **MCC** uses all four confusion-matrix cells and is useful for imbalanced classes. AUC is unavailable when the uploaded labels contain only one class.")
st.caption("Educational ML dashboard · Only load trusted local joblib files · Uploaded data is evaluated, never used to fit models")
