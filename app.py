"""
HEARTGUARD - Heart Disease Prediction Using Machine Learning
Streamlit App
Author: Kartikkumar Sangada (22012011044)
B.Tech. Semester VIII - Computer Engineering
Ganpat University | U.V. Patel College of Engineering
"""

import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="HeartGuard – Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
STATIC_IMG_DIR = BASE_DIR / "static" / "img"
DATASET_PATH = BASE_DIR / "heart.csv"

# ─────────────────────────────────────────
# LOAD MODEL ASSETS (cached)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading HeartGuard model…")
def load_model_assets():
    try:
        model = joblib.load(MODELS_DIR / "heartguard_model.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
        importances = joblib.load(MODELS_DIR / "importances.pkl")
        importance_labels = joblib.load(MODELS_DIR / "importance_labels.pkl")
        with open(MODELS_DIR / "model_metrics.json", encoding="utf-8") as fp:
            metrics = json.load(fp)
        return model, scaler, feature_names, importances, importance_labels, metrics
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}\n\nPlease run `python train_model.py` first.")
        st.stop()

model, scaler, feature_names, importances, importance_labels, model_metrics = load_model_assets()

# ─────────────────────────────────────────
# FEATURE ORDER (must match training)
# ─────────────────────────────────────────
FEATURE_ORDER = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                 "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

FEATURE_RANGES = {
    "age":      (1,   120),
    "cp":       (0,   3),
    "trestbps": (80,  220),
    "chol":     (100, 600),
    "fbs":      (0,   1),
    "restecg":  (0,   2),
    "thalach":  (60,  220),
    "exang":    (0,   1),
    "oldpeak":  (0.0, 7.0),
    "slope":    (0,   2),
    "ca":       (0,   3),
    "thal":     (1,   3),
}

RISK_HIGH_CUTOFF     = float(model_metrics.get("risk_thresholds", {}).get("high",     0.75))
RISK_MODERATE_CUTOFF = float(model_metrics.get("risk_thresholds", {}).get("moderate", 0.50))

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_risk_level(prediction: int, confidence: float) -> str:
    if prediction != 1:
        return "LOW"
    if confidence >= RISK_HIGH_CUTOFF:
        return "HIGH"
    if confidence > RISK_MODERATE_CUTOFF:
        return "MODERATE"
    return "LOW"


def run_prediction(features: dict) -> dict:
    arr = pd.DataFrame([[features[n] for n in FEATURE_ORDER]], columns=FEATURE_ORDER)
    scaled = scaler.transform(arr)
    pred = int(model.predict(scaled)[0])
    probs = model.predict_proba(scaled)[0]
    conf = float(probs[pred])

    top5_idx = np.argsort(model.feature_importances_)[::-1][:5]
    top5 = [
        {"feature": FEATURE_ORDER[i], "importance": round(float(model.feature_importances_[i]), 4)}
        for i in top5_idx
    ]

    return {
        "prediction":   pred,
        "label":        "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected",
        "confidence":   round(conf * 100, 2),
        "risk_level":   get_risk_level(pred, conf),
        "top5_factors": top5,
        "advice": (
            "High cardiac risk detected. Please consult a cardiologist immediately."
            if pred == 1
            else "Low cardiac risk at this time. Maintain a healthy lifestyle and regular check-ups."
        ),
    }

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
/* General */
body { font-family: 'Segoe UI', sans-serif; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid #0f3460;
    margin-bottom: 12px;
}
.metric-card .metric-value { font-size: 2.2rem; font-weight: 700; color: #e94560; }
.metric-card .metric-label { font-size: 0.85rem; color: #a0a0a0; margin-top: 4px; }

/* Risk badges */
.risk-high     { background:#dc3545; color:white; padding:4px 12px; border-radius:20px; font-weight:600; }
.risk-moderate { background:#fd7e14; color:white; padding:4px 12px; border-radius:20px; font-weight:600; }
.risk-low      { background:#28a745; color:white; padding:4px 12px; border-radius:20px; font-weight:600; }

/* Result box */
.result-disease  { background: linear-gradient(135deg,#dc354520,#dc354510); border:2px solid #dc3545; border-radius:12px; padding:24px; }
.result-healthy  { background: linear-gradient(135deg,#28a74520,#28a74510); border:2px solid #28a745; border-radius:12px; padding:24px; }

/* Progress bar override */
.stProgress > div > div > div { background-color: #e94560 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ HeartGuard")
    st.markdown("*AI-Powered Heart Disease Prediction*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔬 Predict", "📊 Analytics & EDA", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()

    bm = model_metrics.get("best_model", {})
    st.markdown("**Model Performance**")
    st.markdown(f"✅ Accuracy : **{bm.get('accuracy',0)*100:.1f}%**")
    st.markdown(f"🎯 Precision: **{bm.get('precision',0)*100:.1f}%**")
    st.markdown(f"📡 Recall   : **{bm.get('recall',0)*100:.1f}%**")
    st.markdown(f"📈 ROC-AUC  : **{bm.get('roc_auc',0):.3f}**")
    st.divider()
    st.caption("Ganpat University | UVPCE\nKartikkumar Sangada – 22012011044")

# ─────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("# ❤️ HeartGuard")
    st.markdown("### AI-Powered Cardiovascular Risk Assessment System")
    st.markdown("---")

    ds = model_metrics.get("dataset", {})
    bm = model_metrics.get("best_model", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{bm.get('accuracy',0)*100:.1f}%</div>
            <div class="metric-label">Model Accuracy</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{bm.get('roc_auc',0):.3f}</div>
            <div class="metric-label">ROC-AUC Score</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{ds.get('records',0)}</div>
            <div class="metric-label">Training Records</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{ds.get('features',0)}</div>
            <div class="metric-label">Clinical Features</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("### 🫀 About HeartGuard")
        st.markdown("""
HeartGuard is a machine learning-based clinical decision support tool that predicts 
the likelihood of heart disease using **13 clinical and demographic features** from 
the **UCI Cleveland Heart Disease Dataset**.

The system uses a tuned **Random Forest** classifier achieving industry-grade 
predictive performance. It is designed to assist healthcare professionals and 
patients in early cardiovascular risk stratification.

**Key Capabilities:**
- 🔬 Real-time individual risk prediction
- 📊 Exploratory data analysis & model insights
- 📈 Multi-model comparison (RF, LR, DT, KNN, SVM)
- 🔍 Top-5 risk factor explanation per prediction
        """)

    with col_b:
        st.markdown("### 🧬 Clinical Features Used")
        features_info = {
            "age":      "Age (years)",
            "sex":      "Sex (0=Female, 1=Male)",
            "cp":       "Chest Pain Type (0–3)",
            "trestbps": "Resting Blood Pressure (mmHg)",
            "chol":     "Serum Cholesterol (mg/dl)",
            "fbs":      "Fasting Blood Sugar > 120 mg/dl",
            "restecg":  "Resting ECG Results (0–2)",
            "thalach":  "Max Heart Rate Achieved",
            "exang":    "Exercise-Induced Angina (0/1)",
            "oldpeak":  "ST Depression (exercise vs rest)",
            "slope":    "Slope of Peak Exercise ST Segment",
            "ca":       "No. of Major Vessels (0–3)",
            "thal":     "Thalassemia (1=Normal, 2=Fixed, 3=Rev.)",
        }
        for feat, desc in features_info.items():
            st.markdown(f"- **{feat}**: {desc}")

    st.markdown("---")
    st.info("👉 Go to **🔬 Predict** in the sidebar to run a heart disease risk assessment.")

# ─────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────
elif page == "🔬 Predict":
    st.markdown("# 🔬 Heart Disease Risk Prediction")
    st.markdown("Fill in the patient's clinical values and click **Run Prediction**.")
    st.markdown("---")

    with st.form("prediction_form"):
        st.markdown("### 👤 Patient Information")
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name", placeholder="e.g. John Doe")
        with col2:
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")

        st.markdown("### 🩺 Clinical Parameters")
        c1, c2, c3 = st.columns(3)

        with c1:
            age      = st.number_input("Age (years)",           min_value=1,   max_value=120, value=52)
            cp       = st.selectbox("Chest Pain Type (cp)",
                                    options=[0, 1, 2, 3],
                                    format_func=lambda x: {
                                        0: "0 – Typical Angina",
                                        1: "1 – Atypical Angina",
                                        2: "2 – Non-Anginal Pain",
                                        3: "3 – Asymptomatic"}[x])
            trestbps = st.number_input("Resting BP (mmHg)",     min_value=80,  max_value=220, value=125)
            chol     = st.number_input("Cholesterol (mg/dl)",   min_value=100, max_value=600, value=212)
            fbs      = st.selectbox("Fasting Blood Sugar > 120 (fbs)",
                                    options=[0, 1],
                                    format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")

        with c2:
            restecg  = st.selectbox("Resting ECG (restecg)",
                                    options=[0, 1, 2],
                                    format_func=lambda x: {
                                        0: "0 – Normal",
                                        1: "1 – ST-T Abnormality",
                                        2: "2 – LV Hypertrophy"}[x])
            thalach  = st.number_input("Max Heart Rate",        min_value=60,  max_value=220, value=168)
            exang    = st.selectbox("Exercise-Induced Angina",
                                    options=[0, 1],
                                    format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
            oldpeak  = st.number_input("ST Depression (oldpeak)",
                                       min_value=0.0, max_value=7.0, value=1.0, step=0.1, format="%.1f")

        with c3:
            slope    = st.selectbox("ST Slope (slope)",
                                    options=[0, 1, 2],
                                    format_func=lambda x: {
                                        0: "0 – Upsloping",
                                        1: "1 – Flat",
                                        2: "2 – Downsloping"}[x])
            ca       = st.selectbox("Major Vessels (ca)",
                                    options=[0, 1, 2, 3],
                                    format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''}")
            thal     = st.selectbox("Thalassemia (thal)",
                                    options=[1, 2, 3],
                                    format_func=lambda x: {
                                        1: "1 – Normal",
                                        2: "2 – Fixed Defect",
                                        3: "3 – Reversible Defect"}[x])

        submitted = st.form_submit_button("🔍 Run Prediction", use_container_width=True, type="primary")

    if submitted:
        if not patient_name.strip():
            st.warning("⚠️ Please enter the patient name.")
        else:
            features = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
                "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
                "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
            }
            result = run_prediction(features)

            st.markdown("---")
            st.markdown("## 📋 Prediction Results")

            # Result banner
            if result["prediction"] == 1:
                risk_badge = {
                    "HIGH":     '<span class="risk-high">🔴 HIGH RISK</span>',
                    "MODERATE": '<span class="risk-moderate">🟠 MODERATE RISK</span>',
                }.get(result["risk_level"], '<span class="risk-moderate">🟠 MODERATE RISK</span>')
                st.markdown(f"""
<div class="result-disease">
<h2>⚠️ {result['label']}</h2>
<p><b>Patient:</b> {patient_name.strip()} &nbsp;|&nbsp; <b>Risk Level:</b> {risk_badge} &nbsp;|&nbsp;
<b>Confidence:</b> {result['confidence']}%</p>
<p>💬 {result['advice']}</p>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="result-healthy">
<h2>✅ {result['label']}</h2>
<p><b>Patient:</b> {patient_name.strip()} &nbsp;|&nbsp;
<span class="risk-low">🟢 LOW RISK</span> &nbsp;|&nbsp;
<b>Confidence:</b> {result['confidence']}%</p>
<p>💬 {result['advice']}</p>
</div>""", unsafe_allow_html=True)

            st.markdown("")

            # Metrics + top factors
            col_m, col_f = st.columns([1, 1])
            with col_m:
                st.markdown("### 📊 Prediction Metrics")
                st.metric("Prediction", result["label"])
                st.metric("Confidence", f"{result['confidence']}%")
                st.metric("Risk Level", result["risk_level"])
                st.progress(result["confidence"] / 100)

            with col_f:
                st.markdown("### 🔍 Top 5 Contributing Factors")
                for item in result["top5_factors"]:
                    pct = item["importance"] * 100
                    st.markdown(f"**{item['feature']}** — {pct:.2f}%")
                    st.progress(item["importance"])

            # Input summary table
            st.markdown("### 📋 Input Summary")
            summary_data = {
                "Feature": ["Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
                             "Fasting BS", "RestECG", "Max HR", "Exang", "ST Depression",
                             "ST Slope", "CA", "Thal"],
                "Value":   [age,
                             "Male" if sex == 1 else "Female",
                             cp, trestbps, chol,
                             "Yes" if fbs == 1 else "No",
                             restecg, thalach,
                             "Yes" if exang == 1 else "No",
                             oldpeak, slope, ca, thal],
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────
# PAGE: ANALYTICS & EDA
# ─────────────────────────────────────────
elif page == "📊 Analytics & EDA":
    st.markdown("# 📊 Analytics & Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗂️ Dataset", "📈 EDA Plots", "🔥 Correlation", "🤖 Model Comparison", "🏆 Feature Importances"
    ])

    # ── Tab 1: Dataset ──
    with tab1:
        st.markdown("### Dataset Overview")
        ds = model_metrics.get("dataset", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", ds.get("records", "N/A"))
        c2.metric("Features",      ds.get("features", "N/A"))
        c3.metric("Train Size",    ds.get("train_size", "N/A"))
        c4.metric("Test Size",     ds.get("test_size", "N/A"))

        cd = ds.get("class_distribution", {})
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Healthy (0)", cd.get("healthy", "N/A"))
        with col_b:
            st.metric("Disease (1)", cd.get("disease", "N/A"))

        if DATASET_PATH.exists():
            df = pd.read_csv(DATASET_PATH)
            if len(df.columns) == 14 and df.columns[0] != "age":
                df.columns = ["age","sex","cp","trestbps","chol","fbs","restecg",
                               "thalach","exang","oldpeak","slope","ca","thal","target"]
            df.replace("?", np.nan, inplace=True)
            df.dropna(inplace=True)
            df = df.astype(float)
            df["target"] = (df["target"] > 0).astype(int)

            st.markdown("#### Sample Records")
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown("#### Descriptive Statistics")
            st.dataframe(df.describe().round(3), use_container_width=True)
        else:
            st.info("heart.csv not found — showing model_metrics only.")

    # ── Tab 2: EDA Plots ──
    with tab2:
        st.markdown("### Distribution Plots")
        img_path = STATIC_IMG_DIR / "eda_distributions.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        elif DATASET_PATH.exists():
            df = pd.read_csv(DATASET_PATH)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            df[df["target"] == 0]["age"].hist(bins=15, color="#28a745", alpha=0.7, ax=axes[0], label="No Disease")
            df[df["target"] == 1]["age"].hist(bins=15, color="#dc3545", alpha=0.7, ax=axes[0], label="Disease")
            axes[0].set_title("Age Distribution by Target"); axes[0].legend()
            df[df["target"] == 0]["chol"].hist(bins=15, color="#28a745", alpha=0.7, ax=axes[1], label="No Disease")
            df[df["target"] == 1]["chol"].hist(bins=15, color="#dc3545", alpha=0.7, ax=axes[1], label="Disease")
            axes[1].set_title("Cholesterol Distribution by Target"); axes[1].legend()
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("EDA image not found. Run train_model.py to generate plots.")

        st.markdown("### ROC Curves – All Models")
        roc_path = STATIC_IMG_DIR / "roc_curves.png"
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.info("ROC curve image not found. Run train_model.py.")

    # ── Tab 3: Correlation ──
    with tab3:
        st.markdown("### Feature Correlation Heatmap")
        heatmap_path = STATIC_IMG_DIR / "correlation_heatmap.png"
        if heatmap_path.exists():
            st.image(str(heatmap_path), use_container_width=True)
        elif DATASET_PATH.exists():
            df = pd.read_csv(DATASET_PATH)
            fig, ax = plt.subplots(figsize=(12, 9))
            corr = df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                        linewidths=0.5, annot_kws={"size": 8}, ax=ax)
            ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Correlation heatmap not found. Run train_model.py.")

        st.markdown("### Confusion Matrix")
        cm_path = STATIC_IMG_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Confusion matrix image not found. Run train_model.py.")

    # ── Tab 4: Model Comparison ──
    with tab4:
        st.markdown("### Candidate Model Comparison")
        candidates = model_metrics.get("candidate_models", [])
        if candidates:
            df_models = pd.DataFrame([
                {
                    "Model":     m["name"],
                    "Accuracy":  f"{m.get('accuracy',0)*100:.1f}%",
                    "Precision": f"{m.get('precision',0)*100:.1f}%",
                    "Recall":    f"{m.get('recall',0)*100:.1f}%",
                    "F1-Score":  f"{m.get('f1_score',0)*100:.1f}%",
                    "ROC-AUC":   f"{m.get('roc_auc',0):.3f}",
                    "Status":    "✅ Deployed" if m.get("status") == "deployed" else "🔬 Tested",
                }
                for m in candidates
            ])
            st.dataframe(df_models, use_container_width=True, hide_index=True)

            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            names  = [m["name"] for m in candidates]
            accs   = [m.get("accuracy", 0) for m in candidates]
            aucs   = [m.get("roc_auc",  0) for m in candidates]
            x = np.arange(len(names))
            bars1 = ax.bar(x - 0.2, accs, 0.35, label="Accuracy", color="#3498db")
            bars2 = ax.bar(x + 0.2, aucs, 0.35, label="ROC-AUC",  color="#e74c3c")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No candidate models found in model_metrics.json.")

    # ── Tab 5: Feature Importances ──
    with tab5:
        st.markdown("### Feature Importances (Random Forest)")
        fi_path = STATIC_IMG_DIR / "feature_importances.png"
        if fi_path.exists():
            st.image(str(fi_path), use_container_width=True)
        elif importances and importance_labels:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors_bar = ["#dc3545" if i < 5 else "#6c757d" for i in range(len(importance_labels))]
            ax.bar(importance_labels, importances, color=colors_bar, edgecolor="white")
            ax.set_xlabel("Features"); ax.set_ylabel("Importance Score")
            ax.set_title("Feature Importances – Random Forest", fontsize=13, fontweight="bold")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

        if importance_labels and importances:
            st.markdown("#### Importance Values")
            df_fi = pd.DataFrame({"Feature": importance_labels, "Importance": [round(v, 4) for v in importances]})
            df_fi["Importance %"] = (df_fi["Importance"] * 100).round(2).astype(str) + "%"
            st.dataframe(df_fi, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About HeartGuard")
    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
### 🎓 Project Details
| Field | Info |
|---|---|
| Project | HeartGuard – Heart Disease Prediction |
| Author | Kartikkumar Sangada |
| Enrollment | 22012011044 |
| Degree | B.Tech. Computer Engineering |
| Semester | VIII |
| Institute | U.V. Patel College of Engineering |
| University | Ganpat University |

### 📦 Tech Stack
- **ML Framework**: scikit-learn (Random Forest)
- **UI**: Streamlit
- **Data**: UCI Cleveland Heart Disease Dataset
- **Libraries**: pandas, numpy, matplotlib, seaborn, joblib
        """)

    with col_b:
        st.markdown("""
### 🧠 Model Architecture
The HeartGuard system uses a **tuned Random Forest** ensemble classifier:

- **Base learners**: 100–300 decision trees
- **Hyperparameter tuning**: GridSearchCV with 5-fold StratifiedKFold
- **Feature scaling**: StandardScaler (fit on train, applied to test)
- **Comparison models**: Logistic Regression, Decision Tree, KNN, SVM

### ⚠️ Disclaimer
HeartGuard is a **research and educational tool** built to demonstrate 
machine learning applications in clinical decision support. It is **not** a 
certified medical device and should **not** replace professional medical advice.

Always consult a qualified cardiologist for clinical decisions.

### 📜 Dataset
[UCI Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)  
Creators: R. Detrano, M.D., Ph.D. et al.  
Records: 2004 | Features: 13 | Target: Binary (0/1)
        """)

    st.markdown("---")
    bm = model_metrics.get("best_model", {})
    st.markdown("### 🏆 Final Model Performance Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{bm.get('accuracy',  0)*100:.1f}%")
    c2.metric("Precision", f"{bm.get('precision', 0)*100:.1f}%")
    c3.metric("Recall",    f"{bm.get('recall',    0)*100:.1f}%")
    c4.metric("F1-Score",  f"{bm.get('f1_score',  0)*100:.1f}%")
    c5.metric("ROC-AUC",   f"{bm.get('roc_auc',   0):.3f}")

    if bm.get("best_params"):
        st.markdown("### ⚙️ Best Hyperparameters")
        st.json(bm["best_params"])
