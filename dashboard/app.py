import json
from pathlib import Path
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="EduPredict AI", page_icon="🎓", layout="wide")
st.markdown("""
<style>

/* Main Page Background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #172554 50%,
        #312e81 100%
    );
    color: #ffffff;
}

/* Main Content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Main Headings */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Paragraph and Labels */
p, label, .stMarkdown {
    color: #e2e8f0;
}

/* Input Text Area */
textarea {
    background-color: #ffffff !important;
    color: #1e293b !important;
    -webkit-text-fill-color: #1e293b !important;
    font-size: 15px !important;
    border: 2px solid #818cf8 !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #6366f1, #a855f7);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: 700;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #4f46e5, #9333ea);
    transform: translateY(-2px);
}

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 15px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.12);
}

</style>
""", unsafe_allow_html=True)

st.title("🎓 EduPredict AI")
st.caption("Student Dropout & Academic Success Intelligence | ML-18 Capstone")
api_url = st.sidebar.text_input("API Base URL", "http://127.0.0.1:8000").rstrip("/")
if st.sidebar.button("Refresh dashboard"):
    st.rerun()

def get_json(path):
    try:
        response = requests.get(f"{api_url}{path}", timeout=8)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"API unavailable: {exc}")
        return None

health = get_json("/health")
monitor = get_json("/monitoring/summary")
metrics_path = Path(__file__).resolve().parents[1] / "artifacts" / "metrics.json"
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

c1,c2,c3,c4 = st.columns(4)
c1.metric("API Status", "Online" if health and health.get("status")=="ok" else "Offline")
c2.metric("Model", metrics.get("best_model","Not trained"))
c3.metric("Test Accuracy", f"{metrics.get('test_accuracy',0):.1%}" if metrics else "—")
c4.metric("Macro F1", f"{metrics.get('test_f1_macro',0):.3f}" if metrics else "—")

left,right = st.columns([1.25,1])
with left:
    st.subheader("🔮 Student Outcome Prediction")
    schema = get_json("/schema")
    if schema:
        sample = schema.get("sample_input", {})
        with st.form("prediction_form"):
            st.caption("Edit the sample values. All required dataset features are included.")
            raw = st.text_area("Student feature values (JSON)", json.dumps(sample, indent=2), height=300)
            submitted = st.form_submit_button("Predict student outcome")
        if submitted:
            try:
                payload = {"student": json.loads(raw)}
                result = requests.post(f"{api_url}/predict", json=payload, timeout=20)
                if result.ok:
                    data = result.json()
                    st.success(f"Predicted outcome: {data['prediction']}")
                    if data.get("probabilities"):
                        st.bar_chart(pd.Series(data["probabilities"], name="Probability"))
                    st.caption(data.get("disclaimer",""))
                else:
                    st.error(result.text)
            except Exception as exc:
                st.error(f"Please check the JSON input: {exc}")

with right:
    st.subheader("🛡️ Production Monitoring")
    if monitor:
        st.metric("Prediction Requests", monitor.get("prediction_count",0))
        st.write("Monitoring status:", monitor.get("status","unknown").replace("_"," ").title())
        drift = monitor.get("drift",{})
        if drift:
            st.dataframe(pd.DataFrame([{"Feature":k,"PSI":v} for k,v in drift.items()]).sort_values("PSI",ascending=False), use_container_width=True)
        else:
            st.info("Make at least 5 predictions to calculate basic drift indicators.")
    st.subheader("📊 Model Comparison")
    comparison = metrics.get("model_comparison",{})
    if comparison:
        chart = pd.DataFrame([{"Model":k,"CV Macro F1":v["cv_f1_macro_mean"]} for k,v in comparison.items()]).set_index("Model")
        st.bar_chart(chart)

st.divider()
st.caption("Responsible use: this is a decision-support prototype, not a definitive judgment about any student.")
