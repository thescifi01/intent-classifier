import hashlib
import io
import html

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="IntentIQ | Customer Intelligence", page_icon="✦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#172033;--muted:#758198;--line:#e5eaf2;--brand:#6558f5}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:linear-gradient(180deg,#f7f8fc,#f9fbff 55%,#f5f7fb);color:var(--ink)}
.block-container{max-width:1260px;padding-top:1.5rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:#11182a;border-right:1px solid #222c43}
[data-testid="stSidebar"] *{color:#eef2ff}
[data-testid="stSidebar"] .stCaption{color:#aab5ce!important}
.brand{display:flex;align-items:center;gap:12px;margin-bottom:20px}
.brand-icon{width:44px;height:44px;border-radius:15px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#8b80ff,#5b4bea);font-size:24px;box-shadow:0 8px 20px #6558f533}
.brand-title{font-family:Manrope,sans-serif;font-size:20px;font-weight:800;letter-spacing:-.6px}
.brand-sub{font-size:10px;letter-spacing:1.8px;text-transform:uppercase;color:#aab5ce}
.hero{padding:32px;border-radius:25px;color:white;background:radial-gradient(circle at 86% 4%,#756aff 0%,transparent 30%),linear-gradient(120deg,#17223c,#252d53 55%,#4c42a8);box-shadow:0 18px 45px #252d531c;margin-bottom:24px}
.hero-pill{display:inline-block;padding:7px 11px;border:1px solid #ffffff33;background:#ffffff12;border-radius:999px;color:#e9eaff;font-size:10px;font-weight:700;letter-spacing:1px;margin-bottom:17px}
.hero h1{font-family:Manrope,sans-serif;color:white;font-size:clamp(28px,4vw,43px);line-height:1.12;letter-spacing:-1.5px;margin:0 0 12px}
.hero p{color:#d6ddf4;font-size:14px;max-width:650px;line-height:1.7;margin:0}
.section-title{font-family:Manrope,sans-serif;font-size:21px;font-weight:800;letter-spacing:-.6px;color:#172033;margin:8px 0 4px}
.section-sub{color:#778198;font-size:13px;margin-bottom:16px}
.metric-card{background:white;border:1px solid #e7ebf3;border-radius:17px;padding:18px;min-height:119px;box-shadow:0 6px 18px #17203305}
.metric-label{color:#778198;font-size:11px;font-weight:700;letter-spacing:.5px;margin-bottom:12px}
.metric-value{font-family:Manrope,sans-serif;color:#172033;font-size:28px;font-weight:800;letter-spacing:-1px;line-height:1.1}
.metric-note{color:#8c96a9;font-size:11px;margin-top:9px}
.panel{background:white;border:1px solid #e7ebf3;border-radius:19px;padding:21px 23px;box-shadow:0 6px 18px #17203305;margin-bottom:16px}
.panel-title{font-family:Manrope,sans-serif;font-size:16px;font-weight:800;color:#1c2740;margin-bottom:5px}
.panel-sub{color:#8791a5;font-size:12px;margin-bottom:15px}
.empty{background:white;border:1px solid #e7ebf3;border-radius:22px;padding:34px 22px;text-align:center}
.empty-icon{font-size:35px;margin-bottom:10px}
.empty-title{font-family:Manrope,sans-serif;font-size:20px;font-weight:800;color:#1d2740;margin-bottom:8px}
.empty-copy{max-width:540px;margin:auto;color:#7a859b;font-size:13px;line-height:1.7}
.step{display:flex;gap:12px;align-items:flex-start;padding:10px 0;border-bottom:1px solid #edf0f6}
.step:last-child{border-bottom:0}
.step-num{width:28px;height:28px;flex:0 0 28px;border-radius:9px;background:#efedff;color:#5d50e9;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:12px}
.step-title{color:#27324b;font-weight:700;font-size:13px;margin:1px 0 3px}
.step-copy{color:#8490a4;font-size:12px;line-height:1.5}
[data-testid="stFileUploader"]{background:#fff;border:1.5px dashed #cbd3e5;border-radius:16px;padding:12px}
[data-testid="stFileUploader"]:hover{border-color:#6558f5}
div.stButton>button{border-radius:12px;border:0;background:linear-gradient(110deg,#6659f5,#5144dc);color:white;font-weight:700;min-height:44px;box-shadow:0 8px 18px #6558f52a}
div.stButton>button:hover{background:linear-gradient(110deg,#594ce8,#4437c8);color:white;border:0}
.stTextArea textarea{border-radius:14px;border:1px solid #dfe5f0;background:#fff}
[data-testid="stDataFrame"]{border:1px solid #e7ebf3;border-radius:13px;overflow:hidden}
.footer{text-align:center;color:#9aa3b4;font-size:10px;letter-spacing:1px;padding:26px 0 0}
@media(max-width:700px){.block-container{padding:1rem .8rem 2rem}.hero{padding:23px 20px;border-radius:19px}.metric-card{padding:13px;min-height:105px}.metric-value{font-size:23px}.panel{padding:16px}}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">✦</div><div><div class="brand-title">IntentIQ</div><div class="brand-sub">Customer intelligence</div></div></div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Overview", "Intent Studio", "Model insights"])
    st.markdown("---")
    st.markdown("**YOUR WORKFLOW**")
    st.caption("01  ·  Upload labeled data")
    st.caption("02  ·  Train your classifier")
    st.caption("03  ·  Explore predictions")
    st.markdown("---")
    st.markdown("**MODEL ENGINE**")
    st.caption("TF-IDF + Logistic Regression")
    st.caption("Powered by Python and Streamlit")
    st.markdown('<div style="margin-top:24px;padding:13px;border-radius:13px;background:#ffffff0d;border:1px solid #ffffff14"><div style="font-size:10px;color:#9faed0">STATUS</div><div style="font-size:13px;font-weight:700;margin-top:5px">✦ &nbsp; Ready to analyse</div></div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-pill">✦ &nbsp; AI-POWERED TEXT ANALYTICS</div>
  <h1>Understand every<br>customer conversation.</h1>
  <p>Turn customer messages into meaningful intent signals. Train a classifier, explore your data, and test predictions in one clean workspace.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.5, 1], gap="large")
with left:
    st.markdown('<div class="section-title">Connect your dataset</div><div class="section-sub">Upload a labeled CSV to unlock your analytics workspace.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop your customer dataset here", type=["csv"], help="Required columns: utterance and intent")
    st.caption("Required fields: utterance = customer message · intent = target label")
with right:
    st.markdown("""
    <div class="panel" style="margin-top:8px">
      <div class="panel-title">Quick start</div><div class="panel-sub">Three steps to your first prediction</div>
      <div class="step"><div class="step-num">1</div><div><div class="step-title">Upload a CSV</div><div class="step-copy">Bring a dataset with customer messages and intent labels.</div></div></div>
      <div class="step"><div class="step-num">2</div><div><div class="step-title">Train automatically</div><div class="step-copy">Build a TF-IDF and Logistic Regression pipeline.</div></div></div>
      <div class="step"><div class="step-num">3</div><div><div class="step-title">Test your messages</div><div class="step-copy">See predicted intents and the model's top scores.</div></div></div>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown('<div class="empty"><div class="empty-icon">✧</div><div class="empty-title">Your intelligence workspace is ready</div><div class="empty-copy">Start by uploading a CSV above. Include at least five valid examples and two different intent labels. No dataset is bundled with this app.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">INTENTIQ · BUILT WITH PYTHON & STREAMLIT</div>', unsafe_allow_html=True)
    st.stop()

try:
    raw_bytes = uploaded_file.getvalue()
    dataset_hash = hashlib.sha256(raw_bytes).hexdigest()
    df = pd.read_csv(io.BytesIO(raw_bytes))
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

required_columns = {"utterance", "intent"}
missing = required_columns - set(df.columns)
if missing:
    st.error("Missing required column(s): " + ", ".join(sorted(missing)) + f". Available columns: {', '.join(map(str, df.columns))}")
    st.stop()

df = df.dropna(subset=["utterance", "intent"]).copy()
df["utterance"] = df["utterance"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()
df = df[(df["utterance"] != "") & (df["intent"] != "")]
if len(df) < 5:
    st.error("Please upload a dataset with at least 5 valid rows.")
    st.stop()
if df["intent"].nunique() < 2:
    st.error("The intent column must contain at least two different classes.")
    st.stop()

@st.cache_resource(show_spinner="Training your intent model…")
def train_model(csv_bytes: bytes, content_hash: str):
    data = pd.read_csv(io.BytesIO(csv_bytes))
    data = data.dropna(subset=["utterance", "intent"]).copy()
    data["utterance"] = data["utterance"].astype(str).str.strip()
    data["intent"] = data["intent"].astype(str).str.strip()
    data = data[(data["utterance"] != "") & (data["intent"] != "")]
    X, y = data["utterance"], data["intent"]
    stratify_labels = y if y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_labels)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report_text = classification_report(y_test, predictions, zero_division=0)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    return model, accuracy, report, report_text, len(data), len(X_train), len(X_test), y_test, predictions

try:
    model, accuracy, report, report_text, row_count, train_count, test_count, y_test, predictions = train_model(raw_bytes, dataset_hash)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.caption("Check the dataset size and ensure the intent labels contain valid examples.")
    st.stop()

st.markdown('<div class="section-title">Workspace overview</div><div class="section-sub">Your model is trained. Here is a snapshot of your data and evaluation.</div>', unsafe_allow_html=True)

def metric_card(label, value, note):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4, gap="medium")
with c1:
    metric_card("VALID EXAMPLES", f"{row_count:,}", "Clean messages available")
with c2:
    metric_card("INTENT CLASSES", f"{df['intent'].nunique():,}", "Unique labels detected")
with c3:
    metric_card("TRAINING SET", f"{train_count:,}", "Examples used to fit")
with c4:
    metric_card("TEST ACCURACY", f"{accuracy:.1%}", f"Measured on {test_count:,} held-out rows")

if page == "Overview":
    chart_col, info_col = st.columns([1.1, 1], gap="large")
    with chart_col:
        st.markdown('<div class="panel"><div class="panel-title">Intent distribution</div><div class="panel-sub">Most common labels in the uploaded data</div>', unsafe_allow_html=True)
        st.bar_chart(df["intent"].value_counts().head(12), color="#7568f6", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with info_col:
        st.markdown('<div class="panel"><div class="panel-title">Evaluation snapshot</div><div class="panel-sub">Held-out test split · random state 42</div>', unsafe_allow_html=True)
        st.markdown(f"### {accuracy:.1%}")
        st.progress(float(max(0, min(1, accuracy))))
        st.caption("Accuracy is the share of test examples classified correctly. Results depend on data quality and split.")
        st.markdown("---")
        st.markdown("**Model pipeline**")
        st.write("1. TF-IDF text vectorisation")
        st.write("2. Logistic Regression classifier")
        st.write("3. Held-out evaluation")
        st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("Preview uploaded data"):
        st.dataframe(df.head(30), use_container_width=True, hide_index=True)

elif page == "Intent Studio":
    st.markdown('<div class="panel"><div class="panel-title">Intent prediction studio</div><div class="panel-sub">Enter a customer message and inspect the model output.</div>', unsafe_allow_html=True)
    query = st.text_area("Customer message", value="Where is my package and when will it arrive?", height=120, placeholder="Type a customer message here…")
    if st.button("✦  Analyse customer intent", use_container_width=True, type="primary"):
        if not query.strip():
            st.warning("Enter a customer message before running a prediction.")
        else:
            prediction = model.predict([query.strip()])[0]
            st.markdown("---")
            st.markdown("**PREDICTED INTENT**")
            st.markdown(f'<div class="panel" style="background:linear-gradient(120deg,#f1efff,#fff);border-color:#ded9ff"><div style="font-size:10px;letter-spacing:2px;color:#7166f6;font-weight:700">MODEL OUTPUT</div><div style="font-family:Manrope,sans-serif;font-size:27px;font-weight:800;color:#4f43cf;overflow-wrap:anywhere">{html.escape(str(prediction))}</div><div style="color:#7c8499;font-size:12px;margin-top:7px">Predicted label for the submitted customer message</div></div>', unsafe_allow_html=True)
            probabilities = model.predict_proba([query.strip()])[0]
            classes = model.named_steps["classifier"].classes_
            top_indices = probabilities.argsort()[::-1][:5]
            confidence_df = pd.DataFrame({"Intent": classes[top_indices], "Model score": probabilities[top_indices]})
            confidence_df["Model score"] = confidence_df["Model score"].map(lambda value: f"{value:.1%}")
            st.markdown("**Top matching labels**")
            st.dataframe(confidence_df, hide_index=True, use_container_width=True)
            st.caption("Model scores are not guaranteed real-world confidence.")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Model insights":
    st.markdown('<div class="section-title">Model evaluation</div><div class="section-sub">Performance on the held-out test split. Small or imbalanced datasets can produce unstable metrics.</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title">Classification report</div><div class="panel-sub">Precision, recall, F1-score and support by label</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("View full text report"):
        st.code(report_text, language="text")
    with st.expander("View test examples"):
        test_messages = df.loc[y_test.index, "utterance"].reset_index(drop=True)
        evaluation = pd.DataFrame({"Customer message": test_messages, "Actual intent": list(y_test.reset_index(drop=True)), "Predicted intent": list(predictions)})
        evaluation["Result"] = evaluation.apply(lambda row: "Correct" if row["Actual intent"] == row["Predicted intent"] else "Review", axis=1)
        st.dataframe(evaluation, use_container_width=True, hide_index=True)

st.markdown('<div class="footer">INTENTIQ · BUILT WITH PYTHON & STREAMLIT · RESULTS DEPEND ON YOUR DATA</div>', unsafe_allow_html=True)
