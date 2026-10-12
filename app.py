import hashlib

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Customer Intent AI",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #10233f, #235b83);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0 0 .35rem 0; color: white;}
    .hero p {margin: 0; color: #e2edf7;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🤖 Customer Intent AI</h1>
      <p>Train a text-classification model and predict what a customer is asking for.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About this app")
    st.write(
        "This demo uses TF-IDF text features and Logistic Regression. "
        "Upload a CSV containing `utterance` and `intent` columns."
    )
    st.caption("Your dataset is processed by this app instance to train the model.")

uploaded_file = st.file_uploader(
    "Upload your dataset (CSV)",
    type=["csv"],
    help="Required columns: utterance and intent",
)

st.info(
    "Expected CSV columns: `utterance` (customer message) and `intent` (label). "
    "Example: “Where is my package?” → a delivery-related intent."
)

if uploaded_file is None:
    st.markdown("### Get started")
    st.write("Upload `Bitext_Sample_Customer_Service_Testing_Dataset.csv` to train the model.")
    st.markdown(
        """
        **What you'll get**
        - Dataset overview and label counts
        - Train/test accuracy and classification report
        - A text box to test your own customer messages
        """
    )
    st.stop()

try:
    raw_bytes = uploaded_file.getvalue()
    dataset_hash = hashlib.sha256(raw_bytes).hexdigest()
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

required_columns = {"utterance", "intent"}
missing = required_columns - set(df.columns)
if missing:
    st.error(
        "This CSV is missing required column(s): "
        + ", ".join(sorted(missing))
        + f". Available columns: {', '.join(map(str, df.columns))}"
    )
    st.stop()

df = df.dropna(subset=["utterance", "intent"]).copy()
df["utterance"] = df["utterance"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()
df = df[(df["utterance"] != "") & (df["intent"] != "")]

if len(df) < 5:
    st.error("Please upload a dataset with at least 5 valid rows.")
    st.stop()

if df["intent"].nunique() < 2:
    st.error("The `intent` column must contain at least two different classes.")
    st.stop()

@st.cache_resource(show_spinner="Training the model…")
def train_model(csv_bytes: bytes, content_hash: str):
    import io

    data = pd.read_csv(io.BytesIO(csv_bytes))
    data = data.dropna(subset=["utterance", "intent"]).copy()
    data["utterance"] = data["utterance"].astype(str).str.strip()
    data["intent"] = data["intent"].astype(str).str.strip()
    data = data[(data["utterance"] != "") & (data["intent"] != "")]

    X = data["utterance"]
    y = data["intent"]

    # Stratify only when every class has at least 2 samples.
    stratify_labels = y if y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_labels
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_test, predictions, zero_division=0
    )
    return model, accuracy, report, report_text, len(data), len(X_train), len(X_test)

try:
    model, accuracy, report, report_text, row_count, train_count, test_count = train_model(
        raw_bytes, dataset_hash
    )
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.caption("Check that the CSV has enough examples and valid text labels.")
    st.stop()

st.success("Model trained successfully.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Valid rows", f"{row_count:,}")
m2.metric("Unique intents", f"{df['intent'].nunique():,}")
m3.metric("Training rows", f"{train_count:,}")
m4.metric("Test accuracy", f"{accuracy:.1%}")

with st.expander("Preview dataset", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.bar_chart(df["intent"].value_counts().head(20))

with st.expander("Classification report", expanded=False):
    st.code(report_text, language="text")
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.round(3), use_container_width=True)

st.divider()
st.subheader("Try a customer message")
query = st.text_area(
    "Customer message",
    value="Where is my package and when will it arrive?",
    height=100,
    placeholder="Type a customer message here…",
)

if st.button("Predict intent", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([query.strip()])[0]
        st.success(f"Predicted intent: **{prediction}**")

        if hasattr(model.named_steps["classifier"], "predict_proba"):
            probabilities = model.predict_proba([query.strip()])[0]
            classes = model.named_steps["classifier"].classes_
            top_indices = probabilities.argsort()[::-1][:5]
            confidence_df = pd.DataFrame({
                "Intent": classes[top_indices],
                "Model score": probabilities[top_indices],
            })
            confidence_df["Model score"] = confidence_df["Model score"].map(
                lambda value: f"{value:.1%}"
            )
            st.markdown("**Top model scores**")
            st.dataframe(confidence_df, hide_index=True, use_container_width=True)
            st.caption(
                "These are model probability scores, not guaranteed real-world confidence."
            )

st.caption(
    "Educational demo only. Accuracy depends on the uploaded dataset and test split."
)
