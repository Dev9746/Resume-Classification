import re
import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# PAGE CONFIG
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume Classifier")
st.write("Paste your resume below and let AI predict the best-fit job category.")

# TEXT CLEANING
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        word for word in text.split()
        if word not in ENGLISH_STOP_WORDS and len(word) > 2
    ]

    return " ".join(tokens)


# LOAD MODELS
MODEL_PATH = "model.pkl"
TFIDF_PATH = "tfidf.pkl"
ENCODER_PATH = "label_encoder.pkl"

st.write("Current files:", os.listdir("."))

if not (
    os.path.exists(MODEL_PATH)
    and os.path.exists(TFIDF_PATH)
    and os.path.exists(ENCODER_PATH)
):
    st.error("❌ Model files not found.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# PREDICTION FUNCTION
def predict_resume(text):
    cleaned = clean_text(text)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    category = label_encoder.inverse_transform([prediction])[0]

    confidence = probabilities.max() * 100

    all_probs = dict(
        sorted(
            zip(label_encoder.classes_, probabilities),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    return category, confidence, all_probs


# INPUT
resume_input = st.text_area(
    "📝 Paste Resume Text Here",
    height=300
)

if st.button("🔍 Predict Job Role"):

    if not resume_input.strip():
        st.warning("Please paste resume text.")
    else:

        category, confidence, all_probs = predict_resume(resume_input)

        st.success(f"Predicted Category: {category}")
        st.info(f"Confidence: {confidence:.2f}%")

        # Top probabilities
        st.subheader("Top Predictions")

        top10 = list(all_probs.items())[:10]

        df = pd.DataFrame(
            top10,
            columns=["Category", "Probability"]
        )

        df["Probability"] = df["Probability"] * 100

        st.dataframe(df)

        # Chart
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.barh(
            df["Category"],
            df["Probability"]
        )

        ax.invert_yaxis()

        ax.set_xlabel("Probability (%)")
        ax.set_title("Top 10 Categories")

        st.pyplot(fig)


st.markdown("---")
st.caption(
    "Resume Classifier • Built with Streamlit & scikit-learn"
)
