"""
=============================================================
  app.py  —  Streamlit UI for Resume Classification
  Run: streamlit run app.py
=============================================================
"""

import re
import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        padding: 10px 0 5px 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: #e8f4f8;
        border-left: 6px solid #1f4e79;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .confidence-badge {
        background-color: #2ecc71;
        color: white;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1rem;
    }
    .stTextArea textarea {
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
MODELS_DIR = "."

MODEL_PATH = "model.pkl"
TFIDF_PATH = "tfidf.pkl"
ENCODER_PATH = "label_encoder.pkl"
import os
st.write("Current files:", os.listdir("."))
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
label_encoder = joblib.load(ENCODER_PATH)
    


# ─────────────────────────────────────────────
# TEXT CLEANING (must match train_model.py)
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\b(\+?\d[\d\s\-().]{7,}\d)\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict(resume_text: str, model, tfidf, le):
    cleaned    = clean_text(resume_text)
    vec        = tfidf.transform([cleaned])
    label      = model.predict(vec)[0]
    proba      = model.predict_proba(vec)[0]
    category   = le.inverse_transform([label])[0]
    confidence = proba.max() * 100
    all_probs  = dict(sorted(zip(le.classes_, proba), key=lambda x: x[1], reverse=True))
    return category, confidence, all_probs


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/resume.png", width=70)
    st.markdown("## 📘 About this App")
    st.info(
        "This app classifies a resume into one of **25 job categories** "
        "using NLP and Machine Learning (TF-IDF + Logistic Regression / Naive Bayes)."
    )
    st.markdown("### 🗂️ Supported Categories")
    categories = [
        "Data Science", "Python Developer", "Java Developer",
        "Testing", "HR", "DevOps Engineer", "Web Designing",
        "Blockchain", "ETL Developer", "Hadoop", "Sales",
        "Mechanical Engineer", "Civil Engineer", "SAP Developer",
        "Network Security Engineer", "Database", "Business Analyst",
        "DotNet Developer", "Automation Testing", "PMO",
        "Advocate", "Arts", "Health and fitness",
        "Electrical Engineering", "Operations Manager",
    ]
    for c in categories:
        st.markdown(f"• {c}")

    st.markdown("---")
    st.markdown("**Model:** Trained on Kaggle Resume Dataset  \n**Features:** TF-IDF (15k, bigrams)")


# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────
st.markdown('<p class="main-header">📄 Resume Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Paste your resume below and let AI predict the best-fit job category.</p>',
    unsafe_allow_html=True,
)

# ── Check model files exist
models_ready = all(os.path.exists(p) for p in [MODEL_PATH, TFIDF_PATH, ENCODER_PATH])

if not models_ready:
    st.error(
        "⚠️ Model files not found in `saved_models/`. "
        "Please run `python train_model.py` first to train and save the models."
    )
    st.stop()

# ── Load models
try:
    model, tfidf, le = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ── Input area
resume_input = st.text_area(
    "📝 Paste Resume Text Here",
    height=300,
    placeholder=(
        "Example:\nSkills: Python, Machine Learning, TensorFlow, scikit-learn, "
        "pandas, NLP, SQL, Tableau...\nExperience: 3 years as Data Scientist at XYZ Corp...\n"
        "Education: B.Tech Computer Science, ABC University 2021..."
    ),
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 Predict Job Role", use_container_width=True, type="primary")

# ── Prediction result
if predict_btn:
    if not resume_input.strip():
        st.warning("⚠️ Please paste some resume text before predicting.")
    elif len(resume_input.strip().split()) < 10:
        st.warning("⚠️ Resume text seems too short. Please paste a more detailed resume.")
    else:
        with st.spinner("Analysing resume …"):
            category, confidence, all_probs = predict(resume_input, model, tfidf, le)

        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")

        # ── Top prediction card
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown(
                f"""
                <div class="prediction-box">
                    <h3 style="margin:0; color:#1f4e79;">Predicted Job Category</h3>
                    <h1 style="margin:5px 0; color:#2c3e50;">💼 {category}</h1>
                    <span class="confidence-badge">Confidence: {confidence:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_b:
            # Gauge-style confidence meter
            fig, ax = plt.subplots(figsize=(3, 3))
            colors = ["#2ecc71" if confidence >= 70 else "#f39c12" if confidence >= 40 else "#e74c3c"]
            ax.pie(
                [confidence, 100 - confidence],
                colors=[colors[0], "#ecf0f1"],
                startangle=90,
                counterclock=False,
            )
            ax.add_patch(plt.Circle((0, 0), 0.65, color="white"))
            ax.text(0, 0, f"{confidence:.0f}%", ha="center", va="center",
                    fontsize=22, fontweight="bold", color=colors[0])
            ax.set_title("Confidence", fontsize=10)
            ax.axis("equal")
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        # ── Top-10 probability bar chart
        st.markdown("### 📊 Top 10 Category Probabilities")
        top10 = dict(list(all_probs.items())[:10])
        df_probs = pd.DataFrame({
            "Category": list(top10.keys()),
            "Probability (%)": [v * 100 for v in top10.values()]
        })

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        bar_colors = ["#1f4e79" if c == category else "#aed6f1" for c in df_probs["Category"]]
        bars = ax2.barh(df_probs["Category"], df_probs["Probability (%)"], color=bar_colors)
        ax2.set_xlabel("Probability (%)")
        ax2.set_title("Top 10 Predicted Categories", fontweight="bold")
        ax2.invert_yaxis()
        for bar in bars:
            w = bar.get_width()
            ax2.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{w:.1f}%", va="center", fontsize=9)
        legend_handles = [
            mpatches.Patch(color="#1f4e79", label="Top Prediction"),
            mpatches.Patch(color="#aed6f1", label="Other Categories"),
        ]
        ax2.legend(handles=legend_handles, loc="lower right")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ── Full probability table (collapsible)
        with st.expander("📋 View All Category Probabilities"):
            df_all = pd.DataFrame({
                "Rank": range(1, len(all_probs) + 1),
                "Category": list(all_probs.keys()),
                "Probability (%)": [f"{v*100:.2f}%" for v in all_probs.values()]
            }).set_index("Rank")
            st.dataframe(df_all, use_container_width=True)

        # ── Resume stats
        st.markdown("### 📝 Resume Text Statistics")
        word_count = len(resume_input.split())
        char_count = len(resume_input)
        unique_words = len(set(resume_input.lower().split()))
        c1, c2, c3 = st.columns(3)
        c1.metric("Word Count", word_count)
        c2.metric("Character Count", char_count)
        c3.metric("Unique Words", unique_words)

# ── Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "Resume Classifier · Built with Streamlit & scikit-learn · Internship Project"
    "</p>",
    unsafe_allow_html=True,
)
