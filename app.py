"""
=============================================================
  app.py — Streamlit UI for Resume Classification
=============================================================
"""

import re
import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# CUSTOM CSS
# ==================================================
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

.stTextArea textarea {
    font-size: 0.92rem;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# PAGE HEADER
# ==================================================
st.markdown(
    '<p class="main-header">📄 Resume Classifier</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-header">'
    'Paste your resume below and let AI predict the best-fit job category.'
    '</p>',
    unsafe_allow_html=True,
)

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/resume.png",
        width=70
    )

    st.markdown("## 📘 About this App")

    st.info(
        "This app classifies resumes into one of "
        "**25 job categories** using NLP and "
        "Machine Learning."
    )

    st.markdown("### 🗂️ Supported Categories")

    categories = [
        "Data Science",
        "Python Developer",
        "Java Developer",
        "Testing",
        "HR",
        "DevOps Engineer",
        "Web Designing",
        "Blockchain",
        "ETL Developer",
        "Hadoop",
        "Sales",
        "Mechanical Engineer",
        "Civil Engineer",
        "SAP Developer",
        "Network Security Engineer",
        "Database",
        "Business Analyst",
        "DotNet Developer",
        "Automation Testing",
        "PMO",
        "Advocate",
        "Arts",
        "Health and Fitness",
        "Electrical Engineering",
        "Operations Manager",
    ]

    for category in categories:
        st.markdown(f"• {category}")

    st.markdown("---")

    st.markdown(
        "**Model:** Trained on Kaggle Resume Dataset\n\n"
        "**Features:** TF-IDF + Machine Learning"
    )

# ==================================================
# MODEL PATHS
# ==================================================
MODEL_PATH = "model.pkl"
TFIDF_PATH = "tfidf.pkl"
ENCODER_PATH = "label_encoder.pkl"

# ==================================================
# CHECK MODEL FILES
# ==================================================
if not (
    os.path.exists(MODEL_PATH)
    and os.path.exists(TFIDF_PATH)
    and os.path.exists(ENCODER_PATH)
):
    st.error(
        "❌ Model files not found.\n\n"
        "Please upload:\n"
        "- model.pkl\n"
        "- tfidf.pkl\n"
        "- label_encoder.pkl"
    )

    st.write("Current Files:", os.listdir("."))

    st.stop()

# ==================================================
# LOAD MODELS
# ==================================================
@st.cache_resource
def load_models():

    model = joblib.load(MODEL_PATH)

    tfidf = joblib.load(TFIDF_PATH)

    label_encoder = joblib.load(ENCODER_PATH)

    return model, tfidf, label_encoder


try:

    model, tfidf, label_encoder = load_models()

except Exception as e:

    st.error(f"Error loading models: {e}")

    st.stop()
  
  # ==================================================
# TEXT CLEANING
# ==================================================
def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove Emails
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove Phone Numbers
    text = re.sub(
        r"\b(\+?\d[\d\s\-().]{7,}\d)\b",
        " ",
        text
    )

    # Keep only alphabets
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Remove stopwords
    tokens = [
        word
        for word in text.split()
        if word not in ENGLISH_STOP_WORDS
        and len(word) > 2
    ]

    return " ".join(tokens)


# ==================================================
# PREDICTION FUNCTION
# ==================================================
def predict_resume(text):

    cleaned = clean_text(text)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    category = label_encoder.inverse_transform(
        [prediction]
    )[0]

    confidence = probabilities.max() * 100

    all_probs = dict(
        sorted(
            zip(
                label_encoder.classes_,
                probabilities
            ),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    return (
        category,
        confidence,
        all_probs,
    )


# ==================================================
# PDF UPLOAD
# ==================================================
st.markdown("## 📂 Upload Resume")

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

resume_input = ""

if uploaded_file is not None:

    try:

        pdf = PdfReader(uploaded_file)

        extracted_text = ""

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                extracted_text += page_text + "\n"

        resume_input = extracted_text

        st.success(
            "✅ PDF uploaded successfully."
        )

    except Exception as e:

        st.error(
            f"Error reading PDF: {e}"
        )


# ==================================================
# RESUME INPUT
# ==================================================
resume_input = st.text_area(
    "📝 Paste Resume Text Here",
    value=resume_input,
    height=300,
    placeholder=(
        "Example:\n"
        "Skills: Python, Machine Learning, SQL...\n"
        "Experience: 3 years as Data Scientist...\n"
        "Education: B.Tech Computer Science..."
    ),
)


# ==================================================
# PREDICT BUTTON
# ==================================================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    predict_btn = st.button(
        "🔍 Predict Job Role",
        use_container_width=True,
        type="primary"
    )
  # ==================================================
# PREDICTION RESULTS
# ==================================================
if predict_btn:

    if not resume_input.strip():

        st.warning(
            "⚠️ Please paste resume text."
        )

    elif len(resume_input.strip().split()) < 10:

        st.warning(
            "⚠️ Resume text seems too short.\n"
            "Please provide more details."
        )

    else:

        with st.spinner(
            "🤖 Analysing Resume..."
        ):

            category, confidence, all_probs = predict_resume(
                resume_input
            )

        st.markdown("---")

        st.markdown(
            "## 🎯 Prediction Results"
        )

        # ==========================================
        # Prediction + Gauge Meter
        # ==========================================
        col_a, col_b = st.columns([2, 1])

        with col_a:

            st.info("🎯 Predicted Job Category")

            st.success(f"💼 {category}")

            st.metric(
                label="Confidence",
                value=f"{confidence:.1f}%"
            )

        with col_b:

            fig, ax = plt.subplots(
                figsize=(3, 3)
            )

            gauge_color = (
                "#2ecc71"
                if confidence >= 70
                else "#f39c12"
                if confidence >= 40
                else "#e74c3c"
            )

            ax.pie(
                [confidence, 100 - confidence],
                colors=[
                    gauge_color,
                    "#ecf0f1"
                ],
                startangle=90,
                counterclock=False,
            )

            ax.add_patch(
                plt.Circle(
                    (0, 0),
                    0.65,
                    color="white"
                )
            )

            ax.text(
                0,
                0,
                f"{confidence:.0f}%",
                ha="center",
                va="center",
                fontsize=22,
                fontweight="bold",
                color=gauge_color,
            )

            ax.set_title(
                "Confidence"
            )

            ax.axis("equal")

            st.pyplot(fig)

            plt.close(fig)

        # ==========================================
        # TOP 10 PROBABILITIES
        # ==========================================
        st.markdown(
            "### 📊 Top 10 Category Probabilities"
        )

        top10 = dict(
            list(all_probs.items())[:10]
        )

        df_probs = pd.DataFrame(
            {
                "Category": list(
                    top10.keys()
                ),
                "Probability (%)": [
                    v * 100
                    for v in top10.values()
                ],
            }
        )

        fig2, ax2 = plt.subplots(
            figsize=(10, 5)
        )

        bar_colors = [
            "#1f4e79"
            if c == category
            else "#aed6f1"
            for c in df_probs["Category"]
        ]

        bars = ax2.barh(
            df_probs["Category"],
            df_probs["Probability (%)"],
            color=bar_colors,
        )

        ax2.set_xlabel(
            "Probability (%)"
        )

        ax2.set_title(
            "Top 10 Predicted Categories",
            fontweight="bold",
        )

        ax2.invert_yaxis()

        for bar in bars:

            width = bar.get_width()

            ax2.text(
                width + 0.3,
                bar.get_y()
                + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                fontsize=9,
            )

        legend_handles = [
            mpatches.Patch(
                color="#1f4e79",
                label="Top Prediction",
            ),
            mpatches.Patch(
                color="#aed6f1",
                label="Other Categories",
            ),
        ]

        ax2.legend(
            handles=legend_handles,
            loc="lower right",
        )

        plt.tight_layout()

        st.pyplot(fig2)

        plt.close(fig2)

        # ==========================================
        # FULL PROBABILITY TABLE
        # ==========================================
        with st.expander(
            "📋 View All Category Probabilities"
        ):

            df_all = pd.DataFrame(
                {
                    "Rank": range(
                        1,
                        len(all_probs) + 1,
                    ),
                    "Category": list(
                        all_probs.keys()
                    ),
                    "Probability (%)": [
                        f"{v*100:.2f}%"
                        for v in all_probs.values()
                    ],
                }
            ).set_index("Rank")

            st.dataframe(
                df_all,
                use_container_width=True,
            )

        # ==========================================
        # RESUME STATISTICS
        # ==========================================
        st.markdown(
            "### 📝 Resume Statistics"
        )

        word_count = len(
            resume_input.split()
        )

        char_count = len(
            resume_input
        )

        unique_word_count = len(
            set(
                resume_input.lower().split()
            )
        )

        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:

            st.metric(
                "📝 Word Count",
                word_count,
            )

        with col_s2:

            st.metric(
                "🔠 Character Count",
                char_count,
            )

        with col_s3:

            st.metric(
                "🧠 Unique Words",
                unique_word_count,
            )

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")

st.caption(
    "Resume Classifier • Built with Streamlit & scikit-learn • Internship Project"
)
