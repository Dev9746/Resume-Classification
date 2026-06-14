"""
=============================================================
  Resume Classifier - Professional Version
  Part 1: Setup + UI + Model Loading
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


# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown("""
<style>

.main-header {
    font-size: 2.6rem;
    font-weight: 700;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 0;
}

.sub-header {
    font-size: 1.1rem;
    color: #555;
    text-align: center;
    margin-bottom: 30px;
}

.stTextArea textarea {
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# HEADER
# ==========================================================
st.markdown(
    '<p class="main-header">📄 Resume Classifier</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-header">'
    'Paste your resume below and let AI predict the best-fit job category.'
    '</p>',
    unsafe_allow_html=True
)


# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/resume.png",
        width=70
    )

    st.markdown("## 📘 About This App")

    st.info(
        "This app classifies resumes into job categories "
        "using NLP and Machine Learning."
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

    for cat in categories:
        st.markdown(f"• {cat}")

    st.markdown("---")

    st.markdown(
        """
        **Model:** Logistic Regression / Naive Bayes
        
        **Features:** TF-IDF
        
        **Dataset:** Kaggle Resume Dataset
        """
    )


# ==========================================================
# MODEL PATHS
# ==========================================================
MODEL_PATH = "model.pkl"
TFIDF_PATH = "tfidf.pkl"
ENCODER_PATH = "label_encoder.pkl"


# ==========================================================
# CHECK FILES EXIST
# ==========================================================
if not (
    os.path.exists(MODEL_PATH)
    and os.path.exists(TFIDF_PATH)
    and os.path.exists(ENCODER_PATH)
):
    st.error(
        "❌ Model files not found.\n\n"
        f"Current files: {os.listdir('.')}"
    )
    st.stop()


# ==========================================================
# LOAD MODELS (CACHED)
# ==========================================================
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


# ==========================================================
# PDF RESUME UPLOAD
# ==========================================================
uploaded_file = st.file_uploader(
    "📄 Upload Resume PDF (Optional)",
    type=["pdf"]
)

resume_input = ""

if uploaded_file:

    try:

        reader = PdfReader(uploaded_file)

        pdf_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pdf_text += text + "\n"

        resume_input = pdf_text

        st.success("✅ PDF uploaded successfully.")

    except Exception as e:

        st.error(f"Unable to read PDF: {e}")


# ==========================================================
# TEXT CLEANING
# ==========================================================
def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    text = re.sub(
        r"\b(\+?\d[\d\s\-().]{7,}\d)\b",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    tokens = [
        word
        for word in text.split()
        if word not in ENGLISH_STOP_WORDS
        and len(word) > 2
    ]

    return " ".join(tokens)

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================
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

    return category, confidence, all_probs


# ==========================================================
# RESUME INPUT
# ==========================================================
resume_input = st.text_area(
    "📝 Paste Resume Text Here",
    height=300,
    value=resume_input,
    placeholder=(
        "Example:\n\n"
        "Skills: Python, Machine Learning, SQL, NLP...\n"
        "Experience: Data Scientist at XYZ Company...\n"
        "Education: B.Tech Computer Science...\n"
    )
)


# ==========================================================
# CENTERED BUTTON
# ==========================================================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    predict_btn = st.button(
        "🔍 Predict Job Role",
        use_container_width=True,
        type="primary"
    )


# ==========================================================
# PREDICTION RESULT
# ==========================================================
if predict_btn:

    if not resume_input.strip():

        st.warning(
            "⚠️ Please paste some resume text."
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

        col_a, col_b = st.columns([2, 1])

        with col_a:

            st.info("🎯 Predicted Job Category")

            st.success(f"💼 {category}")

            st.metric(
                label="Confidence",
                value=f"{confidence:.1f}%"
            )

       

        # ==================================================
        # PREDICTION CARD
        # ==================================================
      col_a, col_b = st.columns([2, 1])

        with col_a:

    st.info("🎯 Predicted Job Category")

    st.success(f"💼 {category}")

    st.metric(
        label="Confidence",
        value=f"{confidence:.1f}%"
    )

        with col_b:

    fig, ax = plt.subplots(figsize=(3, 3))

    gauge_color = (
        "#2ecc71"
        if confidence >= 70
        else "#f39c12"
        if confidence >= 40
        else "#e74c3c"
    )

    ax.pie(
        [confidence, 100 - confidence],
        colors=[gauge_color, "#ecf0f1"],
        startangle=90,
        counterclock=False
    )

    ax.add_patch(
        plt.Circle((0, 0), 0.65, color="white")
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

    ax.set_title("Confidence")
    ax.axis("equal")

    st.pyplot(fig)

    plt.close(fig)



   
   


        # ==================================================
        # TOP 10 CATEGORY PROBABILITIES
        # ==================================================
        st.markdown("### 📊 Top 10 Category Probabilities")

        top10 = dict(
            list(all_probs.items())[:10]
        )

        df_probs = pd.DataFrame({
            "Category": list(top10.keys()),
            "Probability (%)": [
                prob * 100
                for prob in top10.values()
            ]
        })


        # ==================================================
        # BAR CHART
        # ==================================================
        fig2, ax2 = plt.subplots(
            figsize=(10, 5)
        )

        bar_colors = [
            "#1f4e79"
            if category_name == category
            else "#aed6f1"
            for category_name in df_probs["Category"]
        ]

        bars = ax2.barh(
            df_probs["Category"],
            df_probs["Probability (%)"],
            color=bar_colors
        )

        ax2.set_xlabel(
            "Probability (%)"
        )

        ax2.set_title(
            "Top 10 Predicted Categories",
            fontweight="bold"
        )

        ax2.invert_yaxis()


        # Add probability labels
        for bar in bars:

            width = bar.get_width()

            ax2.text(
                width + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                fontsize=9
            )


        # ==================================================
        # LEGEND
        # ==================================================
        legend_handles = [

            mpatches.Patch(
                color="#1f4e79",
                label="Top Prediction"
            ),

            mpatches.Patch(
                color="#aed6f1",
                label="Other Categories"
            ),
        ]

        ax2.legend(
            handles=legend_handles,
            loc="lower right"
        )

        plt.tight_layout()

        st.pyplot(fig2)

        plt.close(fig2)


        # ==================================================
        # FULL PROBABILITY TABLE
        # ==================================================
        with st.expander(
            "📋 View All Category Probabilities"
        ):

            df_all = pd.DataFrame({

                "Rank":
                    range(
                        1,
                        len(all_probs) + 1
                    ),

                "Category":
                    list(all_probs.keys()),

                "Probability (%)":
                    [
                        f"{prob*100:.2f}%"
                        for prob in all_probs.values()
                    ]
            })

            df_all = df_all.set_index(
                "Rank"
            )

            st.dataframe(
                df_all,
                use_container_width=True
            )


        # ==================================================
        # RESUME STATISTICS
        # ==================================================
        st.markdown(
            "### 📝 Resume Statistics"
        )

        word_count = len(
            resume_input.split()
        )

        character_count = len(
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
                word_count
            )

        with col_s2:

            st.metric(
                "🔤 Character Count",
                character_count
            )

        with col_s3:

            st.metric(
                "🧠 Unique Words",
                unique_word_count
            )
# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")

st.caption(
    "Resume Classifier • Built with Streamlit & scikit-learn • Internship Project"
)
