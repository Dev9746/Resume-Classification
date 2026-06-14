"""
=============================================================
  predict.py  —  Resume Prediction using Saved Models
  Usage: python predict.py
=============================================================
"""

import re
import joblib
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
MODELS_DIR    = "saved_models"
MODEL_PATH    = f"{MODELS_DIR}/model.pkl"
TFIDF_PATH    = f"{MODELS_DIR}/tfidf.pkl"
ENCODER_PATH  = f"{MODELS_DIR}/label_encoder.pkl"


# ─────────────────────────────────────────────
# TEXT CLEANER  (mirrors train_model.py)
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Clean raw resume text using the same pipeline as training."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\b(\+?\d[\d\s\-().]{7,}\d)\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
def load_artifacts():
    """Load model, TF-IDF vectorizer, and label encoder from disk."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, TFIDF_PATH, ENCODER_PATH]):
        raise FileNotFoundError(
            "Saved model files not found. Please run train_model.py first."
        )
    model   = joblib.load(MODEL_PATH)
    tfidf   = joblib.load(TFIDF_PATH)
    le      = joblib.load(ENCODER_PATH)
    return model, tfidf, le


# ─────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_resume(resume_text: str, model, tfidf, le) -> dict:
    """
    Predict the job category for a given resume text.

    Returns a dict with:
      category    – predicted job role (string)
      confidence  – probability of top prediction (%)
      all_probs   – {category: probability} for all classes
    """
    if not resume_text or not resume_text.strip():
        return {"error": "Empty resume text provided."}

    cleaned    = clean_text(resume_text)
    vectorized = tfidf.transform([cleaned])

    pred_label = model.predict(vectorized)[0]
    pred_proba = model.predict_proba(vectorized)[0]

    category   = le.inverse_transform([pred_label])[0]
    confidence = pred_proba.max() * 100

    # Build sorted probability dict
    all_probs = dict(
        sorted(
            zip(le.classes_, pred_proba),
            key=lambda x: x[1],
            reverse=True
        )
    )

    return {
        "category":   category,
        "confidence": round(confidence, 2),
        "all_probs":  all_probs
    }


# ─────────────────────────────────────────────
# DEMO  —  5 Sample Resumes
# ─────────────────────────────────────────────
DEMO_RESUMES = {
    "Data Scientist": (
        "Experienced data scientist with expertise in Python, machine learning, "
        "deep learning, TensorFlow, scikit-learn, NLP, data analysis, pandas, "
        "numpy, statistical modeling, A/B testing, Tableau, SQL, Big Data, Spark."
    ),
    "Python Developer": (
        "Python developer with 3 years experience in Django, Flask, REST APIs, "
        "PostgreSQL, Docker, Git, CI/CD pipelines, microservices, unit testing, "
        "AWS Lambda, Celery, Redis, SQLAlchemy, FastAPI, pytest."
    ),
    "HR": (
        "Human Resources professional skilled in talent acquisition, employee "
        "relations, payroll processing, performance management, onboarding, "
        "HRIS systems, labor law compliance, compensation benchmarking, training."
    ),
    "Java Developer": (
        "Java developer proficient in Spring Boot, Hibernate, Maven, RESTful "
        "web services, microservices architecture, MySQL, Oracle DB, JUnit, "
        "Apache Kafka, Docker, Kubernetes, CI/CD, Agile SCRUM methodology."
    ),
    "Testing": (
        "QA testing engineer with expertise in Selenium WebDriver, TestNG, "
        "JUnit, manual testing, automation testing, JIRA, bug tracking, "
        "performance testing, JMeter, API testing with Postman, regression testing."
    ),
}


def main():
    print("=" * 60)
    print("  RESUME CLASSIFIER — PREDICTION DEMO")
    print("=" * 60)

    try:
        model, tfidf, le = load_artifacts()
        print("✅ Models loaded successfully.\n")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    for actual, text in DEMO_RESUMES.items():
        result = predict_resume(text, model, tfidf, le)

        print(f"{'─'*55}")
        print(f"  Actual Category    : {actual}")
        print(f"  Predicted Category : {result['category']}")
        print(f"  Confidence Score   : {result['confidence']:.1f}%")
        print(f"  Resume Snippet     : {text[:70]}…")

        # Show top-3 predictions
        top3 = list(result["all_probs"].items())[:3]
        print("  Top-3 Probabilities:")
        for cat, prob in top3:
            bar = "█" * int(prob * 30)
            print(f"    {cat:<30} {prob*100:5.1f}%  {bar}")
        print()


if __name__ == "__main__":
    main()
