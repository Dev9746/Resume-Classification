"""
=============================================================
  Resume Classification Using NLP - Training Pipeline
  Author  : Internship Project
  Dataset : UpdatedResumeDataSet.csv (Kaggle)
  Models  : Logistic Regression | Multinomial Naive Bayes
=============================================================
"""

# ─────────────────────────────────────────────
# 1.  IMPORTS
# ─────────────────────────────────────────────
import re
import string
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 2.  CONSTANTS
# ─────────────────────────────────────────────
DATA_PATH   = "Resume.csv"
MODELS_DIR  = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# 3.  LOAD & EXPLORE DATASET
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset and print exploratory statistics."""
    print("=" * 60)
    print("  STEP 1 — LOAD & EXPLORE DATASET")
    print("=" * 60)

    df = pd.read_csv(path)

    print(f"\n📂 Dataset loaded from: {path}")
    print(f"   Shape  : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Columns: {df.columns.tolist()}")

    print("\n📊 Class Distribution (Category counts):")
    print(df["Category"].value_counts().to_string())

    print(f"\n❓ Missing Values:\n{df.isnull().sum().to_string()}")

    print("\n📝 Sample Record (first resume snippet):")
    print(df["Resume"].iloc[0][:300], "...")

    # ── Plot class distribution
    plt.figure(figsize=(14, 6))
    order = df["Category"].value_counts().index
    ax = sns.countplot(
        data=df, y="Category", order=order, palette="viridis"
    )
    ax.set_title("Resume Category Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_ylabel("Category")
    plt.tight_layout()
    plt.savefig("screenshots/class_distribution.png", dpi=120)
    plt.close()
    print("\n📈 Class distribution chart saved → screenshots/class_distribution.png")

    return df


# ─────────────────────────────────────────────
# 4.  TEXT PREPROCESSING
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Cleans raw resume text through 8 sequential steps:
      1. Lowercase conversion
      2. URL removal
      3. Email address removal
      4. Phone number removal
      5. Punctuation & special character removal
      6. Extra whitespace removal
      7. Stopword removal  (sklearn's built-in English stopwords)
      8. Short-token filtering (words < 2 chars)
    Returns a single cleaned string ready for TF-IDF.
    """
    # Step 1 – lowercase
    text = text.lower()

    # Step 2 – remove URLs  (http/https/www)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)

    # Step 3 – remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Step 4 – remove phone numbers  (10-digit, with country code, etc.)
    text = re.sub(r"\b(\+?\d[\d\s\-().]{7,}\d)\b", " ", text)

    # Step 5 – remove punctuation & special characters (keep only a-z and spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 6 – collapse multiple whitespace into one
    text = re.sub(r"\s+", " ", text).strip()

    # Step 7 – remove stopwords using sklearn's English stopword list
    tokens = text.split()
    tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]

    # Step 8 – remove very short tokens (noise)
    tokens = [w for w in tokens if len(w) > 2]

    return " ".join(tokens)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text() to the Resume column and add a cleaned_resume column."""
    print("\n" + "=" * 60)
    print("  STEP 2 — DATA PREPROCESSING")
    print("=" * 60)
    print("\n🧹 Cleaning resume text …")
    df = df.copy()
    df["cleaned_resume"] = df["Resume"].apply(clean_text)
    print("   ✅ Text cleaning complete.")
    print("\n   Before cleaning (snippet):")
    print("  ", df["Resume"].iloc[0][:120])
    print("\n   After cleaning (snippet):")
    print("  ", df["cleaned_resume"].iloc[0][:120])
    return df


# ─────────────────────────────────────────────
# 5.  LABEL ENCODING
# ─────────────────────────────────────────────
def encode_labels(df: pd.DataFrame):
    """
    Encode string category labels into integers.
    LabelEncoder maps each unique category → a number,
    so ML models can work with them numerically.
    """
    print("\n" + "=" * 60)
    print("  STEP 3 — LABEL ENCODING")
    print("=" * 60)

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Category"])

    print(f"\n🏷️  {len(le.classes_)} unique categories found.")
    print("   Mapping (first 5):",
          dict(zip(le.classes_[:5], le.transform(le.classes_[:5]))))

    # Save the encoder
    joblib.dump(le, f"{MODELS_DIR}/label_encoder.pkl")
    print(f"   💾 LabelEncoder saved → {MODELS_DIR}/label_encoder.pkl")

    return df, le


# ─────────────────────────────────────────────
# 6.  TF-IDF FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_tfidf(df: pd.DataFrame):
    """
    Convert cleaned text to a TF-IDF matrix.

    TF-IDF (Term Frequency–Inverse Document Frequency):
    ─────────────────────────────────────────────────────
      TF(t,d)  = (occurrences of term t in doc d) / (total terms in d)
      IDF(t)   = log( N / df(t) )   where N = total docs, df(t) = docs containing t
      TF-IDF   = TF × IDF

    Parameters chosen:
      max_features = 15000  → keep the top 15k most informative words
      ngram_range  = (1,2)  → unigrams + bigrams (e.g. "machine learning")
      min_df       = 2      → ignore terms in fewer than 2 docs (reduce noise)
      max_df       = 0.85   → ignore terms in >85% docs (too common, not useful)
      sublinear_tf = True   → use log(1+tf) to dampen high-frequency terms
    """
    print("\n" + "=" * 60)
    print("  STEP 4 — TF-IDF FEATURE ENGINEERING")
    print("=" * 60)

    tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True
    )

    X = tfidf.fit_transform(df["cleaned_resume"])
    y = df["label"].values

    print(f"\n📐 TF-IDF matrix shape: {X.shape}")
    print(f"   Vocabulary size    : {len(tfidf.vocabulary_)}")
    print(f"   Top 10 features    : {list(tfidf.get_feature_names_out()[:10])}")

    # Save vectorizer
    joblib.dump(tfidf, f"{MODELS_DIR}/tfidf.pkl")
    print(f"   💾 TF-IDF vectorizer saved → {MODELS_DIR}/tfidf.pkl")

    return X, y, tfidf


# ─────────────────────────────────────────────
# 7.  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
def split_data(X, y):
    """
    Stratified 80/20 train-test split.
    Stratified = each class keeps its proportion in both train & test sets,
    which is crucial for imbalanced datasets.
    """
    print("\n" + "=" * 60)
    print("  STEP 5 — TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y          # keeps class balance
    )
    print(f"\n   Train set size : {X_train.shape[0]} samples")
    print(f"   Test  set size : {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 8.  MODEL TRAINING
# ─────────────────────────────────────────────
def train_models(X_train, y_train):
    """Train Logistic Regression and Multinomial Naive Bayes models."""
    print("\n" + "=" * 60)
    print("  STEP 6 — MODEL TRAINING")
    print("=" * 60)

    # ── Logistic Regression
    print("\n🔵 Training Logistic Regression …")
    lr_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=RANDOM_STATE
    )
    lr_model.fit(X_train, y_train)
    print("   ✅ Logistic Regression trained.")

    # ── Multinomial Naive Bayes
    print("\n🟢 Training Multinomial Naive Bayes …")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train, y_train)
    print("   ✅ Multinomial Naive Bayes trained.")

    return lr_model, nb_model


# ─────────────────────────────────────────────
# 9.  MODEL EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(model, model_name, X_train, X_test, y_train, y_test, le):
    """
    Comprehensive evaluation:
    Accuracy, Precision, Recall, F1, Classification Report,
    Confusion Matrix heatmap, and 5-fold Cross-Validation.
    """
    print(f"\n{'─'*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'─'*60}")

    y_pred   = model.predict(X_test)
    acc      = accuracy_score(y_test, y_pred)
    prec     = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec      = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1       = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n   Accuracy  : {acc:.4f}")
    print(f"   Precision : {prec:.4f}")
    print(f"   Recall    : {rec:.4f}")
    print(f"   F1 Score  : {f1:.4f}")

    print(f"\n   Classification Report:\n")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    # ── 5-fold Cross-validation on training data
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"   Cross-Val Scores (5-fold): {cv_scores.round(4)}")
    print(f"   Mean CV Accuracy          : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Confusion Matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(18, 14))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_")
    plt.savefig(f"screenshots/confusion_matrix_{fname}.png", dpi=120)
    plt.close()
    print(f"   📊 Confusion matrix saved → screenshots/confusion_matrix_{fname}.png")

    return {"model": model_name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cv_mean": cv_scores.mean()}


# ─────────────────────────────────────────────
# 10. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
def tune_models(X_train, y_train):
    """GridSearchCV hyperparameter tuning for both models."""
    print("\n" + "=" * 60)
    print("  STEP 7 — HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 60)

    # ── Logistic Regression tuning
    print("\n🔵 Tuning Logistic Regression …")
    lr_params = {
        "C":       [0.01, 0.1, 1, 10],
        "max_iter":[500, 1000]
    }
    lr_grid = GridSearchCV(
        LogisticRegression(solver="lbfgs", random_state=RANDOM_STATE),
        lr_params, cv=3, scoring="accuracy", n_jobs=-1, verbose=0
    )
    lr_grid.fit(X_train, y_train)
    print(f"   Best params: {lr_grid.best_params_}")
    print(f"   Best CV acc : {lr_grid.best_score_:.4f}")

    # ── Naive Bayes tuning
    print("\n🟢 Tuning Multinomial Naive Bayes …")
    nb_params = {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0]}
    nb_grid = GridSearchCV(
        MultinomialNB(), nb_params, cv=3, scoring="accuracy", n_jobs=-1, verbose=0
    )
    nb_grid.fit(X_train, y_train)
    print(f"   Best params: {nb_grid.best_params_}")
    print(f"   Best CV acc : {nb_grid.best_score_:.4f}")

    return lr_grid.best_estimator_, nb_grid.best_estimator_


# ─────────────────────────────────────────────
# 11. PERFORMANCE COMPARISON TABLE
# ─────────────────────────────────────────────
def print_comparison(results: list):
    """Print a formatted performance comparison table."""
    print("\n" + "=" * 60)
    print("  PERFORMANCE COMPARISON TABLE")
    print("=" * 60)
    df_res = pd.DataFrame(results)
    df_res = df_res.set_index("model")
    print(df_res.round(4).to_string())

    best = df_res["accuracy"].idxmax()
    print(f"\n🏆 Best Model by Accuracy: {best} ({df_res.loc[best,'accuracy']:.4f})")


# ─────────────────────────────────────────────
# 12. SAVE BEST MODEL
# ─────────────────────────────────────────────
def save_best_model(lr_model, nb_model, X_test, y_test):
    """Save the model with higher test accuracy as the production model."""
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
    nb_acc = accuracy_score(y_test, nb_model.predict(X_test))

    best_model = lr_model if lr_acc >= nb_acc else nb_model
    best_name  = "Logistic Regression" if lr_acc >= nb_acc else "Naive Bayes"

    joblib.dump(best_model, f"{MODELS_DIR}/model.pkl")
    print(f"\n💾 Best model ({best_name}, acc={max(lr_acc,nb_acc):.4f}) saved → {MODELS_DIR}/model.pkl")
    return best_model


# ─────────────────────────────────────────────
# 13. REAL RESUME TESTING
# ─────────────────────────────────────────────
SAMPLE_RESUMES = {
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


def test_sample_resumes(model, tfidf, le):
    """Run predictions on 5 sample resumes and display results."""
    print("\n" + "=" * 60)
    print("  STEP 9 — REAL RESUME TESTING")
    print("=" * 60)

    for actual_category, resume_text in SAMPLE_RESUMES.items():
        cleaned     = clean_text(resume_text)
        vec         = tfidf.transform([cleaned])
        pred_label  = model.predict(vec)[0]
        pred_proba  = model.predict_proba(vec)[0]
        pred_cat    = le.inverse_transform([pred_label])[0]
        confidence  = pred_proba.max() * 100

        print(f"\n  📄 Actual    : {actual_category}")
        print(f"     Predicted : {pred_cat}")
        print(f"     Confidence: {confidence:.1f}%")
        print(f"     Snippet   : {resume_text[:80]}…")


# ─────────────────────────────────────────────
# 14. MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    os.makedirs("screenshots", exist_ok=True)

    # Load & explore
    df = load_data(DATA_PATH)

    # Preprocess
    df = preprocess_data(df)

    # Encode labels
    df, le = encode_labels(df)

    # TF-IDF
    X, y, tfidf = build_tfidf(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train
    lr_model, nb_model = train_models(X_train, y_train)

    # Evaluate (before tuning)
    print("\n" + "=" * 60)
    print("  STEP 7 — MODEL EVALUATION (Before Tuning)")
    print("=" * 60)
    results = []
    results.append(evaluate_model(lr_model, "Logistic Regression", X_train, X_test, y_train, y_test, le))
    results.append(evaluate_model(nb_model,  "Multinomial Naive Bayes",  X_train, X_test, y_train, y_test, le))

    # Hyper-parameter tuning
    lr_tuned, nb_tuned = tune_models(X_train, y_train)

    print("\n" + "=" * 60)
    print("  STEP 8 — MODEL EVALUATION (After Tuning)")
    print("=" * 60)
    results.append(evaluate_model(lr_tuned, "LR (Tuned)",  X_train, X_test, y_train, y_test, le))
    results.append(evaluate_model(nb_tuned, "NB (Tuned)",  X_train, X_test, y_train, y_test, le))

    # Comparison table
    print_comparison(results)

    # Save best model
    best_model = save_best_model(lr_tuned, nb_tuned, X_test, y_test)

    # Real resume tests
    test_sample_resumes(best_model, tfidf, le)

    print("\n✅ Training pipeline complete!")
    print(f"   All models saved in: {MODELS_DIR}/")
    print(f"   Charts saved in    : screenshots/")


if __name__ == "__main__":
    main()
