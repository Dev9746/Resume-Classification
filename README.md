# 📄 Resume Classification Using NLP

> **End-to-end Machine Learning pipeline** that classifies resumes into job roles using Natural Language Processing (NLP) and scikit-learn — built as a professional internship project.

---

## 🎯 Project Overview

Given a raw resume in plain text, this system predicts the **most likely job category** (e.g., *Data Scientist*, *Java Developer*, *HR*) along with a confidence score. The pipeline covers everything from raw text cleaning to a deployable Streamlit web app.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📥 Data ingestion | Auto-loads `Resume.csv` (962 resumes × 25 categories) |
| 🧹 Text cleaning | URL/email/phone removal, lowercasing, stopword filtering |
| 🔢 Vectorisation | TF-IDF (15 000 features, unigrams + bigrams) |
| 🤖 Models | Logistic Regression & Multinomial Naive Bayes |
| 📊 Evaluation | Accuracy, Precision, Recall, F1, CV, Confusion Matrix |
| 🔧 Tuning | GridSearchCV hyperparameter optimisation |
| 💾 Persistence | Saved artifacts via Joblib (`.pkl`) |
| 🌐 Web UI | Streamlit app with probability chart & confidence meter |

---

## 📂 Project Structure

```
Resume-Classification/
│
├── app.py                  ← Streamlit web application
├── train_model.py          ← Full training pipeline (run this first)
├── predict.py              ← Standalone prediction script / demo
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── Resume.csv              ← Dataset (Kaggle UpdatedResumeDataSet)
│
├── saved_models/           ← Auto-created after training
│   ├── model.pkl           ← Best trained classifier
│   ├── tfidf.pkl           ← Fitted TF-IDF vectorizer
│   └── label_encoder.pkl   ← Fitted LabelEncoder
│
├── notebooks/
│   └── Resume_Classification.ipynb   ← Beginner-friendly walkthrough
│
└── screenshots/            ← Auto-saved charts from training
    ├── class_distribution.png
    ├── confusion_matrix_logistic_regression.png
    └── confusion_matrix_multinomial_naive_bayes.png
```

---

## 📦 Dataset

- **Name:** UpdatedResumeDataSet.csv
- **Source:** [Kaggle — Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
- **Size:** 962 resumes
- **Columns:** `Category` (job role label), `Resume` (raw text)
- **Categories (25):**

```
Data Science · Python Developer · Java Developer · Testing · HR
DevOps Engineer · Web Designing · Blockchain · ETL Developer · Hadoop
Sales · Mechanical Engineer · Civil Engineer · SAP Developer
Network Security Engineer · Database · Business Analyst
DotNet Developer · Automation Testing · PMO · Advocate · Arts
Health and fitness · Electrical Engineering · Operations Manager
```

---

## ⚙️ Installation

### 1. Clone / Download the project
```bash
git clone <your-repo-url>
cd Resume-Classification
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Train the model
```bash
python train_model.py
```
This will:
- Load and explore `Resume.csv`
- Clean and vectorize all resumes
- Train Logistic Regression & Naive Bayes
- Run GridSearchCV hyperparameter tuning
- Save the best model, TF-IDF, and LabelEncoder to `saved_models/`
- Print a full evaluation report
- Save confusion matrix charts to `screenshots/`

**Expected output (approximate):**
```
Logistic Regression Accuracy : 0.98+
Naive Bayes Accuracy         : 0.96+
Best Model saved             : saved_models/model.pkl
```

### Step 2 — Test predictions (optional)
```bash
python predict.py
```
Runs 5 sample resumes and prints predicted category + confidence.

### Step 3 — Launch the Streamlit web app
```bash
streamlit run app.py
```
Opens at **http://localhost:8501** — paste any resume text and click **Predict Job Role**.

### Step 4 — Open the Jupyter Notebook (optional walkthrough)
```bash
jupyter notebook notebooks/Resume_Classification.ipynb
```

---

## 📊 Model Performance (Expected)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | ~0.98 | ~0.98 | ~0.98 | ~0.98 |
| Multinomial Naive Bayes | ~0.96 | ~0.96 | ~0.96 | ~0.96 |
| LR (Tuned) | ~0.99 | ~0.99 | ~0.99 | ~0.99 |

> Logistic Regression consistently outperforms Naive Bayes on this dataset because resume text patterns are linearly separable in TF-IDF space.

---

## 🖥️ Streamlit App — Example Output

```
Input  : "Python developer with Django, Flask, REST APIs, PostgreSQL..."
Output : 💼 Python Developer  |  Confidence: 97.3%
```

The app also shows:
- Confidence meter (pie chart)
- Top-10 category probability bar chart
- Full probability table (expandable)
- Resume word/character count stats

---

## 🔮 Future Enhancements

- [ ] Add BERT / sentence-transformers for richer embeddings
- [ ] Support PDF resume upload in Streamlit
- [ ] Add skill extraction using Named Entity Recognition (NER)
- [ ] Deploy to Streamlit Cloud / Hugging Face Spaces
- [ ] Add experience level prediction (Junior / Mid / Senior)
- [ ] Multi-label classification (a resume can fit multiple roles)
- [ ] Add resume scoring / feedback feature

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML & NLP | scikit-learn, TF-IDF |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Model persistence | joblib |
| Web UI | Streamlit |
| Notebook | Jupyter |

---

## 👨‍💻 Author

**Internship Project — Resume Classification Using NLP**
Built as an end-to-end ML internship submission demonstrating the full model development lifecycle.

---

## 📄 License

This project is for educational purposes. The dataset is from Kaggle under its respective license.
