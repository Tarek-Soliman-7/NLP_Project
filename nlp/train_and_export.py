# -*- coding: utf-8 -*-
"""
Train BOTH NLP sentiment models (matching Work2.ipynb exactly)
and export nb_model.pkl, lr_model.pkl, and vectorizer.pkl to this folder.
At prediction time the app compares confidence and picks the better model.
"""
import os
import re
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ── paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH     = os.path.join(SCRIPT_DIR, "Movies Reviews.csv")
NB_MODEL_OUT = os.path.join(SCRIPT_DIR, "nb_model.pkl")
LR_MODEL_OUT = os.path.join(SCRIPT_DIR, "lr_model.pkl")
VECTOR_OUT   = os.path.join(SCRIPT_DIR, "vectorizer.pkl")

# ── 1. load data ───────────────────────────────────────────────────────
print("[1/6] Loading dataset ...")
data = pd.read_csv(CSV_PATH)
data.dropna(inplace=True)

# ── 2. clean (same as notebook) ────────────────────────────────────────
print("[2/6] Cleaning text ...")
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

data["text"] = data["text"].apply(clean)

# map numeric labels  0 → "negative"  /  1 → "positive"
# (model.py expects string class names)
data["label"] = data["label"].map({0: "negative", 1: "positive"})

# ── 3. split ───────────────────────────────────────────────────────────
print("[3/6] Splitting data ...")
X = data["text"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. vectorize (shared by both models) ───────────────────────────────
print("[4/6] Vectorising with TF-IDF ...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                             stop_words='english', min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── 5a. train Naive Bayes ──────────────────────────────────────────────
print("[5/6] Training MultinomialNB ...")
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_vec, y_train)
nb_acc = accuracy_score(y_test, nb_model.predict(X_test_vec))
print(f"    [OK] Naive Bayes Accuracy: {nb_acc * 100:.2f}%")

# ── 5b. train Logistic Regression ──────────────────────────────────────
print("       Training LogisticRegression ...")
lr_model = LogisticRegression(max_iter=2000, C=10, solver='liblinear')
lr_model.fit(X_train_vec, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test_vec))
print(f"    [OK] Logistic Regression Accuracy: {lr_acc * 100:.2f}%")

# ── 6. save all artefacts ─────────────────────────────────────────────
print("[6/6] Saving model files ...")
joblib.dump(nb_model,   NB_MODEL_OUT)
joblib.dump(lr_model,   LR_MODEL_OUT)
joblib.dump(vectorizer, VECTOR_OUT)
print(f"    [OK] Saved -> {NB_MODEL_OUT}")
print(f"    [OK] Saved -> {LR_MODEL_OUT}")
print(f"    [OK] Saved -> {VECTOR_OUT}")
print("\n[DONE] You can now start the FastAPI server.")
