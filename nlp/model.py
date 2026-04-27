import os
import joblib
import numpy as np

# Traditional Machine Learning Algorithms (as described in the project report)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

NB_MODEL_PATH = os.path.join(os.path.dirname(__file__), "nb_model.pkl")
LR_MODEL_PATH = os.path.join(os.path.dirname(__file__), "lr_model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

# Initialize models globally so they are only loaded once when the server starts
nb_model = None
lr_model = None
vectorizer = None

if (os.path.exists(NB_MODEL_PATH)
    and os.path.exists(LR_MODEL_PATH)
    and os.path.exists(VECTORIZER_PATH)):
    nb_model   = joblib.load(NB_MODEL_PATH)
    lr_model   = joblib.load(LR_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("[OK] Loaded both NB and LR models.")
else:
    print("[WARNING] Real ML model files not found. Using initialised fallback models.")
    # Initialize a basic vectorizer and both models to emulate the real logic
    vectorizer = TfidfVectorizer(max_features=1000)
    dummy_texts = ["good great amazing excellent love", "bad terrible awful hate worst"]
    dummy_labels = ["positive", "negative"]
    X_train = vectorizer.fit_transform(dummy_texts)

    nb_model = MultinomialNB()
    nb_model.fit(X_train, dummy_labels)

    lr_model = LogisticRegression()
    lr_model.fit(X_train, dummy_labels)


def predict_sentiment(text: str) -> dict:
    """
    Runs the input text through BOTH Naive Bayes and Logistic Regression,
    compares their confidence, and returns the prediction from the more
    confident model — exactly matching the Work2.ipynb 'Choosing best model'
    logic.
    """
    if not text.strip():
        return {
            "sentiment": "negative",
            "positive_percentage": 0.0,
            "negative_percentage": 100.0,
            "chosen_model": "none",
            "nb_confidence": 0.0,
            "lr_confidence": 0.0,
        }

    # Vectorize text
    X_input = vectorizer.transform([text])

    # ── Naive Bayes prediction ─────────────────────────────────────────
    nb_proba   = nb_model.predict_proba(X_input)[0]
    nb_classes = nb_model.classes_
    nb_conf    = float(max(nb_proba))

    # ── Logistic Regression prediction ─────────────────────────────────
    lr_proba   = lr_model.predict_proba(X_input)[0]
    lr_classes = lr_model.classes_
    lr_conf    = float(max(lr_proba))

    # ── Choose the model with higher confidence ────────────────────────
    if lr_conf > nb_conf:
        chosen_proba   = lr_proba
        chosen_classes = lr_classes
        chosen_name    = "Logistic Regression"
    else:
        chosen_proba   = nb_proba
        chosen_classes = nb_classes
        chosen_name    = "Naive Bayes"

    # Map probabilities to our two categories
    neg_idx = list(chosen_classes).index("negative")
    pos_idx = list(chosen_classes).index("positive")

    neg_pct = round(float(chosen_proba[neg_idx]) * 100, 1)
    pos_pct = round(float(chosen_proba[pos_idx]) * 100, 1)

    sentiment = "positive" if pos_pct >= neg_pct else "negative"

    return {
        "sentiment": sentiment,
        "positive_percentage": pos_pct,
        "negative_percentage": neg_pct,
        "chosen_model": chosen_name,
        "nb_confidence": round(nb_conf * 100, 1),
        "lr_confidence": round(lr_conf * 100, 1),
    }
