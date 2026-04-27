import joblib
import os

VECTORIZER_PATH = "nlp/vectorizer.pkl"
LR_MODEL_PATH = "nlp/lr_model.pkl"

vectorizer = joblib.load(VECTORIZER_PATH)
lr_model = joblib.load(LR_MODEL_PATH)

feature_names = vectorizer.get_feature_names_out()
word = "hate"

if word in feature_names:
    idx = list(feature_names).index(word)
    weight = lr_model.coef_[0][idx]
    print(f"Weight for '{word}': {weight}")
    # In LR, positive weight usually means positive class if classes are [0, 1]
    print(f"Classes: {lr_model.classes_}")
else:
    print(f"'{word}' not in vectorizer.")
