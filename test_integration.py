from nlp.model import predict_sentiment
from nlp.preprocess import clean_text
import os

test_text = "I love this movie, it was fantastic!"
cleaned = clean_text(test_text)
prediction = predict_sentiment(cleaned)

print(f"Original: {test_text}")
print(f"Cleaned: {cleaned}")
print(f"Sentiment: {prediction['sentiment']}")
print(f"Chosen Model: {prediction['chosen_model']}")
print(f"NB Confidence: {prediction['nb_confidence']}%")
print(f"LR Confidence: {prediction['lr_confidence']}%")
print(f"Positive: {prediction['positive_percentage']}%  |  Negative: {prediction['negative_percentage']}%")

from db import save_prediction
try:
    doc_id = save_prediction(test_text, cleaned, prediction['sentiment'], prediction['positive_percentage'], prediction['negative_percentage'])
    print(f"DB Save ID: {doc_id}")
except Exception as e:
    print(f"DB Save Error: {e}")
