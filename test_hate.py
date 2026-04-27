from nlp.model import predict_sentiment, nb_model, lr_model
from nlp.preprocess import clean_text

print(f"NB Classes: {nb_model.classes_}")
print(f"LR Classes: {lr_model.classes_}")

test_text = "i hate this man"
cleaned = clean_text(test_text)
prediction = predict_sentiment(cleaned)

print(f"Original: {test_text}")
print(f"Cleaned: {cleaned}")
print(f"Sentiment: {prediction['sentiment']}")
print(f"Chosen Model: {prediction['chosen_model']}")
print(f"NB Confidence: {prediction['nb_confidence']}%")
print(f"LR Confidence: {prediction['lr_confidence']}%")
print(f"Positive: {prediction['positive_percentage']}%  |  Negative: {prediction['negative_percentage']}%")
