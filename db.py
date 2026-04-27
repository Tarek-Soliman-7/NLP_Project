import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Initialize Firebase App
# We look for a service account file path in the environment,
# or default to "serviceAccountKey.json" in the backend folder.
SERVICE_ACCOUNT_FILE = os.getenv("FIREBASE_SERVICE_ACCOUNT", "serviceAccountKey.json")

# Ensure the file exists so we don't crash the server unnecessarily if this is just testing,
# but print a warning.
db_client = None

if os.path.exists(SERVICE_ACCOUNT_FILE):
    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    firebase_admin.initialize_app(cred)
    db_client = firestore.client()
    print("[SUCCESS] Firebase initialized successfully.")
else:
    print(f"[WARNING] '{SERVICE_ACCOUNT_FILE}' not found. Firebase will NOT be connected! Make sure to download it from your Firebase project settings.")

def save_prediction(original_text: str, cleaned_text: str, sentiment: str, pos_pct: float, neg_pct: float,
                     chosen_model: str = "", nb_confidence: float = 0.0, lr_confidence: float = 0.0) -> str:
    """
    Saves the full sentiment prediction result to Firestore.
    """
    if db_client is None:
        return "fake_id_firebase_not_connected"
    
    doc_ref = db_client.collection("predictions").document()
    doc_ref.set({
        "original_text": original_text,
        "cleaned_text": cleaned_text,
        "sentiment": sentiment,
        "positive_percentage": pos_pct,
        "negative_percentage": neg_pct,
        "chosen_model": chosen_model,
        "nb_confidence": nb_confidence,
        "lr_confidence": lr_confidence,
        "timestamp": firestore.SERVER_TIMESTAMP
    })
    
    return doc_ref.id

def get_sentiment_stats() -> dict:
    """
    Retrieves summary stats of sentiments from Firestore for the Dashboard team.
    """
    if db_client is None:
        return {"positive": 0, "negative": 0, "neutral": 0, "error": "Firebase not connected"}
    
    stats = {}
    try:
        total = 0
        counts = {}
        for sentiment in ["positive", "negative"]:
            query = db_client.collection("predictions").where("sentiment", "==", sentiment)
            results = query.get()
            count = len(results)
            counts[sentiment] = count
            total += count
            
        stats["counts"] = counts
        stats["total_predictions"] = total
        
        # Calculate percentages
        stats["percentages"] = {
            "positive": f"{round((counts.get('positive', 0) / total) * 100, 1)}%" if total > 0 else "0%",
            "negative": f"{round((counts.get('negative', 0) / total) * 100, 1)}%" if total > 0 else "0%"
        }
            
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return {"error": "Failed to retrieve stats"}
        
    return stats
