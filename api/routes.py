from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from nlp.preprocess import clean_text
from nlp.model import predict_sentiment
from db import save_prediction, get_sentiment_stats

router = APIRouter()

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    original_text: str
    cleaned_text: str
    sentiment: str
    positive_percentage: float
    negative_percentage: float
    chosen_model: str
    

@router.post("/predict", response_model=PredictionResponse)
def handle_prediction(req: PredictionRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    # 1. Ask Preprocessing Team's code to clean the text
    normalized_text = clean_text(req.text)
    
    # 2. Ask Model Team's code to run sentiment prediction
    prediction_result = predict_sentiment(normalized_text)
    sentiment    = prediction_result.get("sentiment")
    pos_pct      = prediction_result.get("positive_percentage")
    neg_pct      = prediction_result.get("negative_percentage")
    chosen_model = prediction_result.get("chosen_model")
    nb_conf      = prediction_result.get("nb_confidence")
    lr_conf      = prediction_result.get("lr_confidence")
    
    # 3. Store the interaction in our Firebase Database
    doc_id = save_prediction(
        original_text=req.text,
        cleaned_text=normalized_text,
        sentiment=sentiment,
        pos_pct=pos_pct,
        neg_pct=neg_pct
    )
    
    # 4. Respond to the Frontend App
    return PredictionResponse(
        original_text=req.text,
        cleaned_text=normalized_text,
        sentiment=sentiment,
        positive_percentage=pos_pct,
        negative_percentage=neg_pct,
        chosen_model=chosen_model,
        nb_confidence=nb_conf,
        lr_confidence=lr_conf
    )

@router.get("/stats")
def handle_dashboard_stats():
    """
    Endpoint for Dashboard Team to draw their charts.
    """
    stats = get_sentiment_stats()
    return stats
