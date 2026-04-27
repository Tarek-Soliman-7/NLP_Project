from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router

app = FastAPI(
    title="Sentiment Analyzer API",
    description="Backend API for predicting and storing sentiment analysis results.",
    version="1.0.0"
)

# Setup CORS so the frontend dashboard can communicate seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev, limit this in production to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount our custom routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analyzer API! Go to /docs for the API Sandbox."}
