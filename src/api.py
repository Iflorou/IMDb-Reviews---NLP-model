"""FastAPI application for IMDb sentiment predictions."""

from contextlib import asynccontextmanager
"""
contextlib is a built-in Python library for managing resources that need setup and cleanup.

Think of things like:

opening a database connection
loading an ML model
opening a file
starting a server """

from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException  # Returns proper HTTP errors
from pydantic import BaseModel, Field

"""
FastAPI()
│      Creates the web server
│
├── BaseModel
│      Defines request/response objects
│
├── Field
│      Validation + examples


"""


MODEL_PATH = Path("saved_models/imdb_dense_v2.keras")

model: tf.keras.Model | None = None


@asynccontextmanager  # Runs startup/shutdown code
async def lifespan(app: FastAPI):
    """Load the TensorFlow model once when the API starts."""
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file was not found at: {MODEL_PATH.resolve()}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    yield #The yield separates the two phases.

    model = None


app = FastAPI(
    title="IMDb Sentiment Analysis API",
    description=(
        "Classifies IMDb-style movie reviews as positive or negative "
        "using a TensorFlow NLP model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class ReviewRequest(BaseModel):
    review: str = Field(
        min_length=1,
        examples=["The movie was excellent and I loved the performances."]
    )


class PredictionResponse(BaseModel):
    sentiment: str
    predicted_class: int
    probability_positive: float
    confidence: float


@app.get("/")
def root():
    return {
        "message": "IMDb Sentiment Analysis API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The model is not loaded."
        )

    review_tensor = tf.constant([request.review])

    prediction = model.predict(
        review_tensor,
        verbose=0
    )

    probability_positive = float(np.squeeze(prediction))
    predicted_class = int(probability_positive >= 0.5)

    if predicted_class == 1:
        sentiment = "positive"
        confidence = probability_positive
    else:
        sentiment = "negative"
        confidence = 1.0 - probability_positive

    return PredictionResponse(
        sentiment=sentiment,
        predicted_class=predicted_class,
        probability_positive=round(probability_positive, 4),
        confidence=round(confidence, 4),
    )