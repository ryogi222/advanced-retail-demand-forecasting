from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.predict import load_model, predict_quantiles
from src.inference.future_features import (
    build_next_day_features,
    add_calendar_features,
)

app = FastAPI(
    title="Advanced Retail Demand Forecasting API",
    description="Production API for probabilistic retail demand forecasting.",
    version="1.0.0",
)


# Load production model once when API starts
model = load_model()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_daily_features_calendar.csv"
)

CALENDAR_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "m5"
    / "calendar.csv"
)
class PredictionRequest(BaseModel):
    day_of_week: int
    day_of_month: int
    month: int
    year: int
    week_of_year: int
    is_weekend: int

    lag_1: float
    lag_7: float
    lag_14: float
    lag_28: float

    rolling_mean_7: float
    rolling_mean_28: float
    rolling_std_7: float
    rolling_std_28: float

    snap_CA: int

    has_event_1: int
    has_event_2: int
    has_any_event: int

    event_type_cultural: int
    event_type_national: int
    event_type_religious: int
    event_type_sporting: int

    is_superbowl: int
    is_valentines: int
    is_christmas: int


@app.get("/")
def root():
    return {
        "message": "Advanced Retail Demand Forecasting API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__,
        "trees": len(model.estimators_),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    features = request.model_dump()

    prediction = predict_quantiles(
        model,
        features,
    )

    return {
        "p10": float(prediction["p10"][0]),
        "p50": float(prediction["p50"][0]),
        "p90": float(prediction["p90"][0]),
        "mean": float(prediction["mean"][0]),
    }
@app.get("/forecast/next-day")
def forecast_next_day():
    """
    Generate the first unseen next-day demand forecast automatically.
    """

    history = pd.read_csv(HISTORY_PATH)
    calendar = pd.read_csv(CALENDAR_PATH)

    features = build_next_day_features(history)
    features = add_calendar_features(features, calendar)

    prediction = predict_quantiles(
        model,
        features,
    )

    return {
        "forecast_date": pd.to_datetime(features["date"]).strftime("%Y-%m-%d"),
        "p10": float(prediction["p10"][0]),
        "p50": float(prediction["p50"][0]),
        "p90": float(prediction["p90"][0]),
        "mean": float(prediction["mean"][0]),
    }