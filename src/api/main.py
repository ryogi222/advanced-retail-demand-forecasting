from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.predict import load_model, predict_quantiles


app = FastAPI(
    title="Advanced Retail Demand Forecasting API",
    description="Production API for probabilistic retail demand forecasting.",
    version="1.0.0",
)


# Load production model once when API starts
model = load_model()


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