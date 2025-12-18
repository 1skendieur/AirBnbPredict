import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Абсолютный путь к корню проекта
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

FEATURES = [
    "latitude","longitude","room_type","neighbourhood",
    "minimum_nights","number_of_reviews","reviews_per_month",
    "calculated_host_listings_count","availability_365"
]

def load_model(model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)

def predict_price_one(input_dict: dict, model=None) -> float:
    if model is None:
        model = load_model()

    missing = [c for c in FEATURES if c not in input_dict]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    X = pd.DataFrame([input_dict])[FEATURES]
    log_pred = float(model.predict(X)[0])
    price = float(np.expm1(log_pred))
    return max(0.0, price)
