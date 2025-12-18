import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/processed.csv"     
MODEL_PATH = "models/model.joblib"

FEATURES = [
    "latitude","longitude","room_type","neighbourhood",
    "minimum_nights","number_of_reviews","reviews_per_month",
    "calculated_host_listings_count","availability_365"
]
TARGET = "log_price"

NUM_COLS = [
    "latitude","longitude","minimum_nights","number_of_reviews",
    "reviews_per_month","calculated_host_listings_count","availability_365"
]
CAT_COLS = ["room_type","neighbourhood"]


def eval_real_price(y_true_log, y_pred_log):
    y_true = np.expm1(np.asarray(y_true_log))
    y_pred = np.expm1(np.asarray(y_pred_log))
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUM_COLS),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CAT_COLS),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)

    mae, rmse = eval_real_price(y_val, pred)
    print(f"Validation MAE (£):  {mae:.3f}")
    print(f"Validation RMSE (£): {rmse:.3f}")

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()