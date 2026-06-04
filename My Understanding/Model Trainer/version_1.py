import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Version 1: Raw & Hardcoded (Procedural)
# Key steps and file locations are marked below with minimal comments.

# File Location: artifacts/<timestamp>/DataTransformationArtifacts/TransformedTrain/
TRAIN_PATH = os.path.join(
    "artifacts",
    "06_04_2026_11_25_42",
    "DataTransformationArtifacts",
    "TransformedTrain",
    "transformed_train_data.npz",
)
# File Location: artifacts/<timestamp>/DataTransformationArtifacts/TransformedTest/
TEST_PATH = os.path.join(
    "artifacts",
    "06_04_2026_11_25_42",
    "DataTransformationArtifacts",
    "TransformedTest",
    "transformed_test_data.npz",
)
# File Location: artifacts/ModelTrainer/
MODEL_OUTPUT_PATH = os.path.join("artifacts", "ModelTrainer", "model_v1.pkl")

# Hardcoded model and settings
TARGET_COLUMN = "shipment_cost"  # hardcoded target column name
RANDOM_STATE = 42
N_ESTIMATORS = 50


def load_npz_data(path):
    data = np.load(path, allow_pickle=True)
    # Try common key names, fall back to arr_0/arr_1
    if "features" in data and "target" in data:
        X = data["features"]
        y = data["target"]
    elif "X" in data and "y" in data:
        X = data["X"]
        y = data["y"]
    else:
        # common np.savez creates arr_0, arr_1
        X = data["arr_0"]
        y = data["arr_1"]
    return X, y


def train_and_evaluate():
    X_train, y_train = load_npz_data(TRAIN_PATH)
    X_test, y_test = load_npz_data(TEST_PATH)

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)

    print("Version 1 — Hardcoded procedural trainer")
    print("Train shape:", getattr(X_train, "shape", None))
    print("Test shape:", getattr(X_test, "shape", None))
    print("RMSE:", rmse)
    print("Saved model to", MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    train_and_evaluate()
