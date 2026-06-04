import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ==========================================
# GLOBAL CONFIGURATION VARIABLES
# ==========================================
# All paths, thresholds, parameters, and credentials are extracted here instead of being hardcoded in loops/functions.

TRAIN_FILE_PATH = os.path.join('artifacts', '06_04_2026_11_25_42', 'DataTransformationArtifacts', 'TransformedTrain', 'transformed_train_data.npz')
TEST_FILE_PATH = os.path.join('artifacts', '06_04_2026_11_25_42', 'DataTransformationArtifacts', 'TransformedTest', 'transformed_test_data.npz')

MODEL_DIR = os.path.join('artifacts', 'ModelTrainer')
MODEL_FILE_NAME = 'model_v2.pkl'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)

TARGET_COLUMN = 'shipment_cost'

MODEL_PARAMS = {
    'n_estimators': 50,
    'random_state': 42
}


# ==========================================
# LOGIC / PROCEDURAL FUNCTIONS
# ==========================================

def load_transformed_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    if 'features' in data and 'target' in data:
        return data['features'], data['target']
    elif 'X' in data and 'y' in data:
        return data['X'], data['y']
    else:
        return data['arr_0'], data['arr_1']


def train_model(X_train, y_train):
    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return rmse


def save_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)


def run_version_2_trainer():
    X_train, y_train = load_transformed_data(TRAIN_FILE_PATH)
    X_test, y_test = load_transformed_data(TEST_FILE_PATH)

    trained_model = train_model(X_train, y_train)

    rmse_score = evaluate_model(trained_model, X_test, y_test)
    
    save_model(trained_model, MODEL_SAVE_PATH)

    print('Version 2 — Procedural with Global Constants')
    print(f'Train Data Shape: X={X_train.shape}, y={y_train.shape}')
    print(f'RMSE Score: {rmse_score}')
    print(f'Model saved at: {MODEL_SAVE_PATH}')


if __name__ == '__main__':
    run_version_2_trainer()
