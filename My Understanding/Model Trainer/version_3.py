import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ==========================================
# OOP WITH FILE LOCATION MAPPINGS
# ==========================================


# File Location (Production): src/components/model_trainer.py
class ModelTrainer:
    def __init__(self):
        # File Location (Production): src/entity/config_entity.py -> ModelTrainerConfig
        # These are settings and destination paths configured by the user/system.
        self.model_dir = os.path.join("artifacts", "ModelTrainer")
        self.model_save_path = os.path.join(self.model_dir, "model_v3.pkl")
        self.n_estimators = 50
        self.random_state = 42

        # File Location (Production): src/entity/artifact_entity.py -> DataTransformationArtifact
        # These reflect outputs from previous components injected into the trainer.
        self.train_file_path = os.path.join(
            "artifacts",
            "06_04_2026_11_25_42",
            "DataTransformationArtifacts",
            "TransformedTrain",
            "transformed_train_data.npz",
        )
        self.test_file_path = os.path.join(
            "artifacts",
            "06_04_2026_11_25_42",
            "DataTransformationArtifacts",
            "TransformedTest",
            "transformed_test_data.npz",
        )

    # File Location (Production): src/utils/main_utils.py -> load_numpy_array_data
    def _load_data(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        if "features" in data and "target" in data:
            return data["features"], data["target"]
        elif "X" in data and "y" in data:
            return data["X"], data["y"]
        else:
            return data["arr_0"], data["arr_1"]

    # File Location (Production): src/components/model_trainer.py
    def _train_model(self, X_train, y_train):
        model = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        model.fit(X_train, y_train)
        return model

    # File Location (Production): src/utils/main_utils.py -> save_object
    def _save_model(self, model, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)

    # File Location (Production): src/components/model_trainer.py -> initiate_model_trainer
    def initiate_model_training(self):
        # 1. Load Data
        X_train, y_train = self._load_data(self.train_file_path)
        X_test, y_test = self._load_data(self.test_file_path)

        # 2. Train Model
        model = self._train_model(X_train, y_train)

        # 3. Evaluate Model
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)

        # 4. Save Model
        self._save_model(model, self.model_save_path)

        # File Location (Production): src/entity/artifact_entity.py -> ModelTrainerArtifact
        # Output of this component
        print("Version 3 - OOP with File Annotations")
        print(f"RMSE: {rmse}")
        print(f"Model saved to: {self.model_save_path}")


# File Location (Production): src/pipeline/training_pipeline.py
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_training()
