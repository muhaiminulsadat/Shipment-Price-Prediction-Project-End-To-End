import os
import sys
import joblib
import yaml
import numpy as np
from dataclasses import dataclass
from typing import Dict
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Exception mapping: src/exception/__init__.py
from src.exception import shippingException

# Config entity mapping: src/entity/config_entity.py
@dataclass
class ModelTrainerConfig:
    model_dir: str
    model_filename: str
    model_params: Dict

# Artifact entity mapping: src/entity/artifact_entity.py
@dataclass
class DataTransformationArtifact:
    transformed_train_path: str
    transformed_test_path: str

# Component mapping: src/components/model_trainer.py
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, artifact: DataTransformationArtifact):
        self.config = config
        self.artifact = artifact

    def _load_npz(self, path: str):
        try:
            data = np.load(path, allow_pickle=True)
            if 'features' in data and 'target' in data:
                return data['features'], data['target']
            if 'X' in data and 'y' in data:
                return data['X'], data['y']
            return data['arr_0'], data['arr_1']
        except Exception as e:
            raise shippingException(e, sys)

    def _train(self, X, y):
        try:
            model = RandomForestRegressor(**self.config.model_params)
            model.fit(X, y)
            return model
        except Exception as e:
            raise shippingException(e, sys)

    def _evaluate(self, model, X, y):
        try:
            preds = model.predict(X)
            return mean_squared_error(y, preds, squared=False)
        except Exception as e:
            raise shippingException(e, sys)

    def _save(self, model):
        try:
            save_path = os.path.join(self.config.model_dir, self.config.model_filename)
            os.makedirs(self.config.model_dir, exist_ok=True)
            joblib.dump(model, save_path)
            return save_path
        except Exception as e:
            raise shippingException(e, sys)

    def run(self):
        try:
            X_train, y_train = self._load_npz(self.artifact.transformed_train_path)
            X_test, y_test = self._load_npz(self.artifact.transformed_test_path)

            model = self._train(X_train, y_train)
            rmse = self._evaluate(model, X_test, y_test)
            saved_path = self._save(model)

            print('Version 4 — OOP Production-style trainer')
            print(f'Train shape: {getattr(X_train, "shape", None)}')
            print(f'Test shape: {getattr(X_test, "shape", None)}')
            print(f'RMSE: {rmse}')
            print(f'Model saved to: {saved_path}')

            return {'model_path': saved_path, 'rmse': rmse}
        except Exception as e:
            raise shippingException(e, sys)


# Utility mapping: src/utils/main_utils.py
def load_yaml(path: str):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise shippingException(e, sys)


def find_latest_artifact_dir(artifacts_root: str = 'artifacts') -> str:
    try:
        entries = [
            os.path.join(artifacts_root, d) for d in os.listdir(artifacts_root)
            if os.path.isdir(os.path.join(artifacts_root, d))
        ]
        # attempt to parse folders as timestamps like 06_04_2026_11_25_42
        def parse_dt(name: str):
            base = os.path.basename(name)
            for fmt in ('%m_%d_%Y_%H_%M_%S', '%Y_%m_%d_%H_%M_%S'):
                try:
                    return datetime.strptime(base, fmt)
                except Exception:
                    continue
            return None

        dated = [(p, parse_dt(p)) for p in entries]
        dated = [(p, d) for (p, d) in dated if d is not None]
        if dated:
            latest = max(dated, key=lambda x: x[1])[0]
            return latest
        # fallback lexicographic
        return max(entries)
    except Exception as e:
        raise shippingException(e, sys)


def discover_transformed_paths(artifacts_root: str = 'artifacts') -> DataTransformationArtifact:
    try:
        latest = find_latest_artifact_dir(artifacts_root)
        train_npz = os.path.join(latest, 'DataTransformationArtifacts', 'TransformedTrain', 'transformed_train_data.npz')
        test_npz = os.path.join(latest, 'DataTransformationArtifacts', 'TransformedTest', 'transformed_test_data.npz')
        if not os.path.exists(train_npz) or not os.path.exists(test_npz):
            raise FileNotFoundError('Transformed train/test npz not found in latest artifact folder')
        return DataTransformationArtifact(transformed_train_path=train_npz, transformed_test_path=test_npz)
    except Exception as e:
        raise shippingException(e, sys)


# Pipeline mapping: src/pipeline/training_pipeline.py
if __name__ == '__main__':
    try:
        # Load model hyperparameter grid and defaults from config/model.yaml
        config_dict = load_yaml(os.path.join('config', 'model.yaml'))
        # For simplicity pick first values as defaults
        rf_params = {}
        try:
            rf_cfg = config_dict.get('train_model', {}).get('RandomForestRegressor', {})
            rf_params['n_estimators'] = int(rf_cfg.get('n_estimators', [100])[0])
            if rf_cfg.get('max_depth'):
                rf_params['max_depth'] = int(rf_cfg.get('max_depth')[0])
            if rf_cfg.get('max_features'):
                rf_params['max_features'] = int(rf_cfg.get('max_features')[0])
        except Exception:
            rf_params = {'n_estimators': 100}

        # Build config and artifact objects (dependency injection)
        model_dir = os.path.join('artifacts', 'ModelTrainer')
        model_filename = 'model_v4.pkl'
        trainer_config = ModelTrainerConfig(model_dir=model_dir, model_filename=model_filename, model_params=rf_params)

        data_artifact = discover_transformed_paths('artifacts')

        trainer = ModelTrainer(config=trainer_config, artifact=data_artifact)
        trainer.run()
    except Exception as e:
        raise shippingException(e, sys)
