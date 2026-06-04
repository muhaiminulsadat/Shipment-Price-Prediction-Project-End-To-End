import os
import sys
import dill
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import production constants
# File Location: src/constants/__init__.py
from src.constants import (
    SCHEMA_FILE_PATH,
    DATA_INGESTION_ARTIFACTS_DIR,
    DATA_INGESTION_TRAIN_DIR,
    DATA_INGESTION_TEST_DIR,
    DATA_INGESTION_TRAIN_FILE_NAME,
    DATA_INGESTION_TEST_FILE_NAME,
    DATA_TRANSFORMATION_ARTIFCATS_DIR,
    TRANSFORMED_TRAIN_DATA_DIR,
    TRANSFORMED_TEST_DATA_DIR,
    TRANSFORMED_TRAIN_DATA_FILE_NAME,
    TRANSFORMED_TEST_DATA_FILE_NAME,
    PREPROCESSOR_OBJECT_FILE_NAME
)

# Custom Exception Class
# File Location: src/exception/__init__.py
class ShippingException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)
        _, _, exc_tb = error_detail.exc_info()
        self.file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        self.line_number = exc_tb.tb_lineno
        self.error_message = f"Error occurred in python script name [{self.file_name}] line number [{self.line_number}] error message [{str(error_message)}]"

    def __str__(self):
        return self.error_message

# Helper function to load YAML
# File Location: src/utils/main_utils.py (inside MainUtils class)
def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "r") as stream:
            return yaml.safe_load(stream)
    except Exception as e:
        raise ShippingException(e, sys) from e

# Helper functions for saving arrays and objects
# File Location: src/utils/main_utils.py (inside MainUtils class)
def save_numpy_array_data(file_path: str, array: np.array) -> str:
    try:
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        return file_path
    except Exception as e:
        raise ShippingException(e, sys) from e

def save_object(file_path: str, obj: object) -> str:
    try:
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        return file_path
    except Exception as e:
        raise ShippingException(e, sys) from e

# Ingestion Artifact Entity
# File Location: src/entity/artifact_entity.py
@dataclass
class DataIngestionArtifacts:
    train_data_file_path: str
    test_data_file_path: str

# Config Entity
# File Location: src/entity/config_entity.py
@dataclass
class DataTransformationConfig:
    schema_config: dict
    data_transformation_artifacts_dir: str
    transformed_train_dir: str
    transformed_test_dir: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessor_file_path: str

# Transformation Artifact Entity
# File Location: src/entity/artifact_entity.py
@dataclass
class DataTransformationArtifacts:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

# Data Transformation Component
# File Location: src/components/data_transformation.py
class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
        data_transformation_config: DataTransformationConfig
    ):
        # Step 1: Dependency Injection
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_transformation_config = data_transformation_config

    # Step 2: Outlier capping helper method
    def _outlier_capping(self, col: str, df: pd.DataFrame) -> pd.DataFrame:
        try:
            percentile25 = df[col].quantile(0.25)
            percentile75 = df[col].quantile(0.75)
            iqr = percentile75 - percentile25
            upper_limit = percentile75 + 1.5 * iqr
            lower_limit = percentile25 - 1.5 * iqr
            df.loc[(df[col] > upper_limit), col] = upper_limit
            df.loc[(df[col] < lower_limit), col] = lower_limit
            return df
        except Exception as e:
            raise ShippingException(e, sys) from e

    # Step 3: Get preprocessor ColumnTransformer object
    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            numerical_columns = self.data_transformation_config.schema_config["numerical_columns"]
            original_onehot_columns = self.data_transformation_config.schema_config["onehot_columns"]
            binary_columns = self.data_transformation_config.schema_config["binary_columns"]
            
            # Exclude binary columns from standard OneHot
            onehot_columns = [
                col for col in original_onehot_columns if col not in binary_columns
            ]
            
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            binary_transformer = OneHotEncoder(
                handle_unknown="ignore", drop="if_binary", sparse_output=False
            )
            
            transformers = []
            if onehot_columns:
                transformers.append(("OneHotEncoder", oh_transformer, onehot_columns))
            if binary_columns:
                transformers.append(("BinaryEncoder", binary_transformer, binary_columns))
            transformers.append(("StandardScaler", numeric_transformer, numerical_columns))
            
            preprocessor = ColumnTransformer(transformers)
            return preprocessor
        except Exception as e:
            raise ShippingException(e, sys) from e

    # Step 4: Orchestrate the entire transformation process and return Artifact Entity
    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            # Create parent artifacts directory
            os.makedirs(self.data_transformation_config.data_transformation_artifacts_dir, exist_ok=True)
            
            train_df = pd.read_csv(self.data_ingestion_artifacts.train_data_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifacts.test_data_file_path)
            
            preprocessor = self.get_data_transformer_object()
            
            target_column_name = self.data_transformation_config.schema_config["target_column"]
            numerical_columns = self.data_transformation_config.schema_config["numerical_columns"]
            
            # Outlier capping
            continuous_columns = [
                feature
                for feature in numerical_columns
                if len(train_df[feature].unique()) >= 25
            ]
            for col in continuous_columns:
                self._outlier_capping(col, train_df)
                self._outlier_capping(col, test_df)
                
            # Split features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # Apply transformations
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            # Concatenate features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Save transformed arrays
            os.makedirs(self.data_transformation_config.transformed_train_dir, exist_ok=True)
            transformed_train_file = save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )
            
            os.makedirs(self.data_transformation_config.transformed_test_dir, exist_ok=True)
            transformed_test_file = save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )
            
            # Save preprocessor object
            preprocessor_obj_file = save_object(
                self.data_transformation_config.preprocessor_file_path,
                preprocessor
            )
            
            return DataTransformationArtifacts(
                transformed_object_file_path=preprocessor_obj_file,
                transformed_train_file_path=transformed_train_file,
                transformed_test_file_path=transformed_test_file
            )
            
        except Exception as e:
            raise ShippingException(e, sys) from e

# Pipeline Execution logic
# File Location: src/pipeline/training_pipeline.py (or execution files like app.py/demo.py)
if __name__ == "__main__":
    try:
        # Load schema configuration dynamically from constants
        schema_data = read_yaml_file(SCHEMA_FILE_PATH)
        
        # Dynamically locate the latest runs/artifacts folder to avoid hardcoded timestamp folder
        artifacts_root_dir = os.path.join(os.getcwd(), "artifacts")
        all_subdirs = [
            os.path.join(artifacts_root_dir, d) 
            for d in os.listdir(artifacts_root_dir) 
            if os.path.isdir(os.path.join(artifacts_root_dir, d))
        ]
        latest_run_dir = max(all_subdirs, key=os.path.getmtime)
        
        # Build paths dynamically
        train_csv_path = os.path.join(
            latest_run_dir,
            DATA_INGESTION_ARTIFACTS_DIR,
            DATA_INGESTION_TRAIN_DIR,
            DATA_INGESTION_TRAIN_FILE_NAME
        )
        test_csv_path = os.path.join(
            latest_run_dir,
            DATA_INGESTION_ARTIFACTS_DIR,
            DATA_INGESTION_TEST_DIR,
            DATA_INGESTION_TEST_FILE_NAME
        )
        
        trans_artifacts_dir = os.path.join(
            latest_run_dir,
            DATA_TRANSFORMATION_ARTIFCATS_DIR
        )
        transformed_train_dir = os.path.join(
            trans_artifacts_dir,
            TRANSFORMED_TRAIN_DATA_DIR
        )
        transformed_test_dir = os.path.join(
            trans_artifacts_dir,
            TRANSFORMED_TEST_DATA_DIR
        )
        transformed_train_path = os.path.join(
            transformed_train_dir,
            TRANSFORMED_TRAIN_DATA_FILE_NAME
        )
        transformed_test_path = os.path.join(
            transformed_test_dir,
            TRANSFORMED_TEST_DATA_FILE_NAME
        )
        preprocessor_pkl_path = os.path.join(
            trans_artifacts_dir,
            PREPROCESSOR_OBJECT_FILE_NAME
        )
        
        # Initialize Entities using dynamically resolved paths
        ingestion_artifacts = DataIngestionArtifacts(
            train_data_file_path=train_csv_path,
            test_data_file_path=test_csv_path
        )
        
        transformation_config = DataTransformationConfig(
            schema_config=schema_data,
            data_transformation_artifacts_dir=trans_artifacts_dir,
            transformed_train_dir=transformed_train_dir,
            transformed_test_dir=transformed_test_dir,
            transformed_train_file_path=transformed_train_path,
            transformed_test_file_path=transformed_test_path,
            preprocessor_file_path=preprocessor_pkl_path
        )
        
        # Execute data transformation component
        transformer = DataTransformation(
            data_ingestion_artifacts=ingestion_artifacts,
            data_transformation_config=transformation_config
        )
        artifacts_output = transformer.initiate_data_transformation()
        
        print("\n--- Version 4: Data Transformation Output ---")
        print(f"Preprocessor Saved At: {artifacts_output.transformed_object_file_path}")
        print(f"Transformed Train Saved At: {artifacts_output.transformed_train_file_path}")
        print(f"Transformed Test Saved At: {artifacts_output.transformed_test_file_path}")
        
    except Exception as e:
        print(f"Data Transformation Pipeline Failed: {e}")
