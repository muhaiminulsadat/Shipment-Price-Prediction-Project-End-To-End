import os
import sys
import json
import yaml
import pandas as pd
from dataclasses import dataclass
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

# Import production constants
# File Location: src/constants/__init__.py
from src.constants import (
    SCHEMA_FILE_PATH,
    DATA_INGESTION_ARTIFACTS_DIR,
    DATA_INGESTION_TRAIN_DIR,
    DATA_INGESTION_TEST_DIR,
    DATA_INGESTION_TRAIN_FILE_NAME,
    DATA_INGESTION_TEST_FILE_NAME,
    DATA_VALIDATION_ARTIFACT_DIR,
    DATA_DRIFT_FILE_NAME
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

# Helper function to load yaml
# File Location: src/utils/main_utils.py (inside MainUtils class)
def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "r") as stream:
            return yaml.safe_load(stream)
    except Exception as e:
        raise ShippingException(e, sys) from e

# Config Entity
# File Location: src/entity/config_entity.py
@dataclass
class DataValidationConfig:
    schema_config: dict
    data_validation_artifacts_dir: str
    data_drift_file_path: str

# Ingestion Artifact Entity
# File Location: src/entity/artifact_entity.py
@dataclass
class DataIngestionArtifacts:
    train_data_file_path: str
    test_data_file_path: str

# Validation Artifact Entity
# File Location: src/entity/artifact_entity.py
@dataclass
class DataValidationArtifacts:
    data_drift_file_path: str
    validation_status: bool

# Data Validation Component
# File Location: src/components/data_validation.py
class DataValidation:
    def __init__(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
        data_validation_config: DataValidationConfig
    ):
        # Step 1: Use Dependency Injection to pass config and ingestion artifacts
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_validation_config = data_validation_config

    # Step 2: Validate column length against schema
    def validate_schema_columns(self, df: pd.DataFrame) -> bool:
        try:
            expected_count = len(self.data_validation_config.schema_config["columns"])
            return len(df.columns) == expected_count
        except Exception as e:
            raise ShippingException(e, sys) from e

    # Step 3: Validate presence of numerical columns
    def is_numerical_column_exists(self, df: pd.DataFrame) -> bool:
        try:
            validation_status = False
            for column in self.data_validation_config.schema_config["numerical_columns"]:
                if column not in df.columns:
                    print(f"Numerical column '{column}' not found in dataframe")
                else:
                    validation_status = True
            return validation_status
        except Exception as e:
            raise ShippingException(e, sys) from e

    # Step 4: Validate presence of categorical columns
    def is_categorical_column_exists(self, df: pd.DataFrame) -> bool:
        try:
            validation_status = False
            for column in self.data_validation_config.schema_config["categorical_columns"]:
                if column not in df.columns:
                    print(f"Categorical column '{column}' not found in dataframe")
                else:
                    validation_status = True
            return validation_status
        except Exception as e:
            raise ShippingException(e, sys) from e

    # Step 5: Detect Dataset Drift using Evidently
    def detect_dataset_drift(self, reference: pd.DataFrame, production: pd.DataFrame) -> bool:
        try:
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])
            data_drift_profile.calculate(reference, production)
            report_json = json.loads(data_drift_profile.json())
            
            os.makedirs(self.data_validation_config.data_validation_artifacts_dir, exist_ok=True)
            with open(self.data_validation_config.data_drift_file_path, "w") as drift_file:
                yaml.dump(report_json, drift_file)
                
            return report_json["data_drift"]["data"]["metrics"]["dataset_drift"]
        except Exception as e:
            raise ShippingException(e, sys) from e

    # Step 6: Orchestrate the entire validation process and return Artifact Entity
    def initiate_data_validation(self) -> DataValidationArtifacts:
        try:
            train_df = pd.read_csv(self.data_ingestion_artifacts.train_data_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifacts.test_data_file_path)
            
            train_col_len_status = self.validate_schema_columns(train_df)
            test_col_len_status = self.validate_schema_columns(test_df)
            
            train_num_status = self.is_numerical_column_exists(train_df)
            test_num_status = self.is_numerical_column_exists(test_df)
            
            train_cat_status = self.is_categorical_column_exists(train_df)
            test_cat_status = self.is_categorical_column_exists(test_df)
            
            dataset_drift = self.detect_dataset_drift(train_df, test_df)
            
            validation_status = False
            if (
                train_col_len_status is True
                and test_col_len_status is True
                and train_num_status is True
                and test_num_status is True
                and train_cat_status is True
                and test_cat_status is True
                and dataset_drift is False
            ):
                validation_status = True
                
            return DataValidationArtifacts(
                data_drift_file_path=self.data_validation_config.data_drift_file_path,
                validation_status=validation_status
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
        val_artifacts_dir = os.path.join(
            latest_run_dir,
            DATA_VALIDATION_ARTIFACT_DIR
        )
        drift_yaml_path = os.path.join(
            val_artifacts_dir,
            DATA_DRIFT_FILE_NAME
        )
        
        # Initialize Entities using dynamically resolved paths
        ingestion_artifacts = DataIngestionArtifacts(
            train_data_file_path=train_csv_path,
            test_data_file_path=test_csv_path
        )
        
        validation_config = DataValidationConfig(
            schema_config=schema_data,
            data_validation_artifacts_dir=val_artifacts_dir,
            data_drift_file_path=drift_yaml_path
        )
        
        # Execute validation component
        validator = DataValidation(
            data_ingestion_artifacts=ingestion_artifacts,
            data_validation_config=validation_config
        )
        artifacts_output = validator.initiate_data_validation()
        
        print("\n--- Version 4: Data Validation Output ---")
        print(f"Data Drift File Saved At: {artifacts_output.data_drift_file_path}")
        print(f"Final Data Validation Status: {artifacts_output.validation_status}")
        
    except Exception as e:
        print(f"Validation Pipeline Failed: {e}")
