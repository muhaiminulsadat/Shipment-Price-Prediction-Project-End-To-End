import os
import json
import yaml
import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

# File Location: src/components/data_validation.py
class DataValidation:
    def __init__(self):
        # Step 1: Initialize all hardcoded values inside the constructor
        
        # File Location: src/entity/artifact_entity.py
        self.train_path = "artifacts/06_02_2026_23_33_10/DataIngestionArtifacts/Train/train.csv"
        self.test_path = "artifacts/06_02_2026_23_33_10/DataIngestionArtifacts/Test/test.csv"
        
        # File Location: src/entity/config_entity.py
        self.artifacts_dir = "artifacts/06_02_2026_23_33_10/DataValidationArtifacts"
        self.drift_report_path = "artifacts/06_02_2026_23_33_10/DataValidationArtifacts/DataDriftReport.yaml"
        self.expected_cols_count = 20
        self.numerical_cols = ["Artist Reputation", "Height", "Width", "Weight", "Price Of Sculpture", "Base Shipping Price"]
        self.categorical_cols = ["Customer Id", "Artist Name", "Material", "International", "Express Shipment", "Installation Included", "Transport", "Fragile", "Customer Information", "Remote Location", "Scheduled Date", "Delivery Date", "Customer Location"]

    # Step 2: Validate column length against schema
    def validate_schema_columns(self, df: pd.DataFrame) -> bool:
        return len(df.columns) == self.expected_cols_count

    # Step 3: Validate presence of numerical columns
    def is_numerical_column_exists(self, df: pd.DataFrame) -> bool:
        validation_status = False
        for column in self.numerical_cols:
            if column not in df.columns:
                print(f"Numerical column '{column}' not found in dataframe")
            else:
                validation_status = True
        return validation_status

    # Step 4: Validate presence of categorical columns
    def is_categorical_column_exists(self, df: pd.DataFrame) -> bool:
        validation_status = False
        for column in self.categorical_cols:
            if column not in df.columns:
                print(f"Categorical column '{column}' not found in dataframe")
            else:
                validation_status = True
        return validation_status

    # Step 5: Detect Dataset Drift using Evidently
    def detect_dataset_drift(self, reference: pd.DataFrame, production: pd.DataFrame) -> bool:
        data_drift_profile = Profile(sections=[DataDriftProfileSection()])
        data_drift_profile.calculate(reference, production)
        report_json = json.loads(data_drift_profile.json())
        
        os.makedirs(self.artifacts_dir, exist_ok=True)
        with open(self.drift_report_path, "w") as drift_file:
            yaml.dump(report_json, drift_file)
            
        return report_json["data_drift"]["data"]["metrics"]["dataset_drift"]

    # Step 6: Orchestrate the entire validation process
    def initiate_data_validation(self) -> bool:
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
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
            
        print("\n--- Version 3: Data Validation Output ---")
        print(f"Train Schema Columns Count Match Status: {train_col_len_status} (Expected: {self.expected_cols_count}, Got: {len(train_df.columns)})")
        print(f"Test Schema Columns Count Match Status: {test_col_len_status} (Expected: {self.expected_cols_count}, Got: {len(test_df.columns)})")
        print(f"Train Numerical Columns Exist: {train_num_status}")
        print(f"Test Numerical Columns Exist: {test_num_status}")
        print(f"Train Categorical Columns Exist: {train_cat_status}")
        print(f"Test Categorical Columns Exist: {test_cat_status}")
        print(f"Dataset Drift Detected: {dataset_drift}")
        print(f"Final Data Validation Status: {validation_status}")
        
        return validation_status

# File Location: src/pipeline/training_pipeline.py
if __name__ == "__main__":
    validator = DataValidation()
    validator.initiate_data_validation()
