import os
import json
import yaml
import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

# Global constants for configuration
TRAIN_DATA_PATH = "artifacts/06_02_2026_23_33_10/DataIngestionArtifacts/Train/train.csv"
TEST_DATA_PATH = "artifacts/06_02_2026_23_33_10/DataIngestionArtifacts/Test/test.csv"

DATA_VALIDATION_ARTIFACTS_DIR = "artifacts/06_02_2026_23_33_10/DataValidationArtifacts"
DATA_DRIFT_FILE_PATH = "artifacts/06_02_2026_23_33_10/DataValidationArtifacts/DataDriftReport.yaml"

EXPECTED_COLUMN_COUNT = 20

NUMERICAL_COLUMNS = ["Artist Reputation", "Height", "Width", "Weight", "Price Of Sculpture", "Base Shipping Price"]
CATEGORICAL_COLUMNS = ["Customer Id", "Artist Name", "Material", "International", "Express Shipment", "Installation Included", "Transport", "Fragile", "Customer Information", "Remote Location", "Scheduled Date", "Delivery Date", "Customer Location"]

if __name__ == "__main__":
    # Step 1: Read Train and Test data
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # Step 2: Check column length against schema
    train_col_len_status = len(train_df.columns) == EXPECTED_COLUMN_COUNT
    test_col_len_status = len(test_df.columns) == EXPECTED_COLUMN_COUNT
    
    # Step 3: Check if numerical columns exist
    train_numerical_status = False
    for column in NUMERICAL_COLUMNS:
        if column not in train_df.columns:
            print(f"Numerical column '{column}' not found in train dataframe")
        else:
            train_numerical_status = True
            
    test_numerical_status = False
    for column in NUMERICAL_COLUMNS:
        if column not in test_df.columns:
            print(f"Numerical column '{column}' not found in test dataframe")
        else:
            test_numerical_status = True
            
    # Step 4: Check if categorical columns exist
    train_categorical_status = False
    for column in CATEGORICAL_COLUMNS:
        if column not in train_df.columns:
            print(f"Categorical column '{column}' not found in train dataframe")
        else:
            train_categorical_status = True
            
    test_categorical_status = False
    for column in CATEGORICAL_COLUMNS:
        if column not in test_df.columns:
            print(f"Categorical column '{column}' not found in test dataframe")
        else:
            test_categorical_status = True
            
    # Step 5: Detect Dataset Drift using Evidently
    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(train_df, test_df)
    report_json = json.loads(data_drift_profile.json())
    
    os.makedirs(DATA_VALIDATION_ARTIFACTS_DIR, exist_ok=True)
    with open(DATA_DRIFT_FILE_PATH, "w") as drift_file:
        yaml.dump(report_json, drift_file)
        
    dataset_drift = report_json["data_drift"]["data"]["metrics"]["dataset_drift"]
    
    # Step 6: Determine the final validation status
    validation_status = False
    if (
        train_col_len_status is True
        and test_col_len_status is True
        and train_numerical_status is True
        and test_numerical_status is True
        and train_categorical_status is True
        and test_categorical_status is True
        and dataset_drift is False
    ):
        validation_status = True
        
    print("\n--- Version 2: Data Validation Output ---")
    print(f"Train Schema Columns Count Match Status: {train_col_len_status} (Expected: {EXPECTED_COLUMN_COUNT}, Got: {len(train_df.columns)})")
    print(f"Test Schema Columns Count Match Status: {test_col_len_status} (Expected: {EXPECTED_COLUMN_COUNT}, Got: {len(test_df.columns)})")
    print(f"Train Numerical Columns Exist: {train_numerical_status}")
    print(f"Test Numerical Columns Exist: {test_numerical_status}")
    print(f"Train Categorical Columns Exist: {train_categorical_status}")
    print(f"Test Categorical Columns Exist: {test_categorical_status}")
    print(f"Dataset Drift Detected: {dataset_drift}")
    print(f"Final Data Validation Status: {validation_status}")
