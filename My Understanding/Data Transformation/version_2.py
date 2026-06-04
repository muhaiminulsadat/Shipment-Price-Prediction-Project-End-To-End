import os
import dill
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
INPUT_TRAIN_DATA_PATH = "artifacts/06_02_2026_23_33_10/DataIngestionArtifacts/Train/train.csv"
INPUT_TEST_DATA_PATH = "artifacts/06_02_2026_23_33_10/DataIngestionArtifacts/Test/test.csv"

TRANSFORMED_TRAIN_DIR = "artifacts/06_02_2026_23_33_10/DataTransformationArtifacts/TransformedTrain"
TRANSFORMED_TEST_DIR = "artifacts/06_02_2026_23_33_10/DataTransformationArtifacts/TransformedTest"

OUTPUT_TRAIN_ARRAY_PATH = "artifacts/06_02_2026_23_33_10/DataTransformationArtifacts/TransformedTrain/transformed_train_data.npz"
OUTPUT_TEST_ARRAY_PATH = "artifacts/06_02_2026_23_33_10/DataTransformationArtifacts/TransformedTest/transformed_test_data.npz"
OUTPUT_PREPROCESSOR_PATH = "artifacts/06_02_2026_23_33_10/DataTransformationArtifacts/shipping_preprocessor.pkl"

TARGET_COLUMN = "Cost"
NUMERICAL_COLUMNS = ["Artist Reputation", "Height", "Width", "Weight", "Price Of Sculpture", "Base Shipping Price"]
ONEHOT_COLUMNS = ["Material", "Express Shipment", "Installation Included", "Transport", "Fragile", "Customer Information", "Remote Location"]
BINARY_COLUMNS = ["International"]

CONTINUOUS_UNIQUE_THRESHOLD = 25

if __name__ == "__main__":
    # Step 1: Read Train and Test data
    train_df = pd.read_csv(INPUT_TRAIN_DATA_PATH)
    test_df = pd.read_csv(INPUT_TEST_DATA_PATH)
    
    # Step 2: Outlier capping for continuous numerical columns
    continuous_columns = []
    for col in NUMERICAL_COLUMNS:
        if len(train_df[col].unique()) >= CONTINUOUS_UNIQUE_THRESHOLD:
            continuous_columns.append(col)
            
    for col in continuous_columns:
        # Capping outliers in training data
        q25_train = train_df[col].quantile(0.25)
        q75_train = train_df[col].quantile(0.75)
        iqr_train = q75_train - q25_train
        upper_limit_train = q75_train + 1.5 * iqr_train
        lower_limit_train = q25_train - 1.5 * iqr_train
        train_df.loc[train_df[col] > upper_limit_train, col] = upper_limit_train
        train_df.loc[train_df[col] < lower_limit_train, col] = lower_limit_train

        # Capping outliers in testing data
        q25_test = test_df[col].quantile(0.25)
        q75_test = test_df[col].quantile(0.75)
        iqr_test = q75_test - q25_test
        upper_limit_test = q75_test + 1.5 * iqr_test
        lower_limit_test = q25_test - 1.5 * iqr_test
        test_df.loc[test_df[col] > upper_limit_test, col] = upper_limit_test
        test_df.loc[test_df[col] < lower_limit_test, col] = lower_limit_test

    # Step 3: Separate input features and target feature
    x_train = train_df.drop(columns=[TARGET_COLUMN], axis=1)
    y_train = train_df[TARGET_COLUMN]
    
    x_test = test_df.drop(columns=[TARGET_COLUMN], axis=1)
    y_test = test_df[TARGET_COLUMN]

    # Step 4: Define encoders/scalers and preprocessor object
    oh_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    binary_encoder = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)
    scaler = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("OneHotEncoder", oh_encoder, ONEHOT_COLUMNS),
            ("BinaryEncoder", binary_encoder, BINARY_COLUMNS),
            ("StandardScaler", scaler, NUMERICAL_COLUMNS)
        ]
    )

    # Step 5: Transform input features
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)

    # Step 6: Combine transformed features with target feature
    train_arr = np.c_[x_train_transformed, np.array(y_train)]
    test_arr = np.c_[x_test_transformed, np.array(y_test)]

    # Step 7: Save preprocessor object and arrays
    os.makedirs(TRANSFORMED_TRAIN_DIR, exist_ok=True)
    os.makedirs(TRANSFORMED_TEST_DIR, exist_ok=True)
    
    with open(OUTPUT_TRAIN_ARRAY_PATH, "wb") as f:
        np.save(f, train_arr)
        
    with open(OUTPUT_TEST_ARRAY_PATH, "wb") as f:
        np.save(f, test_arr)
        
    with open(OUTPUT_PREPROCESSOR_PATH, "wb") as f:
        dill.dump(preprocessor, f)

    print("\n--- Version 2: Data Transformation Output ---")
    print(f"Transformed train shape: {train_arr.shape}")
    print(f"Transformed test shape: {test_arr.shape}")
    print(f"Saved transformed train data to: {OUTPUT_TRAIN_ARRAY_PATH}")
    print(f"Saved transformed test data to: {OUTPUT_TEST_ARRAY_PATH}")
    print(f"Saved preprocessor object to: {OUTPUT_PREPROCESSOR_PATH}")
