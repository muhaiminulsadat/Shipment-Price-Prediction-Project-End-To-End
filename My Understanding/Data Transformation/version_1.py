import os
import dill
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __name__ == "__main__":
    # Step 1: Read Train and Test data
    train_df = pd.read_csv("artifacts/06_02_2026_23_33_10/DataIngestionArtifacts/Train/train.csv")
    test_df = pd.read_csv("artifacts/06_02_2026_23_33_10/DataIngestionArtifacts/Test/test.csv")
    
    # Step 2: Outlier capping for continuous numerical columns (>= 25 unique values)
    numerical_columns = ["Artist Reputation", "Height", "Width", "Weight", "Price Of Sculpture", "Base Shipping Price"]
    continuous_columns = []
    for col in numerical_columns:
        if len(train_df[col].unique()) >= 25:
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
    target_column = "Cost"
    x_train = train_df.drop(columns=[target_column], axis=1)
    y_train = train_df[target_column]
    
    x_test = test_df.drop(columns=[target_column], axis=1)
    y_test = test_df[target_column]

    # Step 4: Define encoders/scalers and preprocessor object
    onehot_columns = ["Material", "Express Shipment", "Installation Included", "Transport", "Fragile", "Customer Information", "Remote Location"]
    binary_columns = ["International"]
    
    oh_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    binary_encoder = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)
    scaler = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("OneHotEncoder", oh_encoder, onehot_columns),
            ("BinaryEncoder", binary_encoder, binary_columns),
            ("StandardScaler", scaler, numerical_columns)
        ]
    )

    # Step 5: Transform input features
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)

    # Step 6: Combine transformed features with target feature
    train_arr = np.c_[x_train_transformed, np.array(y_train)]
    test_arr = np.c_[x_test_transformed, np.array(y_test)]

    # Step 7: Save preprocessor object and arrays
    os.makedirs("artifacts/06_02_2026_23_33_10/DataTransformationArtifacts/TransformedTrain", exist_ok=True)
    os.makedirs("artifacts/06_02_2026_23_33_10/DataTransformationArtifacts/TransformedTest", exist_ok=True)
    
    train_file_path = "artifacts/06_02_2026_23_33_10/DataTransformationArtifacts/TransformedTrain/transformed_train_data.npz"
    with open(train_file_path, "wb") as f:
        np.save(f, train_arr)
        
    test_file_path = "artifacts/06_02_2026_23_33_10/DataTransformationArtifacts/TransformedTest/transformed_test_data.npz"
    with open(test_file_path, "wb") as f:
        np.save(f, test_arr)
        
    preprocessor_file_path = "artifacts/06_02_2026_23_33_10/DataTransformationArtifacts/shipping_preprocessor.pkl"
    with open(preprocessor_file_path, "wb") as f:
        dill.dump(preprocessor, f)

    print("\n--- Version 1: Data Transformation Output ---")
    print(f"Transformed train shape: {train_arr.shape}")
    print(f"Transformed test shape: {test_arr.shape}")
    print(f"Saved transformed train data to: {train_file_path}")
    print(f"Saved transformed test data to: {test_file_path}")
    print(f"Saved preprocessor object to: {preprocessor_file_path}")
