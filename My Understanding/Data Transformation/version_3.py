import os
import dill
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# File Location: src/components/data_transformation.py
class DataTransformation:
    def __init__(self):
        # Step 1: Initialize all hardcoded values inside the constructor
        
        # File Location: src/entity/artifact_entity.py
        self.train_path = "artifacts/06_02_2026_23_33_10/DataIngestionArtifacts/Train/train.csv"
        self.test_path = "artifacts/06_02_2026_23_33_10/DataIngestionArtifacts/Test/test.csv"
        
        # File Location: src/entity/config_entity.py
        self.transformed_train_dir = "artifacts/06_02_2026_23_33_10/DataTransformationArtifacts/TransformedTrain"
        self.transformed_test_dir = "artifacts/06_02_2026_23_33_10/DataTransformationArtifacts/TransformedTest"
        self.output_train_array_path = "artifacts/06_02_2026_23_33_10/DataTransformationArtifacts/TransformedTrain/transformed_train_data.npz"
        self.output_test_array_path = "artifacts/06_02_2026_23_33_10/DataTransformationArtifacts/TransformedTest/transformed_test_data.npz"
        self.output_preprocessor_path = "artifacts/06_02_2026_23_33_10/DataTransformationArtifacts/shipping_preprocessor.pkl"
        
        self.target_column = "Cost"
        self.numerical_cols = ["Artist Reputation", "Height", "Width", "Weight", "Price Of Sculpture", "Base Shipping Price"]
        self.onehot_cols = ["Material", "Express Shipment", "Installation Included", "Transport", "Fragile", "Customer Information", "Remote Location"]
        self.binary_cols = ["International"]
        
        self.continuous_unique_threshold = 25

    # Step 2: Define outlier capping helper method
    def _outlier_capping(self, col: str, df: pd.DataFrame) -> pd.DataFrame:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        upper_limit = q75 + 1.5 * iqr
        lower_limit = q25 - 1.5 * iqr
        df.loc[df[col] > upper_limit, col] = upper_limit
        df.loc[df[col] < lower_limit, col] = lower_limit
        return df

    # Step 3: Get preprocessor ColumnTransformer object
    def get_data_transformer_object(self) -> ColumnTransformer:
        oh_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        binary_encoder = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)
        scaler = StandardScaler()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("OneHotEncoder", oh_encoder, self.onehot_cols),
                ("BinaryEncoder", binary_encoder, self.binary_cols),
                ("StandardScaler", scaler, self.numerical_cols)
            ]
        )
        return preprocessor

    # Step 4: Orchestrate the entire data transformation process
    def initiate_data_transformation(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        # Determine continuous columns and cap outliers
        continuous_columns = [
            col for col in self.numerical_cols 
            if len(train_df[col].unique()) >= self.continuous_unique_threshold
        ]
        
        for col in continuous_columns:
            self._outlier_capping(col, train_df)
            self._outlier_capping(col, test_df)
            
        # Separate features and target
        x_train = train_df.drop(columns=[self.target_column], axis=1)
        y_train = train_df[self.target_column]
        
        x_test = test_df.drop(columns=[self.target_column], axis=1)
        y_test = test_df[self.target_column]
        
        # Transform features
        preprocessor = self.get_data_transformer_object()
        x_train_transformed = preprocessor.fit_transform(x_train)
        x_test_transformed = preprocessor.transform(x_test)
        
        # Concatenate features and target
        train_arr = np.c_[x_train_transformed, np.array(y_train)]
        test_arr = np.c_[x_test_transformed, np.array(y_test)]
        
        # Create directories and save artifacts
        os.makedirs(self.transformed_train_dir, exist_ok=True)
        os.makedirs(self.transformed_test_dir, exist_ok=True)
        
        with open(self.output_train_array_path, "wb") as f:
            np.save(f, train_arr)
            
        with open(self.output_test_array_path, "wb") as f:
            np.save(f, test_arr)
            
        with open(self.output_preprocessor_path, "wb") as f:
            dill.dump(preprocessor, f)
            
        print("\n--- Version 3: Data Transformation Output ---")
        print(f"Transformed train shape: {train_arr.shape}")
        print(f"Transformed test shape: {test_arr.shape}")
        print(f"Saved transformed train data to: {self.output_train_array_path}")
        print(f"Saved transformed test data to: {self.output_test_array_path}")
        print(f"Saved preprocessor object to: {self.output_preprocessor_path}")
        
        # File Location: src/entity/artifact_entity.py
        # Returns paths that will map to DataTransformationArtifacts dataclass
        return (self.output_preprocessor_path, self.output_train_array_path, self.output_test_array_path)

# File Location: src/pipeline/training_pipeline.py
if __name__ == "__main__":
    transformer = DataTransformation()
    transformer.initiate_data_transformation()
