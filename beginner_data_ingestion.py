import os
import sys
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

load_dotenv()

class ShipmentException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)
        _, _, exc_tb = error_detail.exc_info()
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.line_number = exc_tb.tb_lineno
        self.error_message = f"Error occurred in python script name [{self.file_name}] line number [{self.line_number}] error message [{str(error_message)}]"

    def __str__(self):
        return self.error_message


# --- 1. CONFIGURATION ENTITY ---
@dataclass
class DataIngestionConfig:
    mongodb_uri: str = os.getenv("MONGODB_URI")
    db_name: str = "ShipmentDB"
    collection_name: str = "shipment_collection"
    local_csv_path: str = "data/Shipment-data.csv"
    drop_columns: tuple = ("Customer Id", "Artist Name", "Customer Location", "Scheduled Date", "Delivery Date")
    
    artifacts_dir: str = os.path.join(os.getcwd(), "artifacts", "DataIngestion")
    train_dir: str = os.path.join(artifacts_dir, "Train")
    test_dir: str = os.path.join(artifacts_dir, "Test")
    train_file_path: str = os.path.join(train_dir, "train.csv")
    test_file_path: str = os.path.join(test_dir, "test.csv")
    
    test_size: float = 0.2


# --- 2. ARTIFACT ENTITY ---
@dataclass
class DataIngestionArtifacts:
    train_file_path: str
    test_file_path: str


# --- 3. DATA INGESTION COMPONENT ---
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def get_data_from_mongodb(self) -> pd.DataFrame:
        try:
            if self.config.mongodb_uri:
                client = MongoClient(self.config.mongodb_uri, serverSelectionTimeoutMS=5000)
                db = client[self.config.db_name]
                collection = db[self.config.collection_name]
                
                df = pd.DataFrame(list(collection.find()))
                if not df.empty:
                    if "_id" in df.columns:
                        df = df.drop(columns=["_id"])
                    return df

        except Exception:
            pass

        if os.path.exists(self.config.local_csv_path):
            df = pd.read_csv(self.config.local_csv_path)
            return df
        else:
            raise FileNotFoundError(f"Local fallback CSV data not found at: {self.config.local_csv_path}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.drop(list(self.config.drop_columns), axis=1)
            df = df.dropna()
            return df
        except Exception as e:
            raise ShipmentException(e, sys)

    def split_and_save_data(self, df: pd.DataFrame) -> DataIngestionArtifacts:
        try:
            os.makedirs(self.config.train_dir, exist_ok=True)
            os.makedirs(self.config.test_dir, exist_ok=True)
            
            train_set, test_set = train_test_split(df, test_size=self.config.test_size)
            
            train_set.to_csv(self.config.train_file_path, index=False, header=True)
            test_set.to_csv(self.config.test_file_path, index=False, header=True)
            
            artifacts = DataIngestionArtifacts(
                train_file_path=self.config.train_file_path,
                test_file_path=self.config.test_file_path
            )
            return artifacts

        except Exception as e:
            raise ShipmentException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        try:
            raw_df = self.get_data_from_mongodb()
            cleaned_df = self.clean_data(raw_df)
            ingestion_artifacts = self.split_and_save_data(cleaned_df)
            return ingestion_artifacts
        except Exception as e:
            raise ShipmentException(e, sys)


# --- 4. PIPELINE EXECUTION ---
if __name__ == "__main__":
    try:
        ingestion_config = DataIngestionConfig()
        ingestion_component = DataIngestion(config=ingestion_config)
        artifacts_output = ingestion_component.initiate_data_ingestion()
        
        print("\n--- Pipeline Run Output ---")
        print(f"Train File Path: {artifacts_output.train_file_path}")
        print(f"Test File Path: {artifacts_output.test_file_path}")
        
    except Exception as e:
        print(f"Pipeline crashed during execution: {e}")



