from dataclasses import dataclass
import os
from src.logger import logging
from src.exception import CustomException
import sys

@dataclass
class DataIngestionConfig:
    project_path = r'C:\Projects\SensorProject'
    artifact_file: str = os.path.join(project_path, "artifacts")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> str:
        try:
        
            logging.info("Data exporting started.")
            raw_file_path = self.data_ingestion_config.artifact_file

            # Use os.makedirs to create the directory with exist_ok=True
            os.makedirs(raw_file_path, exist_ok=True)

            feature_store_file_path = os.path.join(raw_file_path, 'wafer_fault.csv')

            return feature_store_file_path

        except Exception as e:
            raise CustomException(e, sys)


