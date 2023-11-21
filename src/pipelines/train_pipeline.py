import os
from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:

    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            feature_file_path = data_ingestion.initiate_data_ingestion()
            return feature_file_path
        except Exception as e :
            raise CustomException(e, sys)

    def start_data_transformation(self, data_file_path):
        try:
            data_transformer = DataTransformation(data_file_path)
            train_arr, test_arr = data_transformer.initiate_data_transformation()
            return (train_arr, test_arr)

        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self, train_arr, test_arr):

        model_trainer = ModelTrainer(train_arr, test_arr)
        model_score = model_trainer.initiate_model_training()

        return model_score

    def run_pipeline(self):
        try:
            feature_store_file_path = self.start_data_ingestion()
            train_arr, test_arr = self.start_data_transformation(feature_store_file_path)
            score = self.start_model_training( train_arr, test_arr)

            print("final score is: "+score)
            
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    TrainingPipeline().run_pipeline()