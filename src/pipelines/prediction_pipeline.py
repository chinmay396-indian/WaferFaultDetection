import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import sys
import pandas as pd
import pickle


@dataclass
class PredictionPipelineConfig:
    artifacts_path = os.path.join("artifacts")
    prediction_file_path = os.path.join(artifacts_path, "predicted_file.csv")
    preprocessor_path = os.path.join(artifacts_path, "preprocessor.pkl")
    model_path = os.path.join(artifacts_path, "model.pkl")
    final_predicted_data_path = os.path.join(artifacts_path,"final_predictions.csv")


class PredictionPipeline:
    def __init__(self, request) -> None:
        self.request = request
        self.prediction_pipeine_config = PredictionPipelineConfig()


    def save_input_file(self):
        try:

            prediction_path = self.prediction_pipeine_config.prediction_file_path

            return prediction_path


        except Exception as e:
            raise CustomException(e, sys)
        
    def predict(self, prediction_path):
        try:
            data = pd.read_csv(prediction_path)
            preprocessor_path = self.prediction_pipeine_config.preprocessor_path
            model_path = self.prediction_pipeine_config.model_path
            with open(preprocessor_path, "wb") as file:
                preprocessor = pickle.load(file)
            
            with open(model_path, "wb") as file:
                model = pickle.load(file)

            data_trans = preprocessor.fit_transform(data)
            preds = model.predict(data_trans)

            return preds


        except Exception as e:
            raise CustomException(e, sys)
        
    def get_predicted_dataframe(self, predicted_target_df, prediction_path):
        try:
            independent_data = pd.read_csv(prediction_path)
            dependent_predicted_data = predicted_target_df
            final_predicted_df = pd.concat([independent_data, dependent_predicted_data], axis =1)

            prediction_file_path = self.prediction_pipeine_config.prediction_file_path

            final_predicted_df.to_csv(prediction_file_path, index=False)

            return final_predicted_df
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def run_pipeline(self):
        try:
            
            prediction_path = self.save_input_file()
            predicted_target = self.predict(prediction_path)
            predicted_target_df = pd.DataFrame(predicted_target)
            self.get_predicted_dataframe(predicted_target_df=predicted_target_df, prediction_path= prediction_path)

            return self.prediction_pipeine_config


        except Exception as e:
            raise CustomException(e, sys)


    