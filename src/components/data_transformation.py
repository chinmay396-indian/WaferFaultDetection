import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
import sys
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import pickle

@dataclass
class DataTransformationConfig:
    artifact_file_path = os.path.join("artiacts")
    preprocesor_path = os.path.join(artifact_file_path, "preprocessor.pkl")

class DataTransformation:

    def __init__(self, data_file_path: str):
        self.data_path = data_file_path

    @staticmethod
    def get_data(data_file_path: str) -> pd.DataFrame:

        try:

            data = pd.read_csv(data_file_path)
            if "Unnamed: 0" in data.columns:
                data.drop(columns="Unnamed: 0", inplace=True)
            return data
        except Exception as e:
            raise CustomException(e, sys)
        


        
    def get_data_transformation_object(self):
        try:
            imputer = KNNImputer(n_neighbors=3)
            preprocessor = Pipeline(
                steps=[("Imputer",imputer),("Scaler",RobustScaler())]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)




    def initiate_data_transformation(self):
        try:
            data = self.get_data(data_file_path=self.data_path)
            X = data[:,:-1]
            y = data[:,-1]
            X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)
            preprocessor = self.get_data_transformation_object()
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.fit_transform(X_test)

            X_train_arr = np.c_(X_train_scaled,np.array(y_train))
            X_test_arr = np.c_(X_test, np.array(y_test))

            data_transformation_config = DataTransformationConfig()
            preprocessor_path = data_transformation_config.preprocesor_path

            with open(preprocessor_path,"wb") as file:
                pickle.dump(preprocessor, file)

            return(X_train_arr, X_test_arr)


        except Exception as e:
            raise CustomException(e, sys)

        