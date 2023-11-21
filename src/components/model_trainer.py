from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer
from src.exception import CustomException
import sys
from sklearn.model_selection import GridSearchCV
from dataclasses import dataclass
import os
import pickle
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join('artifacts')
    training_model_path = os.path.join(artifact_folder,"model.pkl")


class ModelTrainer:
    def __init__(self, train_arr, test_arr) -> None:

        '''We will use 
            XGB classifier
            random forest classifier
            Support Vector Machines
        '''

        self.train_arr = train_arr
        self.test_arr = test_arr
        self.X_train = self.train_arr[:,:-1]
        self.y_train = self.train_arr[:,-1]
        self.X_test = self.test_arr[:,:-1]
        self.y_test = self.test_arr[:,-1]

        self.models = {"XGBClassifier": GradientBoostingClassifier(),
                       "RandomForestClassifier": RandomForestClassifier(),
                       "SVC": SVC()
                       }

    def evaluate_model(self, X_train, y_train, X_test, y_test, models):
        try:
            report = {}
            
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                test_score = accuracy_score(y_pred, y_test)
                report[model_name] = test_score

            return report
        
        except Exception as e:
            raise CustomException(sys, e)
        
    def get_best_model(self, report):
        try:
            best_model_name = max(report, key = lambda k : report[k])
            best_score = report[best_model]
            best_model = self.models[best_model]

            return (best_model_name,best_score, best_model)

        except Exception as e:
            raise CustomException(e,sys)
        
    def finetune_model(self, best_model_name, best_model, X_train, y_train):
        try:
            param_grids = {
                "RandomForestClassifier": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
                  },
                "GradientBoostingClassifier": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
                },
                "SVC": {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly']
                }
            }
            
            param_grid = param_grids[best_model_name]

            fine_tuner = GridSearchCV(best_model, param_grid, cv=5, scoring = make_scorer(accuracy_score))
            fine_tuner.fit(X_train, y_train)
            best_estimated_params = fine_tuner.best_params_
            best_finetuned_model = best_model.set_param(**best_estimated_params)

            return best_finetuned_model

        except Exception as e:
            raise CustomException(e, sys)
        

        
    def initiate_model_training(self):
        try:
            report = self.evaluate_model(self.X_train, self.y_train, self.X_test, self.y_test, self.models)
            best_model_name, best_score, best_model =self.get_best_model(report)
            best_finetuned_model = self.finetune_model(best_model_name, best_model, self.X_train, self.y_train)
            best_finetuned_model.fit(self.X_train, self.y_train)
            y_final_pred = best_finetuned_model.predict(self.X_test)
            final_score = accuracy_score(y_final_pred, self.y_test)
            model_trainer_config = ModelTrainerConfig()
            training_model_path = model_trainer_config.training_model_path
            
            logging.info("about to save the model pickle.")

            with open(training_model_path, 'wb') as file:
                pickle.dump(best_finetuned_model, file)

            return final_score



        except Exception as e:
            raise CustomException(e, sys)
        










