from dataclasses import dataclass
import os
import sys
from src.logger import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.utils import evaluate_models
from src.exception import customException
from src.logger import logging
from src.utils import save_object
from sklearn.metrics import r2_score

@dataclass
class ModelTraningConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class modeltrainer:
    def __init__(self):
        self.model_trainer_config = ModelTraningConfig()


    def inittiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("splitting training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models= {
                'dt':DecisionTreeClassifier(random_state=42),
                'rf':RandomForestClassifier(random_state=42)
            }

            model_report:dict=evaluate_models(x_train=x_train, y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise customException("No best Model found")
            
            
            logging.info(f'Best model found on both training and test dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)

            r2 = r2_score(y_test, predicted)
            return r2

        except Exception as e:
            raise customException(e,sys)

