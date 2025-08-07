from src.logger import get_logger
from src.custom_exception import CustomException
from config import training_config , TRAIN_PATH,CONFIG_PATH
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report , precision_score, recall_score, f1_score ,roc_auc_score
import joblib
import os
from utils.common_functions import read_yaml , load_data
import mlflow
import mlflow.sklearn


logger = get_logger(__name__)


class TrainingData:
    def __init__(self,config):
        self.config = config.data_training
        self.root_dir = self.config.root_dir
        self.processed_train = self.config.train_processed
        self.processed_test = self.config.test_processed
        self.model_params = training_config.light_params
        self.random_search = training_config.random_search_params
        self.y_train = self.config.y_train
        self.y_test = self.config.y_test
        self.model = self.config.model

        if not os.path.exists(self.root_dir):
            logger.info("Create Training Folder")
            os.makedirs(self.root_dir , exist_ok=True)





    def loading_data(self):
            
        try:

            logger.info("Loading x_train , x_test")

            x_train = pd.read_csv(self.processed_train)
            x_test = pd.read_csv(self.processed_test)

            print(f"x_train shape :{x_train.shape}")
            print(f"x_test shape :{x_test.shape}")

            logger.info("Loading y_train , y_test")

            y_train = pd.read_csv(self.y_train)
            y_train = y_train.iloc[: , 0]
            y_test = pd.read_csv(self.y_test)
            y_test = y_test.iloc[: , 0]

            print(f"y_train shape :{y_train.shape}")
            print(f"y_test shape :{y_test.shape}")
            print(x_train.columns.tolist())


            return x_train , x_test , y_train , y_test
        
        except Exception as e:
            logger.error(f"Error occured at loading processing Data : {e}")
            raise CustomException("Error in Loading Processed Data" , e)
        

    def lgbm_model(self, x_train ,y_train):
            
        try:

            logger.info("Starting the Training Process")

            model = LGBMClassifier(random_state= self.random_search.random_state)

            logger.info("Starting HyperParameterTuning ")

            model_rs = RandomizedSearchCV(estimator=model , param_distributions=self.model_params ,n_iter=self.random_search.n_iter , cv=self.random_search.cv , verbose=self.random_search.verbose , random_state=self.random_search.random_state , scoring = self.random_search.scoring)

            logger.info("Fit Data")
           
            model_rs.fit(x_train , y_train)

            logger.info("Getting the best parameters")

            best_model_params = model_rs.best_params_
            best_model = model_rs.best_estimator_

            logger.info(f"Best Model Parameters: {best_model_params}")

            return best_model
        
        except Exception as e:
            logger.error("Error in Training LGBM Model")
            raise CustomException("Erorr in Training the model" , e)
        

    def evaluate_model(self, x_test , y_test , best_model):
        try:
            logger.info("Evaluating the model")

            y_pred = best_model.predict(x_test)

            accuracy = accuracy_score(y_test , y_pred)
            roc = roc_auc_score(y_test , y_pred)
            precision = precision_score(y_test , y_pred)
            recall = recall_score(y_test , y_pred)
            f1 = f1_score(y_test , y_pred , average="weighted")


            logger.info(f"Accuracy : {accuracy}")
            logger.info(f"Precision : {precision}")
            logger.info(f"Recall : {recall}")
            logger.info(f"F1 Score : {f1}")

            logger.info(f"ROC : {roc}")
            

            return( {"accuracy" : accuracy , "precision" : precision , "recall" : recall , "f1 score" : f1})
        
        except Exception as e:
            logger.error(f"Error in Evaluating Model {e}")
            raise CustomException("Error in Evaluation", e)
        
    
    def save_model(self , best_model):
        try:
            logger.info("Saving Model")
            joblib.dump(best_model , os.path.join(self.root_dir , "model.pkl"))

        except Exception as e:
            logger.error("Error in Saving the model")
            raise CustomException("Error in Saving model" , e)



    def run(self):
        try :
            mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")


            with mlflow.start_run():
                logger.info("Starting the Training Process")
                logger.info("Starting MLFLOW Experiment Tracking")

                logger.info("Logging the Training  and testing dataset into MLFLOW")

                mlflow.log_artifact(self.processed_train, artifact_path="datasets" ) # make a folder called datasets , and put inside it the processed_train
                mlflow.log_artifact(self.processed_test , artifact_path="datasets")
                x_train , x_test , y_train , y_test = self.loading_data()

                best_model = self.lgbm_model(x_train , y_train)

                metrics = self.evaluate_model(x_test , y_test , best_model)
                self.save_model(best_model)

                logger.info("Logging the model into MLFLOW")
                mlflow.sklearn.log_model(best_model , name="model")
                mlflow.log_artifact(self.model , artifact_path="model")

                logger.info("Logging Model Params & Metrics into MLFLOW")
                mlflow.log_params(best_model.get_params())
                mlflow.log_metrics(metrics)


        except Exception as e:
            logger.error("Error in Running the Entire Training Process")
            raise CustomException("Error in Training Pipeline",e)
        


if __name__ =="__main__":
    parsed_config = read_yaml(CONFIG_PATH)
    training = TrainingData(parsed_config)
    training.run()








    



