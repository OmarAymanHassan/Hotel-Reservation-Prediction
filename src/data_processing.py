from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml,load_data
import config 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.preprocessing import StandardScaler ,OneHotEncoder,OrdinalEncoder , FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report , precision_score, recall_score, f1_score ,roc_auc_score
import pandas as pd
import joblib



logger = get_logger(__name__)

class DataProcessing:

    def __init__(self,config):
        self.config = config.data_processing
        self.root_dir = self.config.processed_dir
        self.train_file = self.config.train_file
        self.test_file = self.config.test_file
        self.id_col = self.config.booking_id
        self.target_col = self.config.target_col
        self.train_ratio = self.config.train_ratio
        self.random_state = self.config.random_state
        self.skew_threshold = self.config.skewness_threshold

        if not os.path.exists(self.root_dir):
            logger.info("Create Processing Folder")
            os.makedirs(self.root_dir,exist_ok=True)
        
    

    def processing_data(self,csv_file):
        logger.info("Starting the PreProcessing Steps")

        logger.info("Loading the CSV File")
        df = load_data(csv_file)
        logger.info("Dropping the ID Column")
        df.drop(columns=[self.id_col] , inplace=True)
        return df 
    

    def splitting_cols(self,df):

        logger.info("Splitting Data Columns into Categorical and Numerical")

        cat_cols = [i for i in df.columns if df[i].dtype == "object" and i != self.target_col]
        numerical_cols = [i for i in df.columns if df[i].dtype !="object"]
        
        skewness = df[numerical_cols].skew()

        skewness_cols = skewness[skewness > self.skew_threshold].index.tolist()
            # ['no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled']
        non_skewness_cols = [i for i in numerical_cols if i not in skewness_cols]

        return numerical_cols , cat_cols , skewness_cols , non_skewness_cols 
    

    
    def create_pipeline(self , numerical_cols , cat_cols , skewness_cols , non_skewness_cols):

        logger.info("Creating the pipeline & ColumnTransformer")

        skeweed_transformer = Pipeline(steps= [("log_transformation",FunctionTransformer(np.log1p , validate=True)) , ("Scaling" , StandardScaler())] )
        non_skew_transformer = Pipeline(steps=[("scaling" , StandardScaler())])
        cat_transformer = Pipeline(steps=[("ordinal_encoding" , OrdinalEncoder(handle_unknown="use_encoded_value" , unknown_value=-1))])

        column_transformer = ColumnTransformer([("skewed_cols" , skeweed_transformer , skewness_cols) , ("non_skewed_cols" , non_skew_transformer , non_skewness_cols) , ("cat_cols" , cat_transformer , cat_cols)])
        full_pipeline = ImbPipeline(steps= [("processing_all_cols" , column_transformer) , ("smote" , SMOTE(random_state=42))])

        return full_pipeline
    

    def splitting_data(self, df, full_pipeline , skewness_cols , non_skewness_cols , cat_cols):

        
        logger.info("Train Test Split")

        x = df.drop(columns=[self.config.target_col])
        print(f"X shape : {x.shape}")
        y = df[self.config.target_col]

        x_train , x_test , y_train , y_test = train_test_split(x,y,train_size=self.config.train_ratio , random_state=self.config.random_state)

        logger.info(f"x_train shape :{x_train.shape}")
        logger.info(f"x_test shape :{x_test.shape}")
        logger.info(f"y_train shape :{y_train.shape}")
        logger.info(f"y_test shape :{y_test.shape}")


        logger.info("Encoding The Target Col")

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        
        logger.info("Modeling Stage")
        logger.info("Applying pipeline to x_train , y_train")

        print("Skewed cols:", skewness_cols)
        print("Non-skewed cols:", non_skewness_cols)
        print("Categorical cols:", cat_cols)
        print("Total columns covered:", len(skewness_cols + non_skewness_cols + cat_cols))

        x_train , y_train = full_pipeline.fit_resample(x_train , y_train)

        print(f"x_train shape After SMOTE :{x_train.shape}")
        print(f"y_train shape : {y_train.shape} ")

        logger.info("Applying pipeline to x_test")


        x_test  = full_pipeline.named_steps["processing_all_cols"].transform(x_test)


        logger.info("Conversion x_train , x_test into DataFrames")



        x_train = pd.DataFrame(x_train , columns=[skewness_cols + non_skewness_cols + cat_cols])
        x_test = pd.DataFrame(x_test , columns=[skewness_cols + non_skewness_cols + cat_cols])


        return x_train , x_test , y_train , y_test , le
    

    def modeling(self , x_train,x_test,y_train,y_test):

        logger.info("Training the Data on the model")

        model = RandomForestClassifier()
        model.fit(x_train , y_train)

        y_pred = model.predict(x_test)
        # Calc The performance on Test DataSet
        accuracy = accuracy_score(y_test , y_pred)
        roc = roc_auc_score(y_test , y_pred)
        precision = precision_score(y_test , y_pred)
        recall = recall_score(y_test , y_pred)
        f1 = f1_score(y_test , y_pred , average="weighted")

        logger.info("Here is the model Performance")

        logger.info(f"Accuracy : {accuracy}")
        logger.info(f"Precision : {precision}")
        logger.info(f"Recall : {recall}")
        logger.info(f"F1 Score : {f1}")

        logger.info(f"ROC : {roc}")

        return model
    

    def save_model(self , x_train , y_train , x_test , y_test  , full_pipeline , le):
        #logger.info("Saving Model and Data and Pipeline")
        #joblib.dump(model ,os.path.join(self.root_dir,"model.pkl"))
        joblib.dump(full_pipeline,os.path.join(self.root_dir,"pipeline.pkl"))
        joblib.dump(le,os.path.join(self.root_dir,"label_encoder.pkl"))

        x_train.to_csv(self.train_file , index=False)
        x_test.to_csv(self.test_file , index=False)

        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)

        y_train.to_csv(os.path.join(self.root_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(self.root_dir, "y_test.csv"), index=False)
    

    def run(self):
        logger.info("Running the Whole Process")

        df = self.processing_data(config.TRAIN_PATH)
        numerical_cols , cat_cols , skewness_cols , non_skewness_cols = self.splitting_cols(df)

        full_pipeline = self.create_pipeline(numerical_cols , cat_cols , skewness_cols , non_skewness_cols)

        x_train ,x_test, y_train , y_test,le = self.splitting_data(df ,full_pipeline , skewness_cols , non_skewness_cols , cat_cols)

        #model = self.modeling(x_train , x_test , y_train , y_test)

        self.save_model(x_train , y_train , x_test , y_test  , full_pipeline ,le)




if __name__ == "__main__":

    config_parsed = read_yaml(config_path=config.CONFIG_PATH)

    data_processing = DataProcessing(config_parsed)
    data_processing.run()



















