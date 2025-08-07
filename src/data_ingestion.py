from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
from utils.common_functions import read_yaml
from sklearn.model_selection import train_test_split
import config 
import os
import gdown
import pandas as pd


config_path = config.CONFIG_PATH


logger = get_logger(__name__)


class DataIngestion:
    def __init__(self,config):
        self.config = config.data_ingestion
        self.train_ratio = self.config.train_ratio
        self.root_dir = self.config.root_dir
        self.train_file = self.config.train_file
        self.test_file = self.config.test_file
        self.source_url = self.config.source_url
        self.raw_file = self.config.raw_file


    def download_data_from_gdrive(self):

        url_file = self.config.source_url
        file_id = url_file.split("/")[-2]
        prefix ='https://drive.google.com/uc?/export=download&id='
        try:
            os.makedirs(self.config.root_dir , exist_ok=True)
            logger.info(f"Creating {self.config.root_dir} for downloading the raw data")
            
            output_path = os.path.join(self.config.root_dir , "raw.csv")
            logger.info(f"Downloading Data from gDrive")
            gdown.download(prefix+file_id , output=output_path)
            logger.info(f"Successfully downloaded")

        
        except Exception as e:
            logger.error(f"Error occured during downloading data from gDrive")
            raise CustomException(f"Failed to download csv file",e)
        

    def split_data(self):
        try:
            data = pd.read_csv(self.config.raw_file)
            logger.info("Starting Train test split")
            train_data , test_data= train_test_split(data , train_size=self.config.train_ratio , random_state=42)
            logger.info("Train Test Split Occur Successfully")
            train_data.to_csv(self.config.train_file)
            logger.info(f"Train Data is saved in {self.config.train_file}")
            test_data.to_csv(self.config.test_file)
            logger.info(f"Test Data is saved successfully in : {self.config.test_file}")

            logger.info("Files are saved successfully")
        except Exception as e:
            logger.error("Error in Data loading & Saving")
            raise CustomException(f"fail in saving the train & test data",e )
        

    def run(self):

        try :
            logger.info("Start the Ingestion Process Through run method")
            self.download_data_from_gdrive()
            self.split_data()

            logger.info("Downloading data & Splitting it into Train and Test occurred Successfully")

        except Exception as e:
            logger.error("Error in run method")
            raise CustomException("Error in Downloading & Splitting Data : ",e)
        
        finally:
            logger.info("Data Ingestion Step is Completed Successfully")

        



if __name__ == "__main__":
    config_parsed = read_yaml(config_path)
    data_ingestion = DataIngestion(config_parsed)
    data_ingestion.run()


