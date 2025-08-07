from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
import os
import config
from box import ConfigBox
import pandas as pd




logger = get_logger(__name__)


def read_yaml(config_path):
    try:
        logger.info(f"Reading Yaml File : {config_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file is not Found in the given path")
        
        with open(config_path,"r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info(f"Loading all the config file successfully")
            return ConfigBox(config)
    except Exception as e:
        logger.error("Error in Loading YAML File as Dictionary")
        raise CustomException("Fail to read yaml file : ",e)




def load_data(path):
    try:
        logger.info(f"Reading the CSV file from : {path}")
        df = pd.read_csv(path)
        logger.info("Data read Successfully")
        return df
    except Exception as e:
        logger.error(f"Error in Reading the CSV file")
        raise CustomException("Error in Loading the csv file", e)
    
