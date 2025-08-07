from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.training_data import TrainingData
from utils.common_functions import read_yaml
import config

if __name__ == "__main__":

    # Data Ingestion 

    config_parsed = read_yaml(config.CONFIG_PATH)
    data_ingestion = DataIngestion(config_parsed)
    data_ingestion.run()




    # Data Processing
    config_parsed = read_yaml(config_path=config.CONFIG_PATH)

    data_processing = DataProcessing(config_parsed)
    data_processing.run()
    

    # Model Training

    parsed_config = read_yaml(config.CONFIG_PATH)
    training = TrainingData(parsed_config)
    training.run()