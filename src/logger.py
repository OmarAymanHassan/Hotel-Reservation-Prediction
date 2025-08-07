import logging 
import os 
from datetime import datetime


# first we need to create a log file !


logs_dir = "logs"
os.makedirs(logs_dir,exist_ok=True)


# store all the log in this file of `logs` folder

logs_file = os.path.join(logs_dir , f"log_{datetime.now().strftime("%Y-%m-%d")}.log")
# because i want the log file be looks like `log_2025-02-20.log`


# it tooks 3 main params 
# filename : our log file
# format : the logging format
logging.basicConfig(
    
    filename=logs_file,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO 
    # start logging from the info level till higher
    # the higher levels than info : warning , error

)

# levelname : is the main levels of the logfile 
# the main 3 levels are : warning,info,error
# info : For the General Messages 
# warning : for warning messages
# error: for error messages


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger 


