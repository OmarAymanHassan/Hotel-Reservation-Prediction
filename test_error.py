from src.custom_exception import CustomException
from src.logger import get_logger

import sys


logger = get_logger(__name__)

def division(a,b):
    try :
        result= a/b
        logger.info("Division two numbers")
        return result
    except Exception as e:
        logger.error("Error Occured")
        raise CustomException("Custom error Zero " , sys)


if __name__ == "__main__":
    try:
        logger.info("Starting the main Program")
        division(10,2)
        logger.info("Division Occured Successfully")
    except CustomException as e:
        logger.error(str(e))
