from src.logger import get_logger
import os
script_name = os.path.basename(__file__)
logger = get_logger(script_name)
logger.info("Hello")

