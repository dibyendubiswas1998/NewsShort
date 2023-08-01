import logging
from datetime import datetime


def log(file_object:str, log_message:str):
    if not file_object or not log_message:
        raise ValueError('file_object and log_message cannot be None or empty')
    try:
        now = datetime.now()
        date = now.date()
        current_time = now.strftime("%H:%M:%S")
        
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()

        file_handler = logging.FileHandler(filename=file_object)
        logger.addHandler(file_handler)
        logging.info(f'{date}\t{current_time}\t{log_message}')
        
    except Exception as e:
        raise e