import os 
import logging

def create_logger(name, log_file, level=logging.INFO):

    folder = log_file.replace(log_file.split('/')[-1], "")
    if os.path.isdir(folder) is False:
        os.mkdir(folder)

    """Function setup as many loggers as you want"""
    logging.basicConfig(filename = log_file, level = level, format = '%(asctime)s %(levelname)s %(message)s')
    handler = logging.StreamHandler()    

    logger = logging.getLogger(name)
    logger.addHandler(handler)

    return logger
    