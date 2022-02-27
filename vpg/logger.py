import logging
from mpi_tools import proc_id

def get_logger(name,log_fp=None):
    if name is None:
        name=__name__
    logger = logging.getLogger(name)
    if log_fp is None:
        return logger
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_fp)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class Logger:
    def __init__(self,name,log_fp):
        self.name=name
        self.log_fp=log_fp
        if proc_id()==0:
            self.logger=get_logger(name,log_fp)
        
    def info(self,*args):
        if proc_id()==0:
            self.logger.info(*args)
        
    def debug(self,*args):
        if proc_id()==0:
            self.logger.debug(*args)