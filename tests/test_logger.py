from utils import Logger
import logging
import sys
logger=Logger.get_logger()
logger.setLevel(logging.DEBUG)
logger.error("erro")
logger.info("info")
logger.debug("test..dfadfa.")