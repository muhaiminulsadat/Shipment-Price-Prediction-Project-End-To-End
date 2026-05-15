from src.logger import logging
from src.exception import shippingException
import sys

try:
    a = 1 / 0
except Exception as e:
    logging.error(e)
    raise shippingException(e, sys) from e
