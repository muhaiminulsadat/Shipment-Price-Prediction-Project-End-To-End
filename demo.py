from src.logger import logging
from src.exception import shippingException
import sys
from src.utils.main_utils import MainUtils

obj = MainUtils()

try:
    data = obj.read_yaml_file("./config/model.yaml")
    print(f"Data read from config.yaml: {data}")
except Exception as e:
    raise shippingException(e, sys) from e
