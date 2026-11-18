import logging
import os
from datetime import datetime

# Create logs directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name
LOG_FILE = datetime.now().strftime("%m_%d_%Y_%H_%M_%S.log")

# Full log file path
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="a",
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    encoding="utf-8"
)

# Create logger object
logger = logging.getLogger("Customer_Churn_Prediction")
logger.setLevel(logging.INFO)