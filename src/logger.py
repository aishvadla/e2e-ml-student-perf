"""Logging configuration for the ML project.

This module sets up logging infrastructure to capture and persist
runtime events, warnings, and error information to timestamped log files.
"""

import logging
import os
from datetime import datetime

# Generate a unique log file name based on current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create logs directory if it does not exist
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging to write to file with timestamp, logger name, level, and message
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
