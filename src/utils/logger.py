# src/utils/logger.py

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Define the logs directory relative to the project root
LOGS_DIR = Path(__file__).resolve().parent.parent.parent / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

# Define the log file path
LOG_FILE = LOGS_DIR / 'app.log'

# Create a custom logger
logger = logging.getLogger('project_logger')
logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all types of logs

# Prevent adding multiple handlers if logger is imported multiple times
if not logger.handlers:
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Console handler set to INFO level

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)  # File handler set to DEBUG level

    # Create formatters
    console_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Assign formatters to handlers
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Optionally, disable propagation to prevent duplicate logs in certain configurations
logger.propagate = False
