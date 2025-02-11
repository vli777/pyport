# src/utils/logger.py

from concurrent_log_handler import ConcurrentRotatingFileHandler
from pathlib import Path
import logging

# Define the logs directory
LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "app.log"

# Create a logger
logger = logging.getLogger("project_logger")
logger.setLevel(logging.DEBUG)

# Prevent duplicate handlers
if not logger.handlers:
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Multi-process safe Rotating File Handler
    file_handler = ConcurrentRotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)

    # Formatters
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add Handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

logger.propagate = False
