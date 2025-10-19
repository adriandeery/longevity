# longevity/logging/logger.py
import logging
import sys
from pathlib import Path
from datetime import datetime

# Create logs directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Create log file with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE

# Configure logging format
logging_format = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"

# Create logger
logger = logging.getLogger("longevity")
logger.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(logging_format))

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(logging_format))

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)
