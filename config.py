import os
import logging

# App configuration
FLASK_PORT = int(os.getenv("FLASK_PORT", 8000))  # Change default port to 8000
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,  # Temporarily set to DEBUG to capture all logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if logger.hasHandlers():
    logger.handlers.clear()

# Filter out noisy loggers
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

logger.info("Logger initialized successfully!")  # Confirm that logger works

# File paths
ABBREVIATION_DICT_PATH = 'kamus_singkatan.csv'
CLEANED_REVIEW_PATH = "cleaned_review.csv"
WORDCLOUD_OUTPUT_FOLDER = os.path.join('static', 'wordclouds')

# Template directories
TEMPLATE_DIR = os.path.join(os.getcwd(), 'templates')

# LDA Model parameters
LDA_NUM_TOPICS_RANGE = range(2, 6)
LDA_PASSES = 20

# Ensure required directories exist
os.makedirs(WORDCLOUD_OUTPUT_FOLDER, exist_ok=True)