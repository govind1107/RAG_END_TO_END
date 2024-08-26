import logging
import os

# Define the path to the logs directory
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')

# Create the logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define the path to the log file
log_file = os.path.join(log_dir, 'app.log')

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Create a logger instance
app_logger = logging.getLogger('RAGAppLogger')
