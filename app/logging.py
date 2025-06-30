import logging
import os
from elasticsearch import Elasticsearch
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class ElasticsearchHandler(logging.Handler):
    """
    Custom logging handler to send logs to Elasticsearch.
    """
    def __init__(self, es_host, es_user, es_pass, index_name):
        super().__init__()
        # Initialize Elasticsearch client with provided credentials
        self.es = Elasticsearch(
            [es_host],
            basic_auth=(es_user, es_pass),
            verify_certs=False,      # Disable SSL certificate verification
            ssl_show_warn=False      # Suppress SSL warnings
        )
        self.index_name = index_name

    def emit(self, record):
        """
        Send a log record to Elasticsearch.
        """
        try:
            # Prepare log entry as a dictionary
            log_entry = {
                "@timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "name": record.name,
                "filename": record.pathname,
                "line": record.lineno,
            }
            # Index the log entry in Elasticsearch
            self.es.index(index=self.index_name, body=log_entry)
        except Exception as e:
            # Print error if logging to Elasticsearch fails
            print(f"Error sending log to Elasticsearch: {e}")

def setup_logger(
    name: str = "api_logger",
    log_file: str = "logs/api.log",
    es_host: str = None,
    es_user: str = None,
    es_pass: str = None,
    es_index_name: str = "logs",
) -> logging.Logger:
    """
    Set up a logger with console, file, and Elasticsearch handlers.
    """
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console handler for stdout
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # File handler for writing logs to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        # Read Elasticsearch credentials from environment if not provided
        es_host = es_host or os.getenv("ES_HOST")
        es_user = es_user or os.getenv("ES_USER")
        es_pass = es_pass or os.getenv("ES_PASS")

        # Elasticsearch handler for sending logs to Elasticsearch
        es_handler = ElasticsearchHandler(es_host, es_user, es_pass, es_index_name)

        # Add all handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(es_handler)

    return logger