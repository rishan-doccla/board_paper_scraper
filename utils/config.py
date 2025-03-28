import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Flask configuration
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('DEBUG', 'True') == 'True'
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key')
    
    # Crawl4AI settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Simulation mode - use when no valid API key is available
    SIMULATE_CRAWLER = os.getenv('SIMULATE_CRAWLER', 'False').lower() in ('true', '1', 't')
    
    # Data storage
    DATA_DIR = os.getenv('DATA_DIR', 'data')
    
    # Crawler settings
    MAX_PAGES_PER_SITE = int(os.getenv('MAX_PAGES_PER_SITE', '30'))
    CRAWL_DELAY = float(os.getenv('CRAWL_DELAY', '1.0'))  # seconds 