#!/usr/bin/env python3
"""
VitalSense Backend Startup Script
Medical Report Analysis System
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vitalsense.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'flask',
        'chromadb', 
        'sentence_transformers',
        'spacy',
        'google.generativeai',
        'pymupdf',
        'pytesseract',
        'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'google.generativeai':
                import google.generativeai
            else:
                __import__(package)
            logger.info(f"✓ {package} - OK")
        except ImportError as e:
            logger.error(f"✗ Missing dependency: {e}")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error("Please install all dependencies: pip install -r requirements.txt")
        return False
    
    return True

def check_environment():
    """Check if required environment variables are set."""
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key:
        logger.warning("GOOGLE_API_KEY not found in environment variables")
        logger.info("You can set it in a .env file or as an environment variable")
        logger.info("The system will work but AI features will be limited")
    else:
        logger.info("✓ GOOGLE_API_KEY - OK")

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['uploads', 'chroma_db', 'logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory exists: {directory}")

def main():
    """Main function to start the VitalSense system."""
    logger.info("Starting VitalSense Medical Report Analysis System")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    check_environment()
    
    # Create directories
    create_directories()
    
    # Import and start the Flask app
    try:
        from app import app
        logger.info("Flask application imported successfully")
        logger.info("Starting server on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)
    except Exception as e:
        logger.error(f"Failed to start Flask application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
