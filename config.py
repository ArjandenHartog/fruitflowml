import os

# Basic configuration
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
DEBUG = True
HOST = '0.0.0.0'
PORT = 5001

# Model configuration
MODEL_PATH = 'apple_classifier_model.h5'
MODEL_VERSION = '1.0'
IMG_WIDTH = 224
IMG_HEIGHT = 224

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO' 