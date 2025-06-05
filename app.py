import os
import logging
from flask import Flask
from flask_cors import CORS

from api import api_bp
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure app from config.py
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['IMG_WIDTH'] = config.IMG_WIDTH
app.config['IMG_HEIGHT'] = config.IMG_HEIGHT
app.config['MODEL_VERSION'] = config.MODEL_VERSION
app.config['ALLOWED_EXTENSIONS'] = config.ALLOWED_EXTENSIONS

# Register blueprint
app.register_blueprint(api_bp)

if __name__ == '__main__':
    logger.info("Starting Apple Classifier API...")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG) 