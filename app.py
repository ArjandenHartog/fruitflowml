import os
import numpy as np
import base64
import io
import logging
import traceback
import json
from flask import Flask, request, render_template, jsonify, send_from_directory, g, send_file
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from PIL import Image
import uuid
import datetime
import cv2
import matplotlib.pyplot as plt

# Configureer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes to allow OutSystems to call the API

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'apple_classifier_model.h5'

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Load the trained model only once at startup
model = load_model(MODEL_PATH)

# Constants for image preprocessing
IMG_WIDTH, IMG_HEIGHT = 224, 224

# Function to generate Grad-CAM heatmap
def generate_gradcam(img_array, model, last_conv_layer_name="conv2d_3"):
    """
    Generate Grad-CAM heatmap for an image
    
    Args:
        img_array: Preprocessed image (normalized, expanded dims)
        model: The trained model
        last_conv_layer_name: Name of the last convolutional layer in the model
    
    Returns:
        Heatmap array and superimposed image with heatmap
    """
    try:
        # Create a model that maps the input image to the activations
        # of the last conv layer and the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute the gradient of the top predicted class with respect to the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        # Get the gradients of the top predicted class with respect to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # Vector of mean intensity of the gradient over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight output feature map with gradient values
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        return heatmap
    except Exception as e:
        logger.error(f"Error in generate_gradcam: {str(e)}")
        # Return a default heatmap if generation fails
        return np.zeros((7, 7))  # Small default heatmap

# Function to overlay heatmap on original image and return both
def create_heatmap_overlay(img_path, heatmap, alpha=0.6):
    """
    Create a superimposed heatmap on the original image and save both
    
    Args:
        img_path: Path to the original image
        heatmap: Generated heatmap array
        alpha: Transparency factor (increased for better visibility)
        
    Returns:
        Base64 encoded heatmap image and overlay image, and filenames
    """
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    
    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (width, height))
    
    # Enhance the heatmap contrast
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    
    # Apply a more vibrant colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Enhance the heatmap intensity
    heatmap = np.float32(heatmap) * 1.2  # Increase intensity by 20%
    heatmap = np.clip(heatmap, 0, 255)
    heatmap = np.uint8(heatmap)
    
    # Superimpose the heatmap on original image with increased alpha
    superimposed_img = heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255)  # Ensure values stay in valid range
    superimposed_img = np.uint8(superimposed_img)
    
    # Convert both images to PIL Images
    heatmap_img = Image.fromarray(heatmap)
    overlay_img = Image.fromarray(superimposed_img)
    
    # Save images and get filenames
    filename = os.path.basename(img_path)
    heatmap_filename, overlay_filename = save_visualization(heatmap_img, overlay_img, filename)
    
    # Convert to base64 for response
    heatmap_buffer = io.BytesIO()
    overlay_buffer = io.BytesIO()
    
    heatmap_img.save(heatmap_buffer, format="PNG")
    overlay_img.save(overlay_buffer, format="PNG")
    
    heatmap_base64 = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
    overlay_base64 = base64.b64encode(overlay_buffer.getvalue()).decode('utf-8')
    
    return heatmap_base64, overlay_base64

# Create PIL image heatmap
def create_pil_heatmap(pil_img, heatmap, alpha=0.4):
    """
    Create heatmap overlay from PIL Image without saving to disk
    
    Args:
        pil_img: PIL Image object
        heatmap: Generated heatmap array
        alpha: Transparency factor
        
    Returns:
        Base64 encoded heatmap image and overlay image
    """
    # Convert PIL image to numpy array
    img = np.array(pil_img)
    height, width, _ = img.shape
    
    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (width, height))
    
    # Convert heatmap to RGB format
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.uint8(superimposed_img)
    
    # Convert both images to base64
    heatmap_img = Image.fromarray(heatmap)
    overlay_img = Image.fromarray(superimposed_img)
    
    # Save to buffer and convert to base64
    heatmap_buffer = io.BytesIO()
    overlay_buffer = io.BytesIO()
    
    heatmap_img.save(heatmap_buffer, format="PNG")
    overlay_img.save(overlay_buffer, format="PNG")
    
    heatmap_base64 = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
    overlay_base64 = base64.b64encode(overlay_buffer.getvalue()).decode('utf-8')
    
    return heatmap_base64, overlay_base64

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize to [0,1]

# Process image from a PIL Image object
def process_pil_image(pil_img):
    pil_img = pil_img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(pil_img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Helper functie om te controleren of een string een geldige base64 is
def is_valid_base64(s):
    try:
        # Check of het een string is
        if not isinstance(s, str):
            return False, "Input is geen string"
        
        # Verwijder data URL prefix als die aanwezig is
        if 'base64,' in s:
            s = s.split('base64,')[1]
        
        # Check of de string niet leeg is
        if not s:
            return False, "Base64 string is leeg"
        
        # Valideer de base64 string door te proberen te decoderen
        base64.b64decode(s)
        return True, None
    except Exception as e:
        return False, str(e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Make prediction
        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image)[0][0]
        
        # Generate heatmap
        try:
            heatmap = generate_gradcam(processed_image, model)
            heatmap_base64, overlay_base64 = create_heatmap_overlay(file_path, heatmap)
        except Exception as e:
            logger.error(f"Error generating heatmap: {str(e)}")
            heatmap_base64, overlay_base64 = None, None
        
        # Convert prediction to percentage
        # In our model, 0 = fresh, 1 = rotten (based on alphabetical order)
        fresh_percentage = (1 - prediction) * 100
        rotten_percentage = prediction * 100
        
        result = {
            'fresh_percentage': f"{fresh_percentage:.2f}%",
            'rotten_percentage': f"{rotten_percentage:.2f}%",
            'filename': filename,
            'heatmap': heatmap_base64,
            'heatmap_overlay': overlay_base64
        }
        
        return jsonify(result)
    
    return jsonify({'error': 'File not allowed'})

# API endpoints for OutSystems integration

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint that accepts image file uploads from OutSystems.
    """
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file uploaded'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty filename'
        }), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename while preserving the original name
        original_filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{original_filename}_{timestamp}_{unique_id}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Make prediction
        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image)[0][0]
        
        # Generate heatmap
        try:
            heatmap = generate_gradcam(processed_image, model)
            heatmap_base64, overlay_base64 = create_heatmap_overlay(file_path, heatmap)
        except Exception as e:
            logger.error(f"Error generating heatmap: {str(e)}")
            heatmap_base64, overlay_base64 = None, None
        
        response = create_api_response(prediction, filename, heatmap_base64, overlay_base64)
        # Add original filename to response
        response['original_filename'] = original_filename
        
        return jsonify(response)
    
    return jsonify({
        'success': False,
        'error': 'File type not allowed'
    }), 400

@app.route('/api/predict/base64', methods=['POST'])
def api_predict_base64():
    """
    API endpoint that accepts base64 encoded images from OutSystems.
    
    Expected JSON payload: { "image_data": "base64_encoded_string" }
    Returns a JSON response with prediction results.
    """
    logger.info("=== /api/predict/base64 - Begin request ===")
    
    # Log de request headers
    logger.info(f"Headers: {dict(request.headers)}")
    
    # Check of het een JSON request is
    if not request.is_json:
        logger.error("Fout: Geen JSON payload ontvangen")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Request data: {request.data[:200]}...")  # Log eerste 200 tekens
        return jsonify({
            'success': False,
            'error': 'JSON payload required'
        }), 400
    
    # Haal de JSON data op
    try:
        data = request.get_json()
        logger.info(f"JSON data keys: {list(data.keys())}")
    except Exception as e:
        logger.error(f"Fout bij het parsen van JSON: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error parsing JSON: {str(e)}'
        }), 400
    
    # Check of image_data aanwezig is
    if 'image_data' not in data:
        logger.error("Fout: Geen 'image_data' veld in de JSON")
        return jsonify({
            'success': False,
            'error': 'Missing image_data field'
        }), 400
    
    # Haal de base64 string op
    base64_data = data['image_data']
    
    # Log de lengte en eerste/laatste karakters van de base64 string
    if isinstance(base64_data, str):
        base64_length = len(base64_data)
        logger.info(f"Base64 string lengte: {base64_length} tekens")
        if base64_length > 0:
            logger.info(f"Base64 string begin: {base64_data[:20]}...")
            logger.info(f"Base64 string einde: ...{base64_data[-20:]}")
        else:
            logger.error("Fout: Base64 string is leeg")
    else:
        logger.error(f"Fout: image_data is geen string maar: {type(base64_data).__name__}")
        return jsonify({
            'success': False,
            'error': f'image_data is not a string but: {type(base64_data).__name__}'
        }), 400
    
    # Valideer de base64 string
    is_valid, validate_error = is_valid_base64(base64_data)
    if not is_valid:
        logger.error(f"Fout: Ongeldige base64 string: {validate_error}")
        return jsonify({
            'success': False,
            'error': f'Invalid base64 string: {validate_error}'
        }), 400
    
    try:
        # Verwijder de data URL prefix als die aanwezig is
        if 'base64,' in base64_data:
            logger.info("Data URL prefix gevonden en verwijderd")
            base64_data = base64_data.split('base64,')[1]
        
        # Decodeer de base64 string
        logger.info("Base64 string decoderen...")
        image_data = base64.b64decode(base64_data)
        logger.info(f"Gedecodeerde data lengte: {len(image_data)} bytes")
        
        # Open de afbeelding met PIL
        logger.info("Afbeelding openen met PIL...")
        img = Image.open(io.BytesIO(image_data))
        logger.info(f"Afbeelding info: mode={img.mode}, size={img.size}")
        
        # Convert naar RGB indien nodig
        if img.mode != 'RGB':
            logger.info(f"Afbeelding mode {img.mode} omzetten naar RGB")
            img = img.convert('RGB')
        
        # Genereer een unieke bestandsnaam
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"base64_upload_{timestamp}_{str(uuid.uuid4())[:8]}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Sla de afbeelding op (nodig voor heatmap generatie)
        img.save(file_path)
        
        # Verwerk de afbeelding en maak voorspelling
        logger.info("Afbeelding verwerken voor model...")
        processed_image = process_pil_image(img)
        
        # Generate heatmap
        try:
            logger.info("Heatmap genereren...")
            heatmap = generate_gradcam(processed_image, model)
            heatmap_base64, overlay_base64 = create_heatmap_overlay(file_path, heatmap)
            logger.info("Heatmap succesvol gegenereerd")
        except Exception as e:
            logger.error(f"Fout bij genereren heatmap: {str(e)}")
            heatmap_base64, overlay_base64 = None, None
        
        logger.info("Voorspelling maken...")
        prediction = model.predict(processed_image)[0][0]
        logger.info(f"Raw voorspelling: {prediction}")
        
        # Zet voorspelling om naar percentages
        fresh_percentage = float((1 - prediction) * 100)
        rotten_percentage = float(prediction * 100)
        
        # Bepaal classificatie
        classification = "fresh" if fresh_percentage > rotten_percentage else "rotten"
        
        # Create image URL
        image_url = f"{request.url_root}uploads/{filename}".replace("http://", "https://")
        
        result = {
            'success': True,
            'fresh_percentage': round(fresh_percentage, 2),
            'rotten_percentage': round(rotten_percentage, 2),
            'classification': classification,
            'filename': filename,
            'image_url': image_url,
            'heatmap_data': heatmap_base64,
            'heatmap_overlay_data': overlay_base64
        }
        
        logger.info(f"Resultaat: {json.dumps({k: '...' for k in result.keys()})}")
        logger.info("=== /api/predict/base64 - Einde request (succes) ===")
        return jsonify(result)
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Fout bij verwerken van afbeelding: {str(e)}")
        logger.error(f"Error traceback: {error_details}")
        logger.info("=== /api/predict/base64 - Einde request (fout) ===")
        return jsonify({
            'success': False,
            'error': f'Error processing image: {str(e)}',
            'details': error_details
        }), 400

@app.route('/api/mockpredict/base64', methods=['POST'])
def api_mock_predict_base64():
    """
    Mock endpoint voor OutSystems mapping. Geeft altijd een geldige, vaste response terug.
    """
    return jsonify({
        "success": True,
        "fresh_percentage": 87.23,
        "rotten_percentage": 12.77,
        "classification": "fresh",
        "filename": "voorbeeld.jpg",
        "heatmap_data": "mock_base64_data_for_heatmap",
        "heatmap_overlay_data": "mock_base64_data_for_overlay"
    })

@app.route('/api/test', methods=['GET'])
def api_test():
    """
    Simple endpoint to test if the API is running.
    """
    return jsonify({
        'success': True,
        'message': 'Apple Classifier API is running',
        'version': '1.1.0',
        'features': ['classification', 'heatmap visualization']
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded images for display purposes.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/predict/base64plain', methods=['POST'])
def api_predict_base64_plain():
    """
    API endpoint that accepts plain base64 encoded image text (not JSON).
    This makes it easier for OutSystems integration when it can't easily send JSON.
    
    Expected: The request body should contain ONLY the base64 encoded image string
    Returns: A JSON response with prediction results
    """
    logger.info("=== /api/predict/base64plain - Begin request ===")
    
    # Log de request headers
    logger.info(f"Headers: {dict(request.headers)}")
    
    try:
        # Haal de request data op als text
        base64_data = request.data.decode('utf-8')
        logger.info(f"Ontvangen data lengte: {len(base64_data)} tekens")
        
        if len(base64_data) == 0:
            logger.error("Fout: Lege base64 string ontvangen")
            return jsonify({
                'success': False,
                'error': 'Empty base64 string received'
            }), 400
        
        # Toon eerste en laatste 20 karakters voor debugging
        logger.info(f"Data begin: {base64_data[:20]}...")
        logger.info(f"Data einde: ...{base64_data[-20:]}")
        
        # Valideer de base64 string
        is_valid, validate_error = is_valid_base64(base64_data)
        if not is_valid:
            logger.error(f"Fout: Ongeldige base64 string: {validate_error}")
            return jsonify({
                'success': False,
                'error': f'Invalid base64 string: {validate_error}'
            }), 400
        
        # Verwijder de data URL prefix als die aanwezig is
        if 'base64,' in base64_data:
            logger.info("Data URL prefix gevonden en verwijderd")
            base64_data = base64_data.split('base64,')[1]
        
        # Decodeer de base64 string
        logger.info("Base64 string decoderen...")
        image_data = base64.b64decode(base64_data)
        logger.info(f"Gedecodeerde data lengte: {len(image_data)} bytes")
        
        # Open de afbeelding met PIL
        logger.info("Afbeelding openen met PIL...")
        img = Image.open(io.BytesIO(image_data))
        logger.info(f"Afbeelding info: mode={img.mode}, size={img.size}")
        
        # Convert naar RGB indien nodig
        if img.mode != 'RGB':
            logger.info(f"Afbeelding mode {img.mode} omzetten naar RGB")
            img = img.convert('RGB')
        
        # Genereer een unieke bestandsnaam en sla de afbeelding op
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outsystems_upload_{timestamp}_{str(uuid.uuid4())[:8]}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Sla de afbeelding op in het uploads folder
        img_copy = img.copy()
        img_copy.save(file_path)
        logger.info(f"Afbeelding opgeslagen als: {filename}")
        
        # Verwerk de afbeelding en maak voorspelling
        logger.info("Afbeelding verwerken voor model...")
        processed_image = process_pil_image(img)
        
        # Generate heatmap
        try:
            logger.info("Heatmap genereren...")
            heatmap = generate_gradcam(processed_image, model)
            heatmap_base64, overlay_base64 = create_heatmap_overlay(file_path, heatmap)
            logger.info("Heatmap succesvol gegenereerd")
        except Exception as e:
            logger.error(f"Fout bij genereren heatmap: {str(e)}")
            heatmap_base64, overlay_base64 = None, None
        
        logger.info("Voorspelling maken...")
        prediction = model.predict(processed_image)[0][0]
        logger.info(f"Raw voorspelling: {prediction}")
        
        # Zet voorspelling om naar percentages
        fresh_percentage = float((1 - prediction) * 100)
        rotten_percentage = float(prediction * 100)
        
        # Bepaal classificatie
        classification = "fresh" if fresh_percentage > rotten_percentage else "rotten"
        
        # Genereer de volledige URL naar de afbeelding
        image_url = f"{request.url_root}uploads/{filename}".replace("http://", "https://")
        
        result = {
            'success': True,
            'fresh_percentage': round(fresh_percentage, 2),
            'rotten_percentage': round(rotten_percentage, 2),
            'classification': classification,
            'filename': filename,
            'image_url': image_url,
            'heatmap_data': heatmap_base64,
            'heatmap_overlay_data': overlay_base64
        }
        
        logger.info(f"Resultaat: {json.dumps({k: '...' for k in result.keys()})}")
        logger.info("=== /api/predict/base64plain - Einde request (succes) ===")
        return jsonify(result)
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Fout bij verwerken van afbeelding: {str(e)}")
        logger.error(f"Error traceback: {error_details}")
        logger.info("=== /api/predict/base64plain - Einde request (fout) ===")
        return jsonify({
            'success': False,
            'error': f'Error processing image: {str(e)}',
            'details': error_details
        }), 400

@app.route('/api/images/<filename>')
def get_image(filename):
    """
    Serve images with proper headers for OutSystems.
    """
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/heatmap/<path:filename>')
def get_heatmap(filename):
    """
    Generate and return heatmap visualization for a specific image.
    """
    try:
        # Try different possible file locations
        possible_paths = [
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if file_path is None:
            logger.error(f"Image not found. Tried paths: {possible_paths}")
            return jsonify({
                'success': False,
                'error': 'Image not found',
                'tried_paths': possible_paths
            }), 404

        # Process image and generate heatmap
        processed_image = preprocess_image(file_path)
        heatmap = generate_gradcam(processed_image, model)
        heatmap_base64, overlay_base64 = create_heatmap_overlay(file_path, heatmap)

        return jsonify({
            'success': True,
            'filename': filename,
            'heatmap_url': f"{request.url_root}api/images/heatmap_{filename}".replace("http://", "https://"),
            'overlay_url': f"{request.url_root}api/images/overlay_{filename}".replace("http://", "https://"),
            'heatmap_data': heatmap_base64,
            'heatmap_overlay_data': overlay_base64,
            'metadata': {
                'model_version': '1.0',
                'heatmap_type': 'Grad-CAM',
                'image_size': f"{IMG_WIDTH}x{IMG_HEIGHT}",
                'timestamp': datetime.datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Error generating heatmap: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def save_visualization(heatmap_img, overlay_img, filename):
    """
    Save heatmap and overlay images to disk
    """
    heatmap_filename = f"heatmap_{filename}"
    overlay_filename = f"overlay_{filename}"
    
    heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
    overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
    
    heatmap_img.save(heatmap_path)
    overlay_img.save(overlay_path)
    
    return heatmap_filename, overlay_filename

# Update the API response structure in existing endpoints
def create_api_response(prediction, filename, heatmap_base64, overlay_base64):
    """
    Create a standardized API response with all necessary URLs and data
    """
    fresh_percentage = float((1 - prediction) * 100)
    rotten_percentage = float(prediction * 100)
    classification = "fresh" if fresh_percentage > rotten_percentage else "rotten"
    
    base_url = request.url_root.replace("http://", "https://")
    
    return {
        'success': True,
        'fresh_percentage': round(fresh_percentage, 2),
        'rotten_percentage': round(rotten_percentage, 2),
        'classification': classification,
        'filename': filename,
        'urls': {
            'original': f"{base_url}image/{filename}",
            'heatmap': f"{base_url}heatmap/{filename}",
            'overlay': f"{base_url}overlay/{filename}",
            'api': {
                'original': f"{base_url}api/images/{filename}",
                'heatmap': f"{base_url}api/heatmap/{filename}",
                'overlay': f"{base_url}api/images/overlay_{filename}"
            }
        },
        'heatmap_data': heatmap_base64,
        'heatmap_overlay_data': overlay_base64,
        'metadata': {
            'model_version': '1.0',
            'heatmap_type': 'Grad-CAM',
            'image_size': f"{IMG_WIDTH}x{IMG_HEIGHT}",
            'timestamp': datetime.datetime.now().isoformat()
        }
    }

@app.route('/heatmap/<path:filename>')
def get_heatmap_image(filename):
    """
    Return the heatmap image directly for browser viewing
    """
    try:
        # Try different possible file locations
        possible_paths = [
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if file_path is None:
            return "Image not found", 404

        # Process image and generate heatmap
        processed_image = preprocess_image(file_path)
        heatmap = generate_gradcam(processed_image, model)
        
        # Create the heatmap overlay with increased visibility
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        
        # Resize the heatmap to match the original image size
        heatmap = cv2.resize(heatmap, (width, height))
        
        # Enhance the heatmap contrast
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        heatmap = np.uint8(255 * heatmap)
        
        # Apply a more vibrant colormap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Enhance the heatmap intensity
        heatmap = np.float32(heatmap) * 1.2  # Increase intensity by 20%
        heatmap = np.clip(heatmap, 0, 255)
        heatmap = np.uint8(heatmap)
        
        # Superimpose the heatmap on original image with increased alpha
        alpha = 0.6  # Increased from 0.4 to 0.6
        superimposed_img = heatmap * alpha + img * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 255)
        superimposed_img = np.uint8(superimposed_img)
        
        # Convert to PIL Image
        overlay_img = Image.fromarray(superimposed_img)
        
        # Save to bytes
        img_io = io.BytesIO()
        overlay_img.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Error generating heatmap image: {str(e)}")
        return str(e), 500

@app.route('/overlay/<path:filename>')
def get_overlay_image(filename):
    """
    Return the overlay image directly for browser viewing
    """
    try:
        overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], f"overlay_{filename}")
        if os.path.exists(overlay_path):
            return send_file(overlay_path, mimetype='image/jpeg')
        else:
            # If overlay doesn't exist, generate it on the fly
            return get_heatmap_image(filename)
    except Exception as e:
        logger.error(f"Error serving overlay image: {str(e)}")
        return str(e), 500

@app.route('/image/<path:filename>')
def get_original_image(filename):
    """
    Return the original image directly for browser viewing
    """
    try:
        # Try different possible file locations
        possible_paths = [
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if file_path is None:
            return "Image not found", 404

        return send_file(file_path, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error serving original image: {str(e)}")
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 