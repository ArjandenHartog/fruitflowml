import os
import base64
import io
import logging
import traceback
import json
import uuid
import datetime

from flask import Blueprint, request, jsonify, render_template, send_from_directory, current_app, send_file
from werkzeug.utils import secure_filename
from PIL import Image

from model_utils import (
    get_model,
    preprocess_image,
    process_pil_image,
    generate_gradcam,
    create_heatmap_overlay,
)
from utils import (
    allowed_file,
    is_valid_base64,
    create_api_response,
    save_visualization
)

logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__)

@api_bp.route('/')
def home():
    return render_template('index.html')

@api_bp.route('/predict', methods=['POST'])
def predict():
    model_instance = get_model()
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        original_filename = filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{original_filename}_{timestamp}_{unique_id}"
        
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        processed_image = preprocess_image(file_path)
        prediction = model_instance.predict(processed_image)[0][0]
        
        heatmap_base64, overlay_base64 = None, None
        try:
            heatmap_raw = generate_gradcam(processed_image, model_instance)
            h_base64, o_base64, hm_pil, ov_pil = create_heatmap_overlay(file_path, heatmap_raw)
            heatmap_base64 = h_base64
            overlay_base64 = o_base64
            save_visualization(hm_pil, ov_pil, filename)
        except Exception as e:
            logger.error(f"Error generating heatmap for /predict: {str(e)}")
        
        fresh_percentage = (1 - prediction) * 100
        rotten_percentage = prediction * 100
        
        result = {
            'fresh_percentage': f"{fresh_percentage:.2f}%",
            'rotten_percentage': f"{rotten_percentage:.2f}%",
            'filename': filename,
            'original_filename': original_filename,
            'heatmap': heatmap_base64,
            'heatmap_overlay': overlay_base64
        }
        
        return jsonify(result)
    
    return jsonify({'error': 'File not allowed'})

@api_bp.route('/api/predict', methods=['POST'])
def api_predict():
    model_instance = get_model()
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
        original_filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{original_filename.rsplit('.', 1)[0]}_{timestamp}_{unique_id}.{original_filename.rsplit('.', 1)[-1]}"

        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        processed_image = preprocess_image(file_path)
        prediction = model_instance.predict(processed_image)[0][0]
        
        heatmap_base64, overlay_base64 = None, None
        try:
            heatmap_raw = generate_gradcam(processed_image, model_instance)
            h_base64, o_base64, hm_pil, ov_pil = create_heatmap_overlay(file_path, heatmap_raw)
            heatmap_base64 = h_base64
            overlay_base64 = o_base64
            save_visualization(hm_pil, ov_pil, filename)
        except Exception as e:
            logger.error(f"Error generating heatmap for /api/predict: {str(e)}")
            
        response = create_api_response(prediction, filename, heatmap_base64, overlay_base64)
        response['original_filename'] = original_filename
        
        return jsonify(response)
    
    return jsonify({
        'success': False,
        'error': 'File type not allowed'
    }), 400

@api_bp.route('/api/predict/base64', methods=['POST'])
def api_predict_base64():
    model_instance = get_model()
    logger.info("=== /api/predict/base64 - Begin request ===")
    
    if not request.is_json:
        logger.error("Fout: Geen JSON payload ontvangen")
        return jsonify({'success': False, 'error': 'JSON payload required'}), 400
    
    try:
        data = request.get_json()
    except Exception as e:
        logger.error(f"Fout bij het parsen van JSON: {str(e)}")
        return jsonify({'success': False, 'error': f'Error parsing JSON: {str(e)}'}), 400
    
    if 'image_data' not in data:
        logger.error("Fout: Geen 'image_data' veld in de JSON")
        return jsonify({'success': False, 'error': 'Missing image_data field'}), 400
    
    base64_data_full = data['image_data']
    
    if not isinstance(base64_data_full, str):
        logger.error(f"Fout: image_data is geen string maar: {type(base64_data_full).__name__}")
        return jsonify({'success': False, 'error': f'image_data is not a string but: {type(base64_data_full).__name__}'}), 400

    is_valid, validate_error = is_valid_base64(base64_data_full)
    if not is_valid:
        logger.error(f"Fout: Ongeldige base64 string: {validate_error}")
        return jsonify({'success': False, 'error': f'Invalid base64 string: {validate_error}'}), 400
    
    try:
        base64_image_data = base64_data_full.split('base64,')[-1]
        
        image_bytes = base64.b64decode(base64_image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"base64_upload_{timestamp}_{unique_id}.jpg"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        img.save(file_path)
        
        processed_image = process_pil_image(img)
        prediction = model_instance.predict(processed_image)[0][0]
        
        heatmap_base64, overlay_base64 = None, None
        try:
            heatmap_raw = generate_gradcam(processed_image, model_instance)
            h_base64, o_base64, hm_pil, ov_pil = create_heatmap_overlay(file_path, heatmap_raw)
            heatmap_base64 = h_base64
            overlay_base64 = o_base64
            save_visualization(hm_pil, ov_pil, filename)
        except Exception as e:
            logger.error(f"Fout bij genereren heatmap: {str(e)}")
        
        api_resp = create_api_response(prediction, filename, heatmap_base64, overlay_base64)

        logger.info("=== /api/predict/base64 - Einde request (succes) ===")
        return jsonify(api_resp)
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Fout bij verwerken van afbeelding: {str(e)}")
        logger.error(f"Error traceback: {error_details}")
        logger.info("=== /api/predict/base64 - Einde request (fout) ===")
        return jsonify({'success': False, 'error': f'Error processing image: {str(e)}', 'details': error_details}), 400

@api_bp.route('/api/mockpredict/base64', methods=['POST'])
def api_mock_predict_base64():
    return jsonify({
        "success": True,
        "fresh_percentage": 87.23,
        "rotten_percentage": 12.77,
        "classification": "fresh",
        "filename": "voorbeeld.jpg",
        "urls": {
            "original": f"{request.url_root.replace('http://', 'https://')}image/voorbeeld.jpg",
            "heatmap": f"{request.url_root.replace('http://', 'https://')}heatmap/voorbeeld.jpg",
            "overlay": f"{request.url_root.replace('http://', 'https://')}overlay/voorbeeld.jpg",
            "api": {
                "original": f"{request.url_root.replace('http://', 'https://')}api/images/voorbeeld.jpg",
                "heatmap": f"{request.url_root.replace('http://', 'https://')}api/heatmap/voorbeeld.jpg",
                "overlay": f"{request.url_root.replace('http://', 'https://')}api/images/overlay_voorbeeld.jpg"
            }
        },
        "heatmap_data": "mock_base64_data_for_heatmap",
        "heatmap_overlay_data": "mock_base64_data_for_overlay",
        "metadata": {
            "model_version": "mock",
            "heatmap_type": "mock",
            "image_size": "mock",
            "timestamp": datetime.datetime.now().isoformat()
        }
    })

@api_bp.route('/api/test', methods=['GET'])
def api_test():
    return jsonify({
        'success': True,
        'message': 'Apple Classifier API is running',
        'version': '1.1.0',
        'features': ['classification', 'heatmap visualization']
    })

@api_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

@api_bp.route('/api/predict/base64plain', methods=['POST'])
def api_predict_base64_plain():
    model_instance = get_model()
    logger.info("=== /api/predict/base64plain - Begin request ===")
    
    try:
        base64_data_full = request.data.decode('utf-8')
        if not base64_data_full:
            logger.error("Fout: Lege base64 string ontvangen")
            return jsonify({'success': False, 'error': 'Empty base64 string received'}), 400
        
        is_valid, validate_error = is_valid_base64(base64_data_full)
        if not is_valid:
            logger.error(f"Fout: Ongeldige base64 string: {validate_error}")
            return jsonify({'success': False, 'error': f'Invalid base64 string: {validate_error}'}), 400
        
        base64_image_data = base64_data_full.split('base64,')[-1]
        
        image_bytes = base64.b64decode(base64_image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"outsystems_upload_{timestamp}_{unique_id}.jpg"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        img.save(file_path)
        logger.info(f"Afbeelding opgeslagen als: {filename}")
        
        processed_image = process_pil_image(img)
        prediction = model_instance.predict(processed_image)[0][0]
        
        heatmap_base64, overlay_base64 = None, None
        try:
            heatmap_raw = generate_gradcam(processed_image, model_instance)
            h_base64, o_base64, hm_pil, ov_pil = create_heatmap_overlay(file_path, heatmap_raw)
            heatmap_base64 = h_base64
            overlay_base64 = o_base64
            save_visualization(hm_pil, ov_pil, filename)
        except Exception as e:
            logger.error(f"Fout bij genereren heatmap: {str(e)}")
            
        api_resp = create_api_response(prediction, filename, heatmap_base64, overlay_base64)
        
        logger.info("=== /api/predict/base64plain - Einde request (succes) ===")
        return jsonify(api_resp)
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Fout bij verwerken van afbeelding: {str(e)}")
        logger.error(f"Error traceback: {error_details}")
        logger.info("=== /api/predict/base64plain - Einde request (fout) ===")
        return jsonify({'success': False, 'error': f'Error processing image: {str(e)}', 'details': error_details}), 400

@api_bp.route('/image/<path:filename>')
def get_original_image_direct(filename):
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(file_path):
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return "Image not found", 404
    return send_file(file_path, mimetype='image/jpeg')

@api_bp.route('/heatmap/<path:filename>')
def get_heatmap_image_direct(filename):
    model_instance = get_model()
    original_image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(original_image_path):
        original_image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(original_image_path):
            logger.error(f"Original image not found for heatmap generation: {filename}")
            return "Original image not found", 404

    try:
        processed_image = preprocess_image(original_image_path)
        heatmap_raw = generate_gradcam(processed_image, model_instance)
        
        _, _, _, overlay_pil = create_heatmap_overlay(original_image_path, heatmap_raw, alpha=0.6)
        
        img_io = io.BytesIO()
        overlay_pil.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        logger.error(f"Error generating heatmap image for {filename}: {str(e)}")
        return str(e), 500

@api_bp.route('/overlay/<path:filename>')
def get_overlay_image_direct(filename):
    overlay_filename = f"overlay_{secure_filename(filename)}"
    overlay_image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], overlay_filename)

    if not os.path.exists(overlay_image_path):
        logger.warn(f"Overlay image {overlay_filename} not found. Attempting dynamic generation for {filename}.")
        return get_heatmap_image_direct(filename)

    return send_file(overlay_image_path, mimetype='image/jpeg')

@api_bp.route('/api/images/<filename>')
def get_image_api(filename):
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(file_path):
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Image not found'}), 404
            
    response = send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@api_bp.route('/api/heatmap/<path:filename>')
def get_heatmap_data_api(filename):
    model_instance = get_model()
    original_image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(original_image_path):
        original_image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(original_image_path):
            logger.error(f"Image not found for /api/heatmap: {filename}")
            return jsonify({'success': False, 'error': 'Image not found'}), 404
            
    try:
        processed_image = preprocess_image(original_image_path)
        heatmap_raw = generate_gradcam(processed_image, model_instance)
        
        heatmap_b64, overlay_b64, hm_pil, ov_pil = create_heatmap_overlay(original_image_path, heatmap_raw)
        
        saved_heatmap_fname, saved_overlay_fname = save_visualization(hm_pil, ov_pil, filename)
        
        base_url = request.url_root.replace("http://", "https://")

        img_width = current_app.config.get('IMG_WIDTH', 224)
        img_height = current_app.config.get('IMG_HEIGHT', 224)

        return jsonify({
            'success': True,
            'filename': filename,
            'urls': {
                'original': f"{base_url}image/{filename}",
                'heatmap_image_direct': f"{base_url}heatmap/{filename}",
                'overlay_image_direct': f"{base_url}overlay/{filename}",
                'api': {
                    'original': f"{base_url}api/images/{filename}",
                    'heatmap_file': f"{base_url}api/images/{saved_heatmap_fname}",
                    'overlay_file': f"{base_url}api/images/{saved_overlay_fname}"
                }
            },
            'heatmap_data': heatmap_b64,
            'heatmap_overlay_data': overlay_b64,
            'metadata': {
                'model_version': current_app.config.get('MODEL_VERSION', '1.0'),
                'heatmap_type': 'Grad-CAM',
                'image_size': f"{img_width}x{img_height}",
                'timestamp': datetime.datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Error in /api/heatmap/{filename}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500 