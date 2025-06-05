import os
import base64
import datetime
from flask import request, current_app
from PIL import Image
import uuid
import config

# Use configuration from config.py
ALLOWED_EXTENSIONS = config.ALLOWED_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_base64(s):
    try:
        if not isinstance(s, str):
            return False, "Input is geen string"
        
        if 'base64,' in s:
            s = s.split('base64,')[1]
        
        if not s:
            return False, "Base64 string is leeg"
        
        base64.b64decode(s)
        return True, None
    except Exception as e:
        return False, str(e)

def save_visualization(heatmap_img, overlay_img, filename):
    """
    Save heatmap and overlay images to disk
    """
    upload_folder = current_app.config['UPLOAD_FOLDER']
    heatmap_filename = f"heatmap_{filename}"
    overlay_filename = f"overlay_{filename}"
    
    heatmap_path = os.path.join(upload_folder, heatmap_filename)
    overlay_path = os.path.join(upload_folder, overlay_filename)
    
    heatmap_img.save(heatmap_path)
    overlay_img.save(overlay_path)
    
    return heatmap_filename, overlay_filename

def create_api_response(prediction, filename, heatmap_base64, overlay_base64):
    """
    Create a standardized API response with all necessary URLs and data
    """
    fresh_percentage = float((1 - prediction) * 100)
    rotten_percentage = float(prediction * 100)
    classification = "fresh" if fresh_percentage > rotten_percentage else "rotten"
    
    # Need to access IMG_WIDTH, IMG_HEIGHT from model_utils or config
    # For now, let's assume they are available via current_app.config or a dedicated config module
    # If they are fixed, we can define them here, or better, pass them as arguments or get from a config object.
    # For this refactoring step, we'll assume they might come from current_app.config if set there,
    # or have to be imported if they become part of model_utils.py
    # Let's assume IMG_WIDTH, IMG_HEIGHT are part of a config or model_utils for now.
    # We'll need to resolve this dependency. For now, I'll use placeholder values or expect them from config.
    img_width = current_app.config.get('IMG_WIDTH', 224)
    img_height = current_app.config.get('IMG_HEIGHT', 224)

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
            'model_version': '1.0', # This could also be a config value
            'heatmap_type': 'Grad-CAM',
            'image_size': f"{img_width}x{img_height}",
            'timestamp': datetime.datetime.now().isoformat()
        }
    } 