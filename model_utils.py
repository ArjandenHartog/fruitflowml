import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
import logging
from utils import save_visualization # Import from our new utils.py
import config

logger = logging.getLogger(__name__)

# Use configuration from config.py
MODEL_PATH = config.MODEL_PATH
IMG_WIDTH, IMG_HEIGHT = config.IMG_WIDTH, config.IMG_HEIGHT

# Initialize model globally within this module so it's loaded once
model = None

def get_model():
    global model
    if model is None:
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
    return model

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def process_pil_image(pil_img):
    pil_img = pil_img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(pil_img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def generate_gradcam(img_array, model_instance, last_conv_layer_name="conv2d_3"):
    try:
        grad_model = tf.keras.models.Model(
            [model_instance.inputs], 
            [model_instance.get_layer(last_conv_layer_name).output, model_instance.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8) # Added epsilon for stability
        heatmap = heatmap.numpy()
        
        return heatmap
    except Exception as e:
        logger.error(f"Error in generate_gradcam: {str(e)}")
        # Return a small, valid heatmap shape to avoid downstream errors
        # The original code returned np.zeros((7,7)) which might be specific to a layer output.
        # Let's make it more generic or ensure it matches expected downstream dimensions if possible.
        # For now, using a placeholder that resize can handle.
        return np.zeros((IMG_HEIGHT // 32, IMG_WIDTH // 32)) # Example based on typical conv net downsampling

def create_heatmap_overlay(img_path, heatmap, alpha=0.6): # img_path is the original image path
    # Ensure model is loaded if not already
    # model_instance = get_model() # Not directly needed here, but good practice if other model ops were here

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    
    heatmap = cv2.resize(heatmap, (width, height))
    
    heatmap = np.maximum(heatmap, 0)
    # Normalize heatmap carefully to avoid division by zero if max is same as min (e.g. all zeros)
    heatmap_min = np.min(heatmap)
    heatmap_max = np.max(heatmap)
    if heatmap_max - heatmap_min < 1e-8: # if heatmap is flat
        heatmap = np.zeros_like(heatmap)
    else:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    heatmap = np.uint8(255 * heatmap)
    
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Adjust intensity - make sure it's float for multiplication
    heatmap_colored = np.float32(heatmap_colored) * 1.2 
    heatmap_colored = np.clip(heatmap_colored, 0, 255)
    heatmap_colored = np.uint8(heatmap_colored)
    
    superimposed_img = heatmap_colored * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255)
    superimposed_img = np.uint8(superimposed_img)
    
    heatmap_pil_img = Image.fromarray(heatmap_colored) # Use the colored heatmap for "heatmap_img"
    overlay_pil_img = Image.fromarray(superimposed_img)
    
    filename = os.path.basename(img_path)
    # save_visualization is now in utils.py and expects current_app for UPLOAD_FOLDER
    # This function create_heatmap_overlay might be called from contexts without flask app context
    # For now, let's assume this will be refactored, or UPLOAD_FOLDER needs to be passed.
    # Temporarily, this will break if called outside app context if save_visualization isn't adapted.
    # To proceed with refactoring, I will keep the call and address app context dependency later
    # or ensure that save_visualization is adapted or this function is only called from within app context.
    
    # The original save_visualization was in app.py and used app.config.
    # The moved version in utils.py uses current_app.config.
    # This function create_heatmap_overlay needs to be callable.
    # For now, we will return PIL images and let the caller handle saving and base64 encoding.
    # This makes model_utils.py more independent of Flask.

    # heatmap_filename, overlay_filename = save_visualization(heatmap_pil_img, overlay_pil_img, filename) # This line requires app context
    
    heatmap_buffer = io.BytesIO()
    overlay_buffer = io.BytesIO()
    
    heatmap_pil_img.save(heatmap_buffer, format="PNG")
    overlay_pil_img.save(overlay_buffer, format="PNG")
    
    heatmap_base64 = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
    overlay_base64 = base64.b64encode(overlay_buffer.getvalue()).decode('utf-8')
    
    return heatmap_base64, overlay_base64, heatmap_pil_img, overlay_pil_img # Return PIL images too for saving

# This function seems to be a duplicate or very similar to create_heatmap_overlay
# but takes a PIL image as input. Let's keep it distinct for now if the processing logic differs significantly.
def create_pil_heatmap(pil_img, heatmap, alpha=0.4):
    img_np = np.array(pil_img.convert('RGB')) # Ensure RGB
    height, width, _ = img_np.shape
    
    heatmap_resized = cv2.resize(heatmap, (width, height)) # Use a different variable name
    
    # Normalize and colorize heatmap
    heatmap_resized = np.maximum(heatmap_resized, 0)
    heatmap_min = np.min(heatmap_resized)
    heatmap_max = np.max(heatmap_resized)
    if heatmap_max - heatmap_min < 1e-8:
        heatmap_normalized = np.zeros_like(heatmap_resized)
    else:
        heatmap_normalized = (heatmap_resized - heatmap_min) / (heatmap_max - heatmap_min)
    heatmap_uint8 = np.uint8(255 * heatmap_normalized)
    
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # Convert to RGB
    
    # Superimpose
    # Ensure img_np is float for blending if alpha is float
    superimposed_img_np = heatmap_colored * alpha + img_np * (1 - alpha) # Corrected blending
    superimposed_img_np = np.clip(superimposed_img_np, 0, 255) # Clip before converting to uint8
    superimposed_img_uint8 = np.uint8(superimposed_img_np)
    
    heatmap_pil = Image.fromarray(heatmap_colored)
    overlay_pil = Image.fromarray(superimposed_img_uint8)
    
    heatmap_buffer = io.BytesIO()
    overlay_buffer = io.BytesIO()
    
    heatmap_pil.save(heatmap_buffer, format="PNG")
    overlay_pil.save(overlay_buffer, format="PNG")
    
    heatmap_base64 = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
    overlay_base64 = base64.b64encode(overlay_buffer.getvalue()).decode('utf-8')
    
    return heatmap_base64, overlay_base64, heatmap_pil, overlay_pil # Return PIL images too 