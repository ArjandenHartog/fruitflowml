# Apple Freshness Classifier

A TensorFlow-based web application to classify apple images as fresh or rotten with percentage confidence scores. This project combines deep learning with interactive visualization tools to provide insights into the model's decision-making process.

## Features

- Real-time apple freshness classification
- Interactive web interface
- Model visualization and explainability tools
- Detailed training metrics and performance analysis
- Support for both image upload and drag-and-drop
- Comprehensive model architecture visualization

## Project Structure

```
├── app.py                           # Flask web application
├── train_model.py                   # Model training script
├── visualize_model.py               # Advanced model visualization tool
├── simple_visualize.py              # Basic visualization utilities
├── explained_visuals.py             # Model explainability tools
├── requirements.txt                 # Project dependencies
├── templates/                       # HTML templates
│   └── index.html                   # Web interface
├── uploads/                         # Folder for uploaded images
├── model_visualizations/            # Generated model visualizations
├── explained_visuals/               # Model explanation outputs
├── apple_model_visualizations/      # Additional visualization artifacts
├── train/                          # Training dataset
│   ├── freshapples/                 # Fresh apple images for training
│   └── rottenapples/                # Rotten apple images for training
└── test/                           # Testing dataset
    ├── freshapples/                 # Fresh apple images for testing
    └── rottenapples/                # Rotten apple images for testing
```

## Requirements

- Python 3.8 or higher
- TensorFlow 2.x
- Flask
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Gradio (for visualization interface)

For a complete list of dependencies, see `requirements.txt`.

## Installation

1. Clone this repository
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the classification model:

```bash
python train_model.py
```

This will:
- Train the model on the images in the `train/` directory
- Evaluate the model on the images in the `test/` directory
- Save the trained model as `apple_classifier_model.h5`
- Save the training history plot as `training_history.png`
- Generate performance metrics and confusion matrices

### Running the Web Application

To start the Flask server:

```bash
python app.py
```

Then open a web browser and navigate to:
```
http://localhost:5000
```

### Using the Web Interface

1. Upload an image of an apple using the "Select Image" button or by dragging and dropping
2. Click "Analyze Image"
3. View the results showing:
   - Percentage confidence of the apple being fresh or rotten
   - Visualization of model attention areas
   - Explanation of the classification decision

### Model Visualization

The project includes several visualization tools:

1. Basic Visualization (`simple_visualize.py`):
   ```bash
   python simple_visualize.py
   ```
   - Generates basic model architecture diagrams
   - Shows layer activation maps
   - Displays feature maps

2. Advanced Visualization (`visualize_model.py`):
   ```bash
   python visualize_model.py
   ```
   - Creates detailed model architecture visualizations
   - Shows filter visualizations
   - Generates activation heatmaps
   - Provides performance analysis graphs

3. Model Explainability (`explained_visuals.py`):
   ```bash
   python explained_visuals.py
   ```
   - Generates Grad-CAM visualizations
   - Creates LIME explanations
   - Shows feature importance maps

## Model Architecture

The model uses a CNN-based architecture optimized for apple freshness classification:

- Input Layer: 224x224x3 (RGB images)
- 4 Convolutional blocks:
  - Conv2D layers with increasing filters (32, 64, 128, 256)
  - BatchNormalization for training stability
  - MaxPooling2D for spatial dimension reduction
  - Dropout layers to prevent overfitting
- Global Average Pooling
- Dense layers with dropout
- Binary output with sigmoid activation

### Training Features

- Data Augmentation:
  - Random rotation
  - Horizontal/vertical flips
  - Zoom variation
  - Brightness adjustment
- Early Stopping to prevent overfitting
- Learning rate scheduling
- Batch size optimization
- Validation split: 20%

## Performance Metrics

The model achieves:
- Training accuracy: ~95%
- Validation accuracy: ~93%
- Test set accuracy: ~92%

Detailed performance metrics and visualizations are generated during training and saved in the `model_visualizations/` directory.

## OutSystems Integration

### API Endpoint

The application provides a REST API endpoint that can be integrated with OutSystems applications.

#### Production Endpoint
- **URL**: `https://api.fruitflow.site/api/predict/base64plain`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Development Endpoint (Local Testing)
- **URL**: `http://your-server:5000/analyze`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Request Parameters
```json
{
    "image": "base64_encoded_image_string"
}
```

#### Response Format
```json
{
    "classification": "fresh",
    "filename": "voorbeeld.jpg",
    "fresh_percentage": 87.23,
    "rotten_percentage": 12.77,
    "success": true,
    "heatmap_data": "base64_encoded_string_hier",
    "heatmap_overlay_data": "base64_encoded_string_hier",
    "image_url": "https://yourdomain.com/uploads/filename.jpg"
}
```

#### Response Fields
- `classification`: String - Either "fresh" or "rotten"
- `filename`: String - Name of the processed image file
- `fresh_percentage`: Number - Confidence percentage for fresh classification (0-100)
- `rotten_percentage`: Number - Confidence percentage for rotten classification (0-100)
- `success`: Boolean - Indicates if the analysis was successful
- `heatmap_data`: String - Base64 encoded heatmap visualization
- `heatmap_overlay_data`: String - Base64 encoded heatmap overlay on original image
- `image_url`: String - URL to the processed image

### OutSystems Integration Steps

1. In OutSystems Service Studio, create a REST API integration:
   - Set the Method to `POST`
   - Set the URL to: `https://api.fruitflow.site/api/predict/base64plain`
   - Set Content-Type to `application/json`
   - Configure the input structure as Text data type
   - Set the request to be sent in the body

2. Create the input structure:
```json
{
    "image": "Text"
}
```

3. Create the output structure matching the response format above:
```json
{
    "classification": "Text",
    "filename": "Text",
    "fresh_percentage": "Decimal",
    "rotten_percentage": "Decimal",
    "success": "Boolean",
    "heatmap_data": "Text",
    "heatmap_overlay_data": "Text",
    "image_url": "Text"
}
```

4. Use the REST API in your OutSystems application:
   - Convert your image to base64 before sending
   - Send the request using the REST API
   - Handle the response in your application logic

### Example Usage in OutSystems

```javascript
// Example structure for the API call in OutSystems
var request = {
    image: $base64EncodedImage
};

// Make the API call to the production endpoint
var response = SendRequest("POST", "https://api.fruitflow.site/api/predict/base64plain", request);

// Handle the response
if (response.success) {
    // Use the classification results
    ShowClassification(response.classification);
    DisplayConfidence(response.fresh_percentage, response.rotten_percentage);
    ShowHeatmap(response.heatmap_data);
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License. 