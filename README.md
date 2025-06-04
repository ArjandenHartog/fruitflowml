# Apple Freshness Classifier

A TensorFlow-based web application to classify apple images as fresh or rotten with percentage confidence scores.

## Project Structure

```
├── app.py               # Flask web application
├── train_model.py       # Model training script
├── requirements.txt     # Project dependencies
├── templates/           # HTML templates
│   └── index.html       # Web interface
├── uploads/             # Folder for uploaded images (created automatically)
├── train/               # Training dataset
│   ├── freshapples/     # Fresh apple images for training
│   └── rottenapples/    # Rotten apple images for training
└── test/                # Testing dataset
    ├── freshapples/     # Fresh apple images for testing
    └── rottenapples/    # Rotten apple images for testing
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the classification model:

```
python train_model.py
```

This will:
- Train the model on the images in the `train/` directory
- Evaluate the model on the images in the `test/` directory
- Save the trained model as `apple_classifier_model.h5`
- Save the training history plot as `training_history.png`

### Running the Web Application

To start the Flask server:

```
python app.py
```

Then open a web browser and navigate to:
```
http://localhost:5000
```

### Using the Web Interface

1. Upload an image of an apple using the "Select Image" button or by dragging and dropping
2. Click "Analyze Image"
3. View the results showing the percentage confidence of the apple being fresh or rotten

## Model Architecture

- CNN-based architecture with 4 convolutional blocks
- Uses data augmentation to improve generalization
- Binary classification (fresh vs rotten)
- Trained with early stopping to prevent overfitting 