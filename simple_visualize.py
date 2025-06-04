import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
import itertools
import seaborn as sns

# Constants
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
MODEL_PATH = 'apple_classifier_model.h5'
VISUALIZATION_DIR = 'apple_model_visualizations'

# Create visualization directory
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

print("Loading the trained model...")
model = load_model(MODEL_PATH)
model.summary()

# Prepare data generator for test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Important for matching predictions with filenames
)

# Store test image paths
test_image_paths = [os.path.join('test', test_generator.filenames[i]) for i in range(len(test_generator.filenames))]
print(f"Found {len(test_image_paths)} test images")

# Plot confusion matrix
def plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.title(f"Normalized {title}")
    else:
        plt.title(title)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the figure
    plt.savefig(os.path.join(VISUALIZATION_DIR, f"{title.lower().replace(' ', '_')}.png"))
    plt.close()

# Function to plot precision-recall curve
def plot_precision_recall_curve(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.fill_between(recall, precision, alpha=0.2, color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(VISUALIZATION_DIR, "precision_recall_curve.png"))
    plt.close()
    
    return avg_precision

# Function to plot prediction distribution
def plot_prediction_distribution(predictions, true_labels):
    plt.figure(figsize=(12, 5))
    
    # Separate predictions by true class
    fresh_preds = predictions[true_labels == 0]
    rotten_preds = predictions[true_labels == 1]
    
    # Plot histograms
    plt.subplot(1, 2, 1)
    plt.hist(fresh_preds, bins=20, alpha=0.7, color='green', label='Fresh class')
    plt.hist(rotten_preds, bins=20, alpha=0.7, color='brown', label='Rotten class')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Decision boundary')
    plt.title('Distribution of Predicted Probabilities by True Class')
    plt.xlabel('Predicted Probability (Rotten)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot violinplots
    plt.subplot(1, 2, 2)
    data = {
        'Probability': np.concatenate([fresh_preds, rotten_preds]),
        'True Class': np.concatenate([['Fresh'] * len(fresh_preds), ['Rotten'] * len(rotten_preds)])
    }
    
    # Convert to DataFrame for seaborn
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Create violin plot
    sns.violinplot(x='True Class', y='Probability', data=df)
    plt.axhline(y=0.5, color='red', linestyle='--')
    plt.title('Violin Plot of Predictions by True Class')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'prediction_distribution.png'))
    plt.close()

# Function to plot sample predictions
def visualize_sample_predictions(predictions, true_labels, image_paths, num_samples=10):
    # Get correct and incorrect predictions
    pred_classes = (predictions > 0.5).astype(int)
    correct_indices = np.where(pred_classes.flatten() == true_labels)[0]
    incorrect_indices = np.where(pred_classes.flatten() != true_labels)[0]
    
    # Select random samples
    n_correct = min(num_samples // 2, len(correct_indices))
    n_incorrect = min(num_samples // 2, len(incorrect_indices))
    
    if n_correct > 0:
        correct_samples = np.random.choice(correct_indices, n_correct, replace=False)
        plot_samples(correct_samples, predictions, true_labels, image_paths, 
                    "correct_predictions.png", "Correctly Classified Samples")
    
    if n_incorrect > 0:
        incorrect_samples = np.random.choice(incorrect_indices, n_incorrect, replace=False)
        plot_samples(incorrect_samples, predictions, true_labels, image_paths, 
                    "incorrect_predictions.png", "Incorrectly Classified Samples")

# Helper function to plot samples
def plot_samples(indices, predictions, true_labels, image_paths, filename, title):
    from tensorflow.keras.preprocessing import image
    
    n_samples = len(indices)
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 3 * n_rows))
    
    for i, idx in enumerate(indices):
        # Get image
        img_path = image_paths[idx]
        img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        
        # Get prediction details
        true_class = 'Fresh' if true_labels[idx] < 0.5 else 'Rotten'
        pred_class = 'Fresh' if predictions[idx] < 0.5 else 'Rotten'
        confidence = predictions[idx] if pred_class == 'Rotten' else 1 - predictions[idx]
        
        # Plot
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img)
        color = 'green' if pred_class == true_class else 'red'
        plt.title(f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}", color=color)
        plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, filename))
    plt.close()

# Function to visualize hard examples
def visualize_hard_examples(predictions, true_labels, image_paths, threshold=0.2, num_samples=10):
    # Get prediction classes
    pred_classes = (predictions > 0.5).astype(int)
    is_correct = pred_classes.flatten() == true_labels
    
    # Calculate confidence scores (distance from decision boundary)
    confidence = np.abs(predictions - 0.5)
    
    # Get hard examples (correct and incorrect with low confidence)
    hard_correct = np.where(is_correct & (confidence < threshold))[0]
    hard_incorrect = np.where((~is_correct) & (confidence < threshold))[0]
    
    # Plot hard examples
    if len(hard_correct) > 0:
        samples = np.random.choice(hard_correct, min(num_samples, len(hard_correct)), replace=False)
        plot_samples(samples, predictions, true_labels, image_paths, 
                    "hard_correct_examples.png", "Hard But Correctly Classified Examples")
    else:
        print("No hard correctly classified examples found")
    
    if len(hard_incorrect) > 0:
        samples = np.random.choice(hard_incorrect, min(num_samples, len(hard_incorrect)), replace=False)
        plot_samples(samples, predictions, true_labels, image_paths, 
                    "hard_incorrect_examples.png", "Hard and Incorrectly Classified Examples")
    else:
        print("No hard incorrectly classified examples found")

# Main visualization workflow
print("Generating model visualizations...")

# Get predictions
print("Getting model predictions...")
test_generator.reset()
y_true = test_generator.classes
y_pred_raw = model.predict(test_generator)
y_pred = y_pred_raw.flatten()
y_pred_classes = (y_pred > 0.5).astype(int)

# Calculate basic metrics
accuracy = (y_pred_classes == y_true).mean()
print(f"Accuracy: {accuracy:.4f}")

# Confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred_classes)
class_names = ['Fresh', 'Rotten']  # Based on alphabetical order
plot_confusion_matrix(cm, class_names, normalize=False)
plot_confusion_matrix(cm, class_names, normalize=True, title='Normalized Confusion Matrix')

# ROC curve
print("Generating ROC curve...")
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(VISUALIZATION_DIR, 'roc_curve.png'))
plt.close()

# Precision-recall curve
print("Generating precision-recall curve...")
avg_precision = plot_precision_recall_curve(y_true, y_pred)
print(f"Average Precision: {avg_precision:.4f}")

# Prediction distribution visualization
print("Generating prediction distribution plots...")
plot_prediction_distribution(y_pred, y_true)

# Sample predictions
print("Generating sample prediction visualizations...")
visualize_sample_predictions(y_pred, y_true, test_image_paths, num_samples=20)

# Hard examples
print("Generating hard examples visualizations...")
visualize_hard_examples(y_pred, y_true, test_image_paths)

# Classification report
print("\nClassification Report:")
report = classification_report(y_true, y_pred_classes, target_names=class_names)
print(report)

# Save report to file
with open(os.path.join(VISUALIZATION_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Save metrics to file
with open(os.path.join(VISUALIZATION_DIR, 'model_metrics.txt'), 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
    f.write(f"Average Precision: {avg_precision:.4f}\n")

print(f"\nAll visualizations saved to '{VISUALIZATION_DIR}' directory") 