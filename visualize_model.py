import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
import itertools
import seaborn as sns
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shutil

IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
MODEL_PATH = 'apple_classifier_model.h5'
VISUALIZATION_DIR = 'model_visualizations'

if os.path.exists(VISUALIZATION_DIR):
    shutil.rmtree(VISUALIZATION_DIR)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

DIRS = ['basic', 'gradcam', 'feature_maps', 'samples', 'interpretability', 'metrics']
for dir_name in DIRS:
    os.makedirs(os.path.join(VISUALIZATION_DIR, dir_name), exist_ok=True)

print("Loading the trained model...")
model = load_model(MODEL_PATH)
model.summary()

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_image_paths = [os.path.join('test', test_generator.filenames[i]) for i in range(len(test_generator.filenames))]
print(f"Found {len(test_image_paths)} test images")

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
    
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'metrics', f"{title.lower().replace(' ', '_')}.png"))
    plt.close()

def generate_gradcam(img_array, model, last_conv_layer_name="conv2d_3"):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
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
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    return heatmap

def create_heatmap_overlay(img_path, heatmap, alpha=0.6):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    
    heatmap = cv2.resize(heatmap, (width, height))
    
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)
    
    return img, heatmap, superimposed_img

def visualize_feature_maps(model, layer_name, img_path, save_path):
    layer_outputs = [layer.output for layer in model.layers if layer_name in layer.name]
    
    if not layer_outputs:
        print(f"Layer {layer_name} not found in model")
        return
    
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    activations = activation_model.predict(img_array)
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    feature_maps = activations[0]
    n_features = min(16, feature_maps.shape[-1])
    
    plt.subplot(1, 2, 2)
    plt.title(f"Feature Maps from {layer_name}")
    
    grid_size = int(np.ceil(np.sqrt(n_features)))
    for i in range(n_features):
        plt.subplot(grid_size + 1, grid_size + 1, i + 1)
        plt.imshow(feature_maps[:, :, i], cmap='viridis')
        plt.axis('off')
    
    plt.suptitle(f"Feature Maps Visualization - {layer_name}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_gradcam_batch(model, image_paths, predictions, true_labels, last_conv_layer='conv2d_3', num_images=10):
    indices = np.random.choice(len(image_paths), min(num_images, len(image_paths)), replace=False)
    
    for i, idx in enumerate(indices):
        img_path = image_paths[idx]
        
        img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img)
        img_tensor = np.expand_dims(img_array, axis=0) / 255.0
        
        prediction = predictions[idx]
        true_label = true_labels[idx]
        
        class_label_pred = "Rotten" if prediction > 0.5 else "Fresh"
        class_label_true = "Rotten" if true_label > 0.5 else "Fresh"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        try:
            heatmap = generate_gradcam(img_tensor, model, last_conv_layer)
            
            original, heatmap_colored, superimposed = create_heatmap_overlay(img_path, heatmap)
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(original)
            plt.title(f"Original\nTrue: {class_label_true}\nPred: {class_label_pred}\nConf: {confidence:.2f}")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap_colored)
            plt.title("Grad-CAM Heatmap")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(superimposed)
            plt.title("Superimposed")
            plt.axis('off')
            
            plt.tight_layout()
            filename = os.path.basename(img_path)
            plt.savefig(os.path.join(VISUALIZATION_DIR, 'gradcam', f"gradcam_{i}_{filename}.png"))
            plt.close()
        except Exception as e:
            print(f"Error generating GradCAM for {img_path}: {e}")

def generate_lime_explanations(model, image_paths, predictions, true_labels, num_samples=5):
    try:
        explainer = lime_image.LimeImageExplainer()
        indices = np.random.choice(len(image_paths), min(num_samples, len(image_paths)), replace=False)
        
        for i, idx in enumerate(indices):
            img_path = image_paths[idx]
            
            img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            
            prediction = predictions[idx]
            true_label = true_labels[idx]
            class_label_pred = "Rotten" if prediction > 0.5 else "Fresh"
            class_label_true = "Rotten" if true_label > 0.5 else "Fresh"
            
            def predict_fn(images):
                images_preprocessed = images.copy()
                return model.predict(images_preprocessed)
            
            explanation = explainer.explain_instance(
                img_array, predict_fn, 
                top_labels=2, hide_color=0, num_samples=1000
            )
            
            temp_1, mask_1 = explanation.get_image_and_mask(
                explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
            )
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(img_array)
            plt.title(f"Original\nTrue: {class_label_true}\nPred: {class_label_pred}")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(mark_boundaries(temp_1, mask_1))
            plt.title("LIME Explanation \n(Green = Supporting Evidence)")
            plt.axis('off')
            
            filename = os.path.basename(img_path)
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATION_DIR, 'interpretability', f"lime_{i}_{filename}.png"))
            plt.close()
            
    except Exception as e:
        print(f"Error generating LIME explanations: {e}")
        print("Try installing LIME with: pip install lime")

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
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'metrics', "precision_recall_curve.png"))
    plt.close()
    
    return avg_precision

def visualize_predictions(predictions, true_labels, image_paths, num_samples=20):
    pred_classes = (predictions > 0.5).astype(int)
    correct_indices = np.where(pred_classes.flatten() == true_labels)[0]
    incorrect_indices = np.where(pred_classes.flatten() != true_labels)[0]
    
    n_correct = min(num_samples // 2, len(correct_indices))
    n_incorrect = min(num_samples // 2, len(incorrect_indices))
    
    if n_correct > 0:
        correct_samples = np.random.choice(correct_indices, n_correct, replace=False)
        plot_prediction_samples(correct_samples, predictions, true_labels, image_paths, 
                               "correct_predictions.png", title="Correctly Classified Samples")
    
    if n_incorrect > 0:
        incorrect_samples = np.random.choice(incorrect_indices, n_incorrect, replace=False)
        plot_prediction_samples(incorrect_samples, predictions, true_labels, image_paths, 
                               "incorrect_predictions.png", title="Incorrectly Classified Samples")

def plot_prediction_samples(indices, predictions, true_labels, image_paths, filename, title):
    n_samples = len(indices)
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 3 * n_rows))
    
    for i, idx in enumerate(indices):
        img_path = image_paths[idx]
        img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        
        true_class = 'Fresh' if true_labels[idx] < 0.5 else 'Rotten'
        pred_class = 'Fresh' if predictions[idx] < 0.5 else 'Rotten'
        confidence = predictions[idx] if pred_class == 'Rotten' else 1 - predictions[idx]
        
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img)
        color = 'green' if pred_class == true_class else 'red'
        plt.title(f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}", color=color)
        plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'samples', filename))
    plt.close()

def plot_prediction_distribution(predictions, true_labels):
    plt.figure(figsize=(12, 5))
    
    fresh_preds = predictions[true_labels == 0]
    rotten_preds = predictions[true_labels == 1]
    
    plt.subplot(1, 2, 1)
    plt.hist(fresh_preds, bins=20, alpha=0.7, color='green', label='Fresh class')
    plt.hist(rotten_preds, bins=20, alpha=0.7, color='brown', label='Rotten class')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Decision boundary')
    plt.title('Distribution of Predicted Probabilities by True Class')
    plt.xlabel('Predicted Probability (Rotten)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    df = {'Prediction': np.concatenate([fresh_preds, rotten_preds]),
          'True Class': np.concatenate([['Fresh'] * len(fresh_preds), ['Rotten'] * len(rotten_preds)])}
    
    sns.violinplot(x='True Class', y='Prediction', data=df, palette={'Fresh': 'green', 'Rotten': 'brown'})
    plt.axhline(y=0.5, color='red', linestyle='--', label='Decision boundary')
    plt.title('Violin Plot of Predictions by True Class')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'metrics', 'prediction_distribution.png'))
    plt.close()

def generate_saliency_maps(model, image_paths, num_samples=5):
    indices = np.random.choice(len(image_paths), min(num_samples, len(image_paths)), replace=False)
    
    for i, idx in enumerate(indices):
        img_path = image_paths[idx]
        
        img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img)
        img_tensor = np.expand_dims(img_array, axis=0) / 255.0
        
        input_tensor = tf.convert_to_tensor(img_tensor)
        
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = model(input_tensor)
            top_pred_idx = tf.argmax(predictions[0])
            top_class = predictions[:, top_pred_idx]
        
        gradients = tape.gradient(top_class, input_tensor)
        
        pooled_gradients = tf.reduce_max(tf.abs(gradients), axis=-1)
        
        pooled_gradients = pooled_gradients / tf.reduce_max(pooled_gradients)
        saliency = pooled_gradients.numpy()[0]
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img_array / 255.0)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(saliency, cmap='inferno')
        plt.title("Saliency Map")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(img_array / 255.0)
        plt.imshow(saliency, cmap='inferno', alpha=0.7)
        plt.title("Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        filename = os.path.basename(img_path)
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'interpretability', f"saliency_{i}_{filename}.png"))
        plt.close()

def visualize_filters(model, layer_name='conv2d_1'):
    filters = None
    for layer in model.layers:
        if layer_name in layer.name:
            filters, biases = layer.get_weights()
            break
    
    if filters is None:
        print(f"Layer {layer_name} not found")
        return
    
    n_filters = min(16, filters.shape[-1])
    filter_size = filters.shape[0]
    
    plt.figure(figsize=(12, 8))
    
    for i in range(n_filters):
        f = filters[:, :, :, i]
        
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        
        for j in range(min(3, f.shape[2])):
            plt.subplot(4, n_filters, i + 1 + j * n_filters)
            plt.imshow(f[:, :, j], cmap='viridis')
            plt.title(f"Filter {i+1}, Channel {j+1}")
            plt.axis('off')
    
    plt.suptitle(f"Learned Filters in Layer {layer_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'feature_maps', f"filters_{layer_name}.png"))
    plt.close()

def visualize_model_architecture(model):
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=os.path.join(VISUALIZATION_DIR, 'basic', 'model_architecture.png'),
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
        )
        print("Model architecture visualization saved")
    except Exception as e:
        print(f"Error generating model architecture: {e}")
        print("Try installing graphviz with: apt-get install graphviz")

def visualize_hard_examples(predictions, true_labels, image_paths, n_samples=10):
    pred_classes = (predictions > 0.5).astype(int)
    is_correct = pred_classes.flatten() == true_labels
    
    confidence = np.abs(predictions - 0.5)
    
    hard_correct = np.where(is_correct & (confidence < 0.2))[0]
    hard_incorrect = np.where((~is_correct) & (confidence < 0.2))[0]
    
    if len(hard_correct) > 0:
        samples = np.random.choice(hard_correct, min(n_samples, len(hard_correct)), replace=False)
        plot_prediction_samples(samples, predictions, true_labels, image_paths, 
                               "hard_correct_examples.png", "Hard But Correctly Classified Examples")
    else:
        print("No hard correctly classified examples found")
    
    if len(hard_incorrect) > 0:
        samples = np.random.choice(hard_incorrect, min(n_samples, len(hard_incorrect)), replace=False)
        plot_prediction_samples(samples, predictions, true_labels, image_paths, 
                               "hard_incorrect_examples.png", "Hard and Incorrectly Classified Examples")
    else:
        print("No hard incorrectly classified examples found")

print("Generating model visualizations...")

print("Getting model predictions...")
test_generator.reset()
y_true = test_generator.classes
y_pred_raw = model.predict(test_generator)
y_pred = y_pred_raw.flatten()
y_pred_classes = (y_pred > 0.5).astype(int)

print("Visualizing model architecture...")
visualize_model_architecture(model)

print("Visualizing model filters...")
visualize_filters(model, 'conv2d_1')
visualize_filters(model, 'conv2d_3')

print("Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred_classes)
class_names = ['Fresh', 'Rotten']
plot_confusion_matrix(cm, class_names, normalize=False)
plot_confusion_matrix(cm, class_names, normalize=True, title='Normalized Confusion Matrix')

print("Generating ROC curve...")
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

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
plt.savefig(os.path.join(VISUALIZATION_DIR, 'metrics', 'roc_curve.png'))
plt.close()

print("Generating precision-recall curve...")
avg_precision = plot_precision_recall_curve(y_true, y_pred)
print(f"Average Precision: {avg_precision:.4f}")

print("Generating prediction distribution...")
plot_prediction_distribution(y_pred, y_true)

print("Generating GradCAM visualizations...")
visualize_gradcam_batch(model, test_image_paths, y_pred, y_true, num_images=10)

print("Generating feature map visualizations...")
if test_image_paths:
    for layer_name in ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4']:
        sample_img = test_image_paths[0]
        save_path = os.path.join(VISUALIZATION_DIR, 'feature_maps', f"feature_maps_{layer_name}.png")
        visualize_feature_maps(model, layer_name, sample_img, save_path)

print("Generating prediction visualizations...")
visualize_predictions(y_pred, y_true, test_image_paths, num_samples=20)

print("Generating hard examples visualizations...")
visualize_hard_examples(y_pred, y_true, test_image_paths)

print("Generating saliency maps...")
generate_saliency_maps(model, test_image_paths)

print("Generating LIME explanations (if available)...")
try:
    generate_lime_explanations(model, test_image_paths, y_pred, y_true)
except:
    print("LIME not available - install with 'pip install lime'")

print("\nClassification Report:")
report = classification_report(y_true, y_pred_classes, target_names=class_names)
print(report)

with open(os.path.join(VISUALIZATION_DIR, 'metrics', 'classification_report.txt'), 'w') as f:
    f.write(report)

accuracy = (y_pred_classes == y_true).mean()
with open(os.path.join(VISUALIZATION_DIR, 'metrics', 'performance_metrics.txt'), 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
    f.write(f"Average Precision: {avg_precision:.4f}\n")

print(f"\nAll visualizations saved to '{VISUALIZATION_DIR}' directory")
print("\nDONE!") 