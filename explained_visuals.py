import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import average_precision_score, classification_report
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import itertools

IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
MODEL_PATH = 'apple_classifier_model.h5'
VISUALIZATION_DIR = 'explained_visuals'

os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def set_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

set_style()

print("Model laden...")
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
print(f"Gevonden testafbeeldingen: {len(test_image_paths)}")

print("Voorspellingen genereren...")
y_true = test_generator.classes
y_pred_raw = model.predict(test_generator)
y_pred = y_pred_raw.flatten()
y_pred_classes = (y_pred > 0.5).astype(int)
accuracy = (y_pred_classes == y_true).mean()
print(f"Nauwkeurigheid: {accuracy:.4f}")

def plot_confusion_matrix_with_explanation():
    cm = confusion_matrix(y_true, y_pred_classes)
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    total = cm.sum()
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 2], height_ratios=[3, 1])
    
    ax0 = plt.subplot(gs[0])
    classes = ['Vers', 'Rot']
    
    im = ax0.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax0.set_title("Confusion Matrix: Appel Classificatie", fontsize=16, pad=20)
    
    tick_marks = np.arange(len(classes))
    ax0.set_xticks(tick_marks)
    ax0.set_yticks(tick_marks)
    ax0.set_xticklabels(classes, fontsize=14)
    ax0.set_yticklabels(classes, fontsize=14)
    ax0.set_ylabel('Werkelijke klasse', fontsize=14)
    ax0.set_xlabel('Voorspelde klasse', fontsize=14)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax0.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                 horizontalalignment="center", fontsize=14,
                 color="white" if cm[i, j] > thresh else "black")
    
    cbar = plt.colorbar(im, ax=ax0, shrink=0.8)
    cbar.set_label('Aantal voorspellingen', fontsize=12)
    
    ax1 = plt.subplot(gs[1])
    ax1.axis('off')
    ax1.text(0, 0.9, "Confusion Matrix Uitleg:", fontsize=14, fontweight='bold')
    ax1.text(0, 0.7, "• True Positives (rechtsboven): Correct als 'rot' geclassificeerd", fontsize=12)
    ax1.text(0, 0.5, "• True Negatives (linksboven): Correct als 'vers' geclassificeerd", fontsize=12)
    ax1.text(0, 0.3, "• False Positives (rechtsboven): Incorrect als 'rot' geclassificeerd", fontsize=12)
    ax1.text(0, 0.1, "• False Negatives (linksonder): Incorrect als 'vers' geclassificeerd", fontsize=12)
    
    precision = cm[1, 1] / cm[:, 1].sum() if cm[:, 1].sum() > 0 else 0
    recall = cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    ax2 = plt.subplot(gs[3])
    ax2.axis('off')
    ax2.text(0, 0.9, "Model Prestatie Metrieken:", fontsize=14, fontweight='bold')
    ax2.text(0, 0.7, f"• Totale nauwkeurigheid: {(cm[0, 0] + cm[1, 1]) / total:.2%}", fontsize=12)
    ax2.text(0, 0.5, f"• Precisie (rot): {precision:.2%}", fontsize=12)
    ax2.text(0, 0.3, f"• Recall (rot): {recall:.2%}", fontsize=12)
    ax2.text(0, 0.1, f"• F1-score (rot): {f1:.2%}", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'confusion_matrix_explained.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_curves_with_explanation():
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12), gridspec_kw={'width_ratios': [3, 2]})
    
    axs[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axs[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[0, 0].set_xlim([0.0, 1.0])
    axs[0, 0].set_ylim([0.0, 1.05])
    axs[0, 0].set_xlabel('False Positive Rate')
    axs[0, 0].set_ylabel('True Positive Rate')
    axs[0, 0].set_title('ROC Curve (Receiver Operating Characteristic)')
    axs[0, 0].legend(loc="lower right")
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    axs[0, 1].axis('off')
    axs[0, 1].text(0, 0.95, "ROC Curve Uitleg:", fontsize=14, fontweight='bold')
    axs[0, 1].text(0, 0.85, "De ROC (Receiver Operating Characteristic) curve toont de trade-off tussen:", fontsize=12)
    axs[0, 1].text(0, 0.75, "• True Positive Rate (Sensitiviteit)", fontsize=12)
    axs[0, 1].text(0, 0.65, "• False Positive Rate (1 - Specificiteit)", fontsize=12)
    axs[0, 1].text(0, 0.50, "AUC (Area Under Curve):", fontsize=12, fontweight='bold')
    axs[0, 1].text(0, 0.40, f"• AUC: {roc_auc:.3f}", fontsize=12)
    axs[0, 1].text(0, 0.30, "• AUC = 1.0: Perfect model", fontsize=12)
    axs[0, 1].text(0, 0.20, "• AUC = 0.5: Willekeurig model (stippellijn)", fontsize=12)
    axs[0, 1].text(0, 0.10, "• Hoe dichter bij 1.0, hoe beter het model", fontsize=12)
    
    axs[1, 0].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    axs[1, 0].fill_between(recall, precision, alpha=0.2, color='blue')
    axs[1, 0].set_xlim([0.0, 1.0])
    axs[1, 0].set_ylim([0.0, 1.05])
    axs[1, 0].set_xlabel('Recall')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 0].set_title('Precision-Recall Curve')
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    axs[1, 1].axis('off')
    axs[1, 1].text(0, 0.95, "Precision-Recall Curve Uitleg:", fontsize=14, fontweight='bold')
    axs[1, 1].text(0, 0.85, "De PR curve is vooral nuttig bij onevenwichtige datasets:", fontsize=12)
    axs[1, 1].text(0, 0.75, "• Precision = TP / (TP + FP)", fontsize=12)
    axs[1, 1].text(0, 0.65, "• Recall = TP / (TP + FN)", fontsize=12)
    axs[1, 1].text(0, 0.50, "Average Precision (AP):", fontsize=12, fontweight='bold')
    axs[1, 1].text(0, 0.40, f"• AP: {avg_precision:.3f}", fontsize=12)
    axs[1, 1].text(0, 0.30, "• AP = 1.0: Perfect model", fontsize=12)
    axs[1, 1].text(0, 0.20, "• Hoge precision betekent weinig vals-positieven", fontsize=12)
    axs[1, 1].text(0, 0.10, "• Hoge recall betekent weinig vals-negatieven", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'performance_curves_explained.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_distribution_with_explanation():
    fresh_preds = y_pred[y_true == 0]
    rotten_preds = y_pred[y_true == 1]
    
    fresh_conf = np.maximum(1 - fresh_preds, fresh_preds)
    rotten_conf = np.maximum(1 - rotten_preds, rotten_preds)
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[3, 3, 2], height_ratios=[4, 1])
    
    ax1 = plt.subplot(gs[0, 0])
    bins = np.linspace(0, 1, 21)
    ax1.hist(fresh_preds, bins=bins, alpha=0.7, color='green', label='Verse appels')
    ax1.hist(rotten_preds, bins=bins, alpha=0.7, color='brown', label='Rotte appels')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Beslissingsgrens')
    ax1.set_xlabel('Voorspelde waarschijnlijkheid (rot)')
    ax1.set_ylabel('Aantal voorspellingen')
    ax1.set_title('Distributie van voorspellingen per klasse')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2 = plt.subplot(gs[0, 1])
    data = {
        'Waarschijnlijkheid': np.concatenate([fresh_preds, rotten_preds]),
        'Werkelijke klasse': np.concatenate([['Vers'] * len(fresh_preds), 
                                            ['Rot'] * len(rotten_preds)])
    }
    df = pd.DataFrame(data)
    
    sns.violinplot(x='Werkelijke klasse', y='Waarschijnlijkheid', 
                  data=df, ax=ax2, palette={'Vers': 'green', 'Rot': 'brown'})
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2)
    ax2.set_ylim(0, 1)
    ax2.set_title('Violin plot van voorspellingsverdelingen')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    ax3 = plt.subplot(gs[0, 2])
    ax3.axis('off')
    ax3.text(0, 0.95, "Voorspellingsdistributie Uitleg:", fontsize=14, fontweight='bold')
    ax3.text(0, 0.85, "Deze grafieken tonen hoe het model voorspellingen doet:", fontsize=12)
    ax3.text(0, 0.75, "• Histogram: Frequentie van voorspelde waarschijnlijkheden", fontsize=12)
    ax3.text(0, 0.65, "• Violin plot: Verdeling van voorspellingen per klasse", fontsize=12)
    ax3.text(0, 0.55, "• Rode lijn: Beslissingsgrens (0.5)", fontsize=12)
    ax3.text(0, 0.40, "Ideale situatie:", fontsize=12, fontweight='bold')
    ax3.text(0, 0.30, "• Verse appels: voorspellingen dichtbij 0", fontsize=12)
    ax3.text(0, 0.20, "• Rotte appels: voorspellingen dichtbij 1", fontsize=12)
    ax3.text(0, 0.10, "• Minder overlap betekent een beter model", fontsize=12)
    
    ax4 = plt.subplot(gs[1, :])
    ax4.axis('off')
    
    correctly_fresh = np.sum(fresh_preds < 0.5)
    correctly_rotten = np.sum(rotten_preds >= 0.5)
    incorrectly_fresh = len(fresh_preds) - correctly_fresh
    incorrectly_rotten = len(rotten_preds) - correctly_rotten
    
    fresh_conf_correct = np.mean(1 - fresh_preds[fresh_preds < 0.5])
    rotten_conf_correct = np.mean(rotten_preds[rotten_preds >= 0.5])
    
    stats_text = (
        f"Statistieken van voorspellingen:\n"
        f"• Verse appels correct geclassificeerd: {correctly_fresh}/{len(fresh_preds)} ({correctly_fresh/len(fresh_preds):.1%})"
        f" met gemiddelde betrouwbaarheid {fresh_conf_correct:.1%}\n"
        f"• Rotte appels correct geclassificeerd: {correctly_rotten}/{len(rotten_preds)} ({correctly_rotten/len(rotten_preds):.1%})"
        f" met gemiddelde betrouwbaarheid {rotten_conf_correct:.1%}\n"
        f"• Voorspellingen dicht bij beslissingsgrens (onzeker, <60% betrouwbaarheid): "
        f"{np.sum((y_pred > 0.4) & (y_pred < 0.6))}/{len(y_pred)} ({np.sum((y_pred > 0.4) & (y_pred < 0.6))/len(y_pred):.1%})"
    )
    
    ax4.text(0.5, 0.5, stats_text, fontsize=12, ha='center', va='center', 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'prediction_distribution_explained.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_performance_summary():
    cm = confusion_matrix(y_true, y_pred_classes)
    
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    precision = cm[1, 1] / cm[:, 1].sum() if cm[:, 1].sum() > 0 else 0
    recall = cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    avg_precision = average_precision_score(y_true, y_pred)
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.5])
    
    ax_title = plt.subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.6, "Appel Classificatie Model: Prestatie Overzicht", 
                 fontsize=18, fontweight='bold', ha='center')
    ax_title.text(0.5, 0.3, f"Model: apple_classifier_model.h5   |   Klassen: Verse en Rotte Appels", 
                 fontsize=14, ha='center', color='gray')
    
    ax_metrics = plt.subplot(gs[1, 0])
    ax_metrics.axis('off')
    metrics = [
        ("Nauwkeurigheid", accuracy, "Percentage correct geclassificeerde appels"),
        ("Precisie", precision, "Percentage van als 'rot' geclassificeerde appels die echt rot zijn"),
        ("Recall", recall, "Percentage rotte appels dat is gedetecteerd"),
        ("F1-Score", f1, "Harmonisch gemiddelde van precisie en recall"),
        ("ROC AUC", roc_auc, "Gebied onder de ROC curve, separatievermogen"),
        ("PR AUC", avg_precision, "Gebied onder de precision-recall curve")
    ]
    
    for i, (name, value, desc) in enumerate(metrics):
        ax_metrics.text(0.1, 0.9 - i*0.15, f"{name}:", fontsize=14, fontweight='bold')
        ax_metrics.text(0.4, 0.9 - i*0.15, f"{value:.3f}", fontsize=16, color='blue', fontweight='bold')
        ax_metrics.text(0.6, 0.9 - i*0.15, desc, fontsize=12, color='gray')
    
    ax_roc = plt.subplot(gs[1, 1])
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle='--', alpha=0.7)
    
    ax_cm = plt.subplot(gs[1, 2])
    classes = ['Vers', 'Rot']
    
    im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xticks(np.arange(len(classes)))
    ax_cm.set_yticks(np.arange(len(classes)))
    ax_cm.set_xticklabels(classes)
    ax_cm.set_yticklabels(classes)
    ax_cm.set_ylabel('Werkelijke klasse')
    ax_cm.set_xlabel('Voorspelde klasse')
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax_cm.text(j, i, format(cm[i, j], 'd'),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'model_performance_summary.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

print("Visualisaties met uitleg genereren...")
plot_confusion_matrix_with_explanation()
plot_curves_with_explanation()
plot_prediction_distribution_with_explanation()
plot_model_performance_summary()

print(f"Alle visualisaties zijn opgeslagen in de map '{VISUALIZATION_DIR}'")
print("Visualisatie voltooid!")