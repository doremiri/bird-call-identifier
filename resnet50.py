import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam    
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import pathlib
import os
import seaborn as sns
from itertools import cycle

# Configuration
input_base_folder = "output-dataset"  # Folder containing species folders with .npy files
num_classes = len(os.listdir(input_base_folder))  # Number of species (classes)
img_height, img_width = 224, 224  # Ensure consistent spectrogram width
batch_size = 32
epochs = 10

def pad_spectrogram(spectrogram, target_height=224, target_width=224):
    # Pad or truncate height
    current_height = spectrogram.shape[0]
    if current_height < target_height:
        pad_height = np.zeros((target_height - current_height, spectrogram.shape[1]))
        spectrogram = np.vstack((spectrogram, pad_height))
    elif current_height > target_height:
        spectrogram = spectrogram[:target_height, :]

    # Pad or truncate width 
    current_width = spectrogram.shape[1]
    if current_width < target_width:
        pad_width = np.zeros((spectrogram.shape[0], target_width - current_width))
        spectrogram = np.hstack((spectrogram, pad_width))
    elif current_width > target_width:
        spectrogram = spectrogram[:, :target_width]

    return spectrogram

def apply_spec_augment(spectrogram, time_mask_ratio=0.1, freq_mask_ratio=0.1, p=0.5):
    if np.random.rand() > p:
        return spectrogram

    spec = spectrogram.copy()
    num_freqs, num_frames = spec.shape

    # Time masking
    time_mask_width = int(num_frames * time_mask_ratio)
    time_start = np.random.randint(0, num_frames - time_mask_width)
    spec[:, time_start:time_start + time_mask_width] = 0

    # Frequency masking
    freq_mask_width = int(num_freqs * freq_mask_ratio)
    freq_start = np.random.randint(0, num_freqs - freq_mask_width)
    spec[freq_start:freq_start + freq_mask_width, :] = 0

    return spec


# Step 1: Load .npy files and prepare dataset
def load_dataset(base_folder):
    X = []  # Spectrograms
    y = []  # Labels
    class_names = sorted(os.listdir(base_folder))  # List of species folders
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(base_folder, class_name)
        print(f"Loading data for class: {class_name}...")
        for file_name in os.listdir(class_folder):
            if file_name.endswith(".npy"):
                file_path = os.path.join(class_folder, file_name)
                spectrogram = np.load(file_path)

                 # Pad if too short
                spectrogram = pad_spectrogram(spectrogram, img_height, img_width)

                print(f"{file_name} shape after padding: {spectrogram.shape}")
                X.append(spectrogram)
                y.append(class_idx)  # Assign class index as label
    X = np.array(X)
    y = np.array(y)
    return X, y, class_names

# Load dataset
X, y, class_names = load_dataset(input_base_folder)

# Add a channel dimension to X (required for CNN input)
X = np.expand_dims(X, axis=-1)  # Shape: (num_samples, height, width, 1)
X = np.repeat(X, 3, axis=-1)  # Convert to 3 channels (RGB)
# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=num_classes)

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = resnet_preprocess(X_train)
X_val = resnet_preprocess(X_val)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}") 

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Freeze the base model
model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history1 = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

# Second phase: Unfreeze last 20 layers and fine-tune
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history_2 = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=batch_size)
model.compile(optimizer=Adam(learning_rate=1e-5), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_combined = {
    'loss': history1.history['loss'] + history_2.history['loss'],
    'val_loss': history1.history['val_loss'] + history_2.history['val_loss'],
    'accuracy': history1.history['accuracy'] + history_2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history_2.history['val_accuracy'],
}

print("Evaluating the model...")
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Save the model
model.save("bird_species_resnet50_model-3.h5")
print("Model saved to bird_species_resnet50_model-3.h5")

# Step 5: Generate evaluation metrics and plots
def evaluate_model(model, X_val, y_val, class_names):
    # Predict probabilities and classes
    y_pred_prob = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    # Accuracy
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Precision, Recall, F1-Score
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # ROC Curve (One-vs-Rest for multi-class)
    plt.figure(figsize=(10, 8))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.show()

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history_combined.history['loss'], label='Training Loss')
    plt.plot(history_combined.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history_combined.history['accuracy'], label='Training Accuracy')
    plt.plot(history_combined.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Evaluate the model
evaluate_model(model, X_val, y_val, class_names)

