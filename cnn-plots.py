import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

# Configuration
input_base_folder = "output-dataset"  # Folder containing species folders with .npy files
num_classes = len(os.listdir(input_base_folder))  # Number of species (classes)
img_height, img_width = 224, 224  # Ensure consistent spectrogram width
batch_size = 32
epochs = 20

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

def zscore_minmax_normalize(spectrogram):
    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    spectrogram = (spectrogram - mean) / (std + 1e-9)  # Z-score
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram) + 1e-9)  # Min-Max to [0, 1]
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
                # Normalize the spectrogram
                spectrogram = zscore_minmax_normalize(spectrogram)

                print(f"{file_name} shape after padding: {spectrogram.shape}")
                X.append(spectrogram)
                y.append(class_idx)  # Assign class index as label
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y, class_names

# Load dataset
X, y, class_names = load_dataset(input_base_folder)

# Add a channel dimension to X (required for CNN input)
X = np.expand_dims(X, axis=-1)  # Shape: (num_samples, height, width, 1)

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=num_classes)

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SpecAugment to training data only
for i in range(X_train.shape[0]):
    X_train[i, :, :, 0] = apply_spec_augment(X_train[i, :, :, 0], time_mask_ratio=0.1, freq_mask_ratio=0.1, p=0.5)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")


# Step 2: Build the CNN model
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Fully connected layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # Dropout for regularization
        layers.Dense(num_classes, activation='softmax')  # Output layer
    ])
    return model

# Build the model
input_shape = (img_height, img_width, 1)  # Input shape for the CNN
model = build_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Step 3: Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val)
)

# Step 4: Evaluate the model
print("Evaluating the model...")
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Save the model
model.save("bird_species_cnn_model-3.h5")
print("Model saved to bird_species_cnn_model-3.h5")

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
    plt.figure(figsize=(30, 30))
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
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Evaluate the model
evaluate_model(model, X_val, y_val, class_names)
