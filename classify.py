import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

# Configuration
model_path = "bird_species_cnn_model.h5"  # Path to the saved model
test_base_folder = "new-output-dataset"  # Folder containing new species folders with .npy files
img_height, img_width = 128, 938  # Ensure consistent spectrogram dimensions

# Function to pad spectrograms to a fixed width (same as during training)
def pad_spectrogram(spectrogram, target_width=938):
    current_width = spectrogram.shape[1]
    if current_width < target_width:
        padding = np.zeros((spectrogram.shape[0], target_width - current_width))
        spectrogram = np.hstack((spectrogram, padding))  # Pad with zeros
    return spectrogram

# Step 1: Load the new dataset
def load_test_dataset(base_folder):
    X_test = []  # Spectrograms
    y_test = []  # True labels (if available)
    class_names = sorted(os.listdir(base_folder))  # List of species folders
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(base_folder, class_name)
        print(f"Loading test data for class: {class_name}...")
        for file_name in os.listdir(class_folder):
            if file_name.endswith(".npy"):
                file_path = os.path.join(class_folder, file_name)
                spectrogram = np.load(file_path)

                # Pad if too short
                spectrogram = pad_spectrogram(spectrogram, img_width)

                print(f"{file_name} shape after padding: {spectrogram.shape}")
                X_test.append(spectrogram)
                y_test.append(class_idx)  # Assign class index as label
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_test, y_test, class_names

# Load the new dataset
X_test, y_test, class_names = load_test_dataset(test_base_folder)

# Add a channel dimension to X_test (required for CNN input)
X_test = np.expand_dims(X_test, axis=-1)  # Shape: (num_samples, height, width, 1)

# Step 2: Load the trained model
print("Loading the trained model...")
model = tf.keras.models.load_model(model_path)

# Step 3: Make predictions on the new dataset
print("Making predictions...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Step 4: Evaluate performance (if true labels are available)
if len(y_test) > 0:
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))
else:
    print("No true labels provided. Predictions were made without evaluation.")

# Step 5: Print predictions (optional)
print("\nPredictions:")
for i in range(len(X_test)):
    print(f"Sample {i + 1}: Predicted = {class_names[y_pred_classes[i]]}")
