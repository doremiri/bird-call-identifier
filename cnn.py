import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Configuration
input_base_folder = "output-dataset"  # Folder containing species folders with .npy files
num_classes = len(os.listdir(input_base_folder))  # Number of species (classes)
img_height, img_width = 128, 938  # Ensure consistent spectrogram width
batch_size = 32
epochs = 20

# Function to pad spectrograms to a fixed width
def pad_spectrogram(spectrogram, target_width=938):
    current_width = spectrogram.shape[1]
    if current_width < target_width:
        padding = np.zeros((spectrogram.shape[0], target_width - current_width))
        spectrogram = np.hstack((spectrogram, padding))  # Pad with zeros
    return spectrogram

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
                spectrogram = pad_spectrogram(spectrogram, img_width)

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

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=num_classes)

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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
        layers.Dense(128, activation='relu'),
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
model.save("bird_species_cnn_model-2.h5")
print("Model saved to bird_species_cnn_model-2.h5")
