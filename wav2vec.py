import time
import os
import torch
from pathlib import Path
import evaluate
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score
)
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment
import pandas as pd
import gc
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from datasets import ClassLabel
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import matplotlib.pyplot as plt
import itertools


# Define the sample rate in Hertz (Hz) expected by Wav2Vec 2.0
RATE_HZ = 16000
# Define the maximum audio interval length to consider in seconds
MAX_SECONDS = 10
# Calculate the maximum audio interval length in samples
MAX_LENGTH = RATE_HZ * MAX_SECONDS

# Define the fraction of records to be used for testing data
TEST_SIZE = 0.2

# Define the paths for the original MP3 dataset and the new WAV dataset
ORIGINAL_DATASET_PATH = Path('audio-dataset').absolute()
WAV_DATASET_PATH = Path('wav-audio-dataset').absolute()
WAV_DATASET_PATH.mkdir(exist_ok=True)

def convert_mp3_to_wav():
    """Converts MP3 files in the original dataset to WAV files in the new directory."""
    if not ORIGINAL_DATASET_PATH.exists():
        print(f"Error: Original directory not found: {ORIGINAL_DATASET_PATH}")
        return

    print(f"Converting MP3 files from: {ORIGINAL_DATASET_PATH} to {WAV_DATASET_PATH}")
    for mp3_file in tqdm(list(ORIGINAL_DATASET_PATH.rglob('*.mp3')), desc="Converting MP3 to WAV"):
        try:
            relative_path = mp3_file.relative_to(ORIGINAL_DATASET_PATH)
            output_dir = WAV_DATASET_PATH / relative_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            wav_file = output_dir / mp3_file.with_suffix('.wav').name
            if not wav_file.exists():
                audio = AudioSegment.from_mp3(mp3_file)
                audio = audio.set_frame_rate(RATE_HZ)  # Ensure the target sample rate
                audio.export(wav_file, format="wav")
        except Exception as e:
            print(f"Error converting {mp3_file}: {e}")

def load_wav_data():
    """Loads the paths and labels of the WAV files from the new dataset directory."""
    file_list = []
    label_list = []
    if not WAV_DATASET_PATH.exists():
        print(f"Error: WAV directory not found: {WAV_DATASET_PATH}")
        return pd.DataFrame(columns=['file', 'label'])

    print(f"Searching for WAV files in: {WAV_DATASET_PATH}")
    for file in WAV_DATASET_PATH.rglob('*.wav'):
        try:
            parent_dir = os.path.basename(os.path.dirname(file))
            label = parent_dir.split('_')[0]
            file_list.append(str(file))  # Store as string for consistency
            label_list.append(label)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    dd = pd.DataFrame()
    dd['file'] = file_list
    dd['label'] = label_list
    return dd

def split_wav_audio(file_path):
    """Loads a WAV audio file, splits it into segments of MAX_LENGTH, and returns a DataFrame (mono)."""
    try:
        audio, rate = torchaudio.load(file_path)
        if rate != RATE_HZ:
            resampler = T.Resample(rate, RATE_HZ)
            audio = resampler(audio)

        # Convert to mono by averaging channels if it's not already mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        num_segments = audio.shape[1] // MAX_LENGTH
        segmented_audio = []
        for i in range(num_segments):
            start = i * MAX_LENGTH
            end = (i + 1) * MAX_LENGTH
            segment = audio[:, start:end].squeeze(0).numpy()
            segmented_audio.append(segment)

        df_segments = pd.DataFrame({'audio': segmented_audio})
        return df_segments

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# --- Main Execution ---

# Convert MP3 to WAV (this will only run if the WAV files don't already exist)
convert_mp3_to_wav()

# Measure execution time for loading WAV data
start_time_load = time.time()

# Load the WAV data into a DataFrame
df = load_wav_data()

# Calculate and print execution time for loading
execution_time_load = time.time() - start_time_load
print(f"\nExecution time for loading WAV data: {execution_time_load:.2f} seconds")

# Sample and display 5 random rows from the WAV DataFrame
#if not df.empty:
#    sample_rows_wav = df.sample(5)
#    print("\nSample of 5 random rows from WAV dataset:")
#    print(sample_rows_wav)
#    print(f"\nTotal number of WAV audio files found: {len(df)}")
#    unique_labels_wav = df['label'].unique()
#    print(f"\nUnique labels found in WAV dataset: {unique_labels_wav}")
#else:
#    print("\nNo WAV audio files were loaded.")
#    exit()

# Apply the 'split_wav_audio' function to each WAV file
df_list_wav = []
tqdm.pandas(desc="Splitting WAV Audio")
for index, row in df.iterrows():
    input_file = row['file']
    input_label = row['label']
    resulting_df = split_wav_audio(input_file)
    if resulting_df is not None:
        resulting_df['label'] = input_label
        df_list_wav.append(resulting_df)

if df_list_wav:
    df_segmented_wav = pd.concat(df_list_wav, axis=0, ignore_index=True)
    print("\nDataFrame after splitting WAV audio:")
    print(df_segmented_wav.sample(5))
else:
    print("\nNo WAV audio files were successfully split")

del df_list_wav
gc.collect()
# Selecting rows in the DataFrame where the 'audio' column is not null (contains non-missing values).
df_segmented_wav = df_segmented_wav[~df_segmented_wav['audio'].isnull()]
df_segmented_wav.info()

if 'file' in df_segmented_wav.columns:
    df_segmented_wav = df_segmented_wav.drop(['file'], axis=1) #removing the file column if it exists

dataset = Dataset.from_pandas(df_segmented_wav)
classes = np.unique(df_segmented_wav['label'])
print(classes)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=df_segmented_wav['label'])
class_weights = dict(zip(classes, weights))
# Print the computed class weights in a more readable format.
print("Class Weights:")
for class_label, weight in class_weights.items():
    print(f"{class_label}: {weight}")

# Create a list of unique labels
labels_list = sorted(list(df_segmented_wav['label'].unique()))

# Initialize empty dictionaries to map labels to IDs and vice versa
label2id, id2label = dict(), dict()

# Iterate over the unique labels and assign each label an ID, and vice versa
for i, label in enumerate(labels_list):
    label2id[label] = i  # Map the label to its corresponding ID
    id2label[i] = label  # Map the ID to its corresponding label

# Print the resulting dictionaries for reference
print("Mapping of IDs to Labels:", id2label, '\n')
print("Mapping of Labels to IDs:", label2id)
# Creating classlabels to match labels to IDs
ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)

# Mapping labels to IDs
def map_label2id(example):
    example['label'] = ClassLabels.str2int(example['label'])
    return example

dataset = dataset.map(map_label2id, batched=True)

# Casting label column to ClassLabel Object
dataset = dataset.cast_column('label', ClassLabels)

# Splitting the dataset into training and testing sets using the predefined train/test split ratio.
dataset = dataset.train_test_split(test_size=TEST_SIZE, shuffle=True, stratify_by_column="label")
del df_segmented_wav
gc.collect()

#specify the model we want to use
model_name = "facebook/wav2vec2-base"
#load the feature extractor 
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#load the pretrained model
model = AutoModelForAudioClassification.from_pretrained(model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id).to(device)
#print("Feature Extractor: ", feature_extractor)
#print("Model: ", model)

# Define a preprocessing function for the dataset
def preprocess_function(batch):
    # Extract audio features from the input batch using the feature_extractor
    inputs = feature_extractor(batch['audio'], sampling_rate=RATE_HZ, max_length=MAX_LENGTH, truncation=True)
    # Extract and store only the 'input_values' component from the extracted features
    inputs['input_values'] = inputs['input_values'][0]
    return inputs

# Apply the preprocess_function to the 'train' split of the dataset, removing the 'audio' column
dataset['train'] = dataset['train'].map(preprocess_function, remove_columns="audio", batched=False)
# Apply the same preprocess_function to the 'test' split of the dataset, removing the 'audio' column
dataset['test'] = dataset['test'].map(preprocess_function, remove_columns="audio", batched=False)
gc.collect()

# Load the "accuracy" metric using the evaluate.load() function.
accuracy = evaluate.load("accuracy")

# Define a function to compute evaluation metrics, which takes eval_pred as input.
def compute_metrics(eval_pred):
    # Extract the model's predictions from eval_pred.
    predictions = eval_pred.predictions
    
    # Apply the softmax function to convert prediction scores into probabilities.
    predictions = np.exp(predictions) / np.exp(predictions).sum(axis=1, keepdims=True)
    
    # Extract the true label IDs from eval_pred.
    label_ids = eval_pred.label_ids
    
    # Calculate accuracy using the loaded accuracy metric by comparing predicted classes
    # (argmax of probabilities) with the true label IDs.
    acc_score = accuracy.compute(predictions=predictions.argmax(axis=1), references=label_ids)['accuracy']
    
    # Return the computed accuracy as a dictionary with a key "accuracy."
    return {
        "accuracy": acc_score
    }

# Define the batch size for training data
batch_size = 4

# Define the number of warmup steps for learning rate scheduling
warmup_steps = 50

# Define the weight decay for regularization
weight_decay = 0.02

# Define the number of training epochs
num_train_epochs = 10

# Define the name for the model directory
model_name = "bird_sounds_classification-wav2vec2"

# Create TrainingArguments object to configure the training process
training_args = TrainingArguments(
    output_dir=model_name,  # Directory to save the model
    logging_dir='./logs',  # Directory for training logs
    num_train_epochs=num_train_epochs,  # Number of training epochs
    per_device_train_batch_size=batch_size,  # Batch size for training
    per_device_eval_batch_size=batch_size,  # Batch size for evaluation
    learning_rate=3e-6,  # Learning rate for training
    logging_strategy='steps',  # Log at specified steps
    logging_first_step=True,  # Log the first step
    load_best_model_at_end=True,  # Load the best model at the end of training
    logging_steps=1,  # Log every step
    evaluation_strategy='epoch',  # Evaluate at the end of each epoch
    warmup_steps=warmup_steps,  # Number of warmup steps for learning rate
    weight_decay=weight_decay,  # Weight decay for regularization
    eval_steps=1,  # Evaluate every step
    gradient_accumulation_steps=1,  # Number of gradient accumulation steps
    gradient_checkpointing=True,  # Enable gradient checkpointing
    save_strategy='epoch',  # Save model at the end of each epoch
    save_total_limit=1,  # Limit the number of saved checkpoints
    fp16=torch.cuda.is_available(),
)

# Create a Trainer object to manage the training process
trainer = Trainer(
    model=model,  # The model to be trained
    args=training_args,  # Training configuration
    train_dataset=dataset['train'],  # Training dataset
    eval_dataset=dataset['test'],  # Evaluation dataset
    tokenizer=feature_extractor,  # Tokenizer
    compute_metrics=compute_metrics, # Compute metrics
)

trainer.evaluate()
trainer.train()
trainer.evaluate()

# Use the trained 'trainer' to make predictions on the test dataset.
outputs = trainer.predict(dataset['test'])

# Print the metrics obtained from the prediction outputs.
print(outputs.metrics)
# Extract the true labels from the model outputs
y_true = outputs.label_ids

# Predict the labels by selecting the class with the highest probability
y_pred = outputs.predictions.argmax(1)

# Define a function to plot a confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8), is_norm=True):
    """
    This function plots a confusion matrix.

    Parameters:
        cm (array-like): Confusion matrix as returned by sklearn.metrics.confusion_matrix.
        classes (list): List of class names, e.g., ['Class 0', 'Class 1'].
        title (str): Title for the plot.
        cmap (matplotlib colormap): Colormap for the plot.
    """
    # Create a figure with a specified size
    plt.figure(figsize=figsize)
    
    
    # Display the confusion matrix as an image with a colormap
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # Define tick marks and labels for the classes on the axes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    if is_norm:
        fmt = '.3f'
    else:
        fmt = '.0f'
    # Add text annotations to the plot indicating the values in the cells
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    # Label the axes
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Ensure the plot layout is tight
    plt.tight_layout()
    # Display the plot
    plt.show()

# Calculate accuracy and F1 score
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

# Display accuracy and F1 score
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Get the confusion matrix if there are a relatively small number of labels
if len(labels_list) <= 120:
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred) # normalize='true'

    # Plot the confusion matrix using the defined function
    plot_confusion_matrix(cm, labels_list, figsize=(18, 16), is_norm=False)

# Finally, display classification report
print()
print("Classification report:")
print()
print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))
