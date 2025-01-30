
import argparse
from pathlib import Path
import cv2
import librosa
import numpy as np
from tqdm import tqdm

# TEST

def list_files(source):
    """
    List all audio files in the given source directory and its subdirectories.

    Args:
        source (str): The source directory path.

    Returns:
        list: A list of `Path` objects representing the audio files.
    """
    print(f"Debug: Listing all audio files in source directory: {source}")
    path = Path(source)
    files = []
    for species_folder in path.iterdir():
        if species_folder.is_dir():  # Check if it's a species folder
            species_files = [file for file in species_folder.glob('*.mp3')]
            files.extend(species_files)  # Add mp3 files from this species folder
    print(f"Debug: Found {len(files)} audio files in species subfolders")
    return files


def noise_reduction(y, sr):
    """
    Perform noise reduction on an audio signal.

    Args:
        y (numpy.ndarray): The audio time-series signal.
        sr (int): The sampling rate of the audio signal.

    Returns:
        numpy.ndarray: The denoised audio signal.
    """
    print("Debug: Performing noise reduction")
    # Estimate noise power using silent parts
    noise_power = librosa.feature.rms(y=y).mean()
    print(f"Debug: Estimated noise power: {noise_power}")

    # Create a threshold for noise reduction
    threshold = noise_power * 1.5
    print(f"Debug: Noise reduction threshold: {threshold}")

    # Apply a simple filter to suppress noise below the threshold
    denoised_signal = np.where(np.abs(y) > threshold, y, 0)
    return denoised_signal


def audio_to_spectrogram(audio_path, save_path, duration):
    """
    Convert an audio file to a colored spectrogram and save it as an image.

    Args:
        audio_path (str): The path to the audio file.
        save_path (str): The path to save the spectrogram image.
        duration (int): Duration of the audio file to process in seconds.

    Returns:
        None
    """
    print(f"Debug: Loading audio file from: {audio_path} with duration {duration} seconds")
    
    # Load audio file
    y, sr = librosa.load(audio_path, sr=22050, duration=duration)
    print(f"Debug: Audio loaded with sample rate {sr} and {len(y)} samples")

    # Perform noise reduction
    y = noise_reduction(y, sr)

    # Compute Mel Spectrogram
    print("Debug: Computing Mel Spectrogram")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    print("Debug: Mel Spectrogram computed, converting to dB scale")
    S = librosa.power_to_db(S, ref=np.max)

    # Normalize values to 0-255 range and convert to uint8
    print("Debug: Normalizing spectrogram values to range 0-255")
    S = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply a colormap
    print("Debug: Applying colormap")
    S_colored = cv2.applyColorMap(S, cv2.COLORMAP_JET)

    # Resize image to a fixed size (128x128)
    print("Debug: Resizing spectrogram to 128x128")
    S_resized = cv2.resize(S_colored, (128, 128))

    # Save as PNG
    print(f"Debug: Saving spectrogram image to: {save_path}")
    cv2.imwrite(save_path, S_resized)
    print("Debug: Spectrogram saved successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='audio-dataset', help='source folder containing species subfolders')
    parser.add_argument('--duration', type=int, default=30, help='duration of audio in seconds to process')
    parser.add_argument('--output', type=str, default='output-dataset', help='output folder for spectrograms')
    opt = parser.parse_args()

    source, duration, output = opt.source, opt.duration, opt.output

    print(f"Debug: Starting process with source={source}, duration={duration}s, output={output}")

    # List all audio files
    file_list = list_files(source)

    for file in tqdm(file_list):
        # Extract the species (subfolder) name
        species_folder = file.parent.name
        print(f"Debug: Processing file {file} from species folder {species_folder}")

        # Create corresponding output path
        new_path = Path(output) / species_folder / file.stem

        # Ensure output directory exists
        print(f"Debug: Ensuring output directory exists: {new_path.parent}")
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # Replace suffix with .png
        new_path = new_path.with_suffix('.png')

        # Convert audio to spectrogram
        print(f"Debug: Converting audio to spectrogram and saving to {new_path}")
        audio_to_spectrogram(str(file), str(new_path), duration)
    print("Debug: Processing completed")
