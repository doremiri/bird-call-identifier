import os
import librosa
import numpy as np
from pydub import AudioSegment
from scipy.io.wavfile import write

def concatenate_audio_files(file_paths, output_file):
    """Concatenate all audio files into a single audio file."""
    print(f"Concatenating {len(file_paths)} audio files into {output_file}...")
    combined = AudioSegment.empty()
    for file_path in file_paths:
        print(f"Loading {file_path}...")
        audio = AudioSegment.from_file(file_path)
        combined += audio
    combined.export(output_file, format="wav")
    print(f"Saved concatenated audio to {output_file}.")
    return output_file

def remove_noise_and_silence(audio, sr, noise_reduction_threshold=0.02, silence_threshold=0.01):
    """Apply basic noise reduction and silence removal."""
    print("Applying noise reduction and silence removal...")
    # Noise reduction: Remove low-amplitude noise
    audio_clean = librosa.effects.preemphasis(audio)
    audio_clean = np.where(np.abs(audio_clean) < noise_reduction_threshold, 0, audio_clean)

    # Silence removal: Trim leading and trailing silence
    audio_trimmed, _ = librosa.effects.trim(audio_clean, top_db=20)
    print(f"Audio length before trimming: {len(audio)}, after trimming: {len(audio_trimmed)}.")
    return audio_trimmed

def split_into_chunks(audio, sr, chunk_length=10):
    """Split audio into fixed-length chunks (in seconds)."""
    print(f"Splitting audio into {chunk_length}-second chunks...")
    chunk_size = sr * chunk_length
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
    # Discard the last chunk if it's smaller than chunk_size
    chunks = [chunk for chunk in chunks if len(chunk) == chunk_size]
    print(f"Created {len(chunks)} chunks of {chunk_length} seconds each.")
    return chunks

def audio_to_mel_spectrogram(audio, sr, n_mels=128):
    """Convert audio to mel spectrogram."""
    print("Converting audio to mel spectrogram...")
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def process_species_audio(species_folder, output_folder, chunk_length=10):
    """Process all audio files for a specific species."""
    print(f"Processing species folder: {species_folder}...")
    # Get all audio files in the folder
    file_paths = [os.path.join(species_folder, f) for f in os.listdir(species_folder) if f.endswith('.mp3')]
    print(f"Found {len(file_paths)} audio files.")

    # Concatenate all audio files
    concatenated_file = os.path.join(output_folder, "concatenated.wav")
    concatenated_file = concatenate_audio_files(file_paths, concatenated_file)

    # Load concatenated audio
    print(f"Loading concatenated audio from {concatenated_file}...")
    audio, sr = librosa.load(concatenated_file, sr=None)

    # Apply noise reduction and silence removal
    audio_clean = remove_noise_and_silence(audio, sr)

    # Split into fixed-length chunks
    chunks = split_into_chunks(audio_clean, sr, chunk_length=chunk_length)

    # Convert each chunk to mel spectrogram and save
    print("Converting chunks to mel spectrograms...")
    spectrograms = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        mel_spec = audio_to_mel_spectrogram(chunk, sr)
        spectrograms.append(mel_spec)
        # Save spectrogram as numpy array
        output_file = os.path.join(output_folder, f"spectrogram_{i}.npy")
        np.save(output_file, mel_spec)
        print(f"Saved spectrogram to {output_file}.")

    print(f"Finished processing {species_folder}.\n")
    return spectrograms

def process_all_species(input_base_folder, output_base_folder, chunk_length=10):
    """Process all species folders in the input base folder."""
    # Create output base folder if it doesn't exist
    os.makedirs(output_base_folder, exist_ok=True)

    # Iterate over each species folder
    for species_name in os.listdir(input_base_folder):
        species_folder = os.path.join(input_base_folder, species_name)
        if os.path.isdir(species_folder):
            print(f"\nProcessing species: {species_name}...")
            # Create output folder for this species
            species_output_folder = os.path.join(output_base_folder, species_name)
            os.makedirs(species_output_folder, exist_ok=True)
            # Process the species audio
            process_species_audio(species_folder, species_output_folder, chunk_length=chunk_length)

# Example usage
input_base_folder = "audio-dataset"
output_base_folder = "output-dataset"
process_all_species(input_base_folder, output_base_folder, chunk_length=10)
