import os
import librosa
import numpy as np
from pydub import AudioSegment
import gc  # For garbage collection

def concatenate_audio_files_in_memory(file_paths, output_file):
    """Concatenate all audio files into a single audio file in memory."""
    print(f"Concatenating {len(file_paths)} audio files in memory...")
    combined = AudioSegment.empty()
    for file_path in file_paths:
        print(f"Loading {file_path}...")
        audio = AudioSegment.from_file(file_path)
        combined += audio    
    combined.export(output_file, format="wav")
    return combined

def remove_noise_and_silence(audio, sr, noise_reduction_threshold=0.1, n_fft=2048, hop_length=512):
    """
    Apply spectral gating to reduce noise and remove silence.
    
    Parameters:
        audio (np.ndarray): Input audio signal.
        sr (int): Sample rate of the audio.
        noise_reduction_threshold (float): Threshold for noise reduction (0 to 1).
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
    
    Returns:
        np.ndarray: Noise-reduced audio signal.
    """
    print("Applying spectral gating for noise reduction...")
    
    # Convert audio to floating-point format and normalize to [-1, 1]
    audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)  # Magnitude and phase components

    # Estimate the noise profile (using the first few frames as noise)
    noise_frames = 5  # Number of frames to use for noise estimation
    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    # Apply spectral gating
    threshold = noise_profile * noise_reduction_threshold
    magnitude_clean = np.where(magnitude < threshold, 0, magnitude)

    # Reconstruct the STFT and convert back to time-domain audio
    stft_clean = magnitude_clean * phase
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length)

    # Trim leading and trailing silence
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


def process_batch(file_paths, output_folder, chunk_length=10, batch_offset=0):
    """Process a batch of audio files."""
    # Concatenate audio files in memory
    concatenated_file = os.path.join(output_folder, "concatenated.wav")     
    concatenated_audio = concatenate_audio_files_in_memory(file_paths, concatenated_file)

    # Convert concatenated audio to numpy array
    print("Converting concatenated audio to numpy array...")
    audio_samples = np.array(concatenated_audio.get_array_of_samples())
    sr = concatenated_audio.frame_rate

    # Free memory used by concatenated_audio
    del concatenated_audio
    gc.collect()

    # Apply noise reduction and silence removal
    audio_clean = remove_noise_and_silence(audio_samples, sr)

    # Split into fixed-length chunks
    chunks = split_into_chunks(audio_clean, sr, chunk_length=chunk_length)

    # Convert each chunk to mel spectrogram and save
    print("Converting chunks to mel spectrograms...")
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        mel_spec = audio_to_mel_spectrogram(chunk, sr)
        # Save spectrogram as numpy array with a unique name
        output_file = os.path.join(output_folder, f"spectrogram_{batch_offset + i}.npy")
        np.save(output_file, mel_spec)
        print(f"Saved spectrogram to {output_file}.")

    # Free memory used by chunks
    del chunks
    gc.collect()

def process_species_audio(species_folder, output_folder, chunk_length=10, batch_size=5):
    """Process all audio files for a specific species in batches."""
    print(f"Processing species folder: {species_folder}...")
    # Get all audio files in the folder
    file_paths = [os.path.join(species_folder, f) for f in os.listdir(species_folder) if f.endswith('.mp3')]
    print(f"Found {len(file_paths)} audio files.")

    # Process files in batches
    total_chunks = 0  # Track the total number of chunks processed so far
    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} of {len(file_paths) // batch_size + 1}...")
        process_batch(batch_files, output_folder, chunk_length=chunk_length, batch_offset=total_chunks)
        
        # Update the total number of chunks processed
        concatenated_file = os.path.join(output_folder, f"temp_concatenated_{i}.wav")  # Temporary file for concatenation
        concatenated_audio = concatenate_audio_files_in_memory(batch_files, concatenated_file)
        audio_samples = np.array(concatenated_audio.get_array_of_samples())
        sr = concatenated_audio.frame_rate
        audio_clean = remove_noise_and_silence(audio_samples, sr)
        chunks = split_into_chunks(audio_clean, sr, chunk_length=chunk_length)
        total_chunks += len(chunks)

        # Free memory
        del concatenated_audio, audio_samples, audio_clean, chunks
        gc.collect()

        # Delete the temporary concatenated file
        if os.path.exists(concatenated_file):
            os.remove(concatenated_file)

    print(f"Finished processing {species_folder}.\n")
    return

def process_all_species(input_base_folder, output_base_folder, chunk_length=10, batch_size=10):
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
            # Process the species audio in batches
            process_species_audio(species_folder, species_output_folder, chunk_length=chunk_length, batch_size=batch_size)

# Example usage
input_base_folder = "audio-dataset"
output_base_folder = "output-dataset"
process_all_species(input_base_folder, output_base_folder, chunk_length=10, batch_size=5)
