import argparse
from pathlib import Path
import cv2
import librosa
import numpy as np
from tqdm import tqdm

def list_files(source):
    """List audio files with supported extensions"""
    valid_extensions = ['.mp3', '.wav', '.ogg', '.flac']
    path = Path(source)
    return [f for f in path.rglob('*') if f.suffix.lower() in valid_extensions]

def improved_noise_reduction(y, sr, noise_duration=0.5):
    """Improved noise reduction using spectral gating"""
    # Get noise profile from first 500ms
    noise_sample = y[:int(sr * noise_duration)]
    noise_stft = librosa.stft(noise_sample)
    noise_profile = np.mean(np.abs(noise_stft), axis=1)
    
    # Apply spectral gating
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    mask = np.abs(D) > noise_profile[:, np.newaxis]
    y_clean = librosa.istft(D * mask)
    return y_clean

def audio_to_spectrogram(audio_path, save_path, duration=30):
    # Load audio with resampling
    y, sr = librosa.load(audio_path, sr=22050, duration=duration)
    
    # Noise reduction
    y = improved_noise_reduction(y, sr)

    # Mel spectrogram parameters
    n_fft = 2048
    hop_length = 512
    n_mels = 256
    fmax = 11025  # Half of sample rate

    # Create mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                      hop_length=hop_length,
                                      n_mels=n_mels, fmax=fmax)
    
    # Convert to dB scale with better normalization
    S_db = librosa.power_to_db(S, ref=np.median if S.max() == 0 else np.max)

    # Percentile-based normalization
    vmin = np.percentile(S_db, 5)
    vmax = np.percentile(S_db, 95)
    S_norm = np.clip((S_db - vmin) / (vmax - vmin), 0, 1)
    
    # Convert to color image
    S_uint8 = (255 * S_norm).astype(np.uint8)
    colored = cv2.applyColorMap(S_uint8, cv2.COLORMAP_MAGMA)

    # Smart resizing while preserving aspect ratio
    target_size = (224, 224)
    height, width = colored.shape[:2]
    
    # Calculate scaling factor
    scale = min(target_size[0]/height, target_size[1]/width)
    new_size = (int(width * scale), int(height * scale))
    
    resized = cv2.resize(colored, new_size, interpolation=cv2.INTER_AREA)
    
    # Pad to target size
    delta_w = target_size[1] - new_size[0]
    delta_h = target_size[0] - new_size[1]
    top, bottom = delta_h//2, delta_h - (delta_h//2)
    left, right = delta_w//2, delta_w - (delta_w//2)
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                               cv2.BORDER_CONSTANT, value=0)

    # Save as PNG
    cv2.imwrite(save_path, padded)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='audio-dataset')
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--output', type=str, default='spectrogram-dataset')
    args = parser.parse_args()

    Path(args.output).mkdir(exist_ok=True, parents=True)
    
    for audio_file in tqdm(list_files(args.source)):
        species = audio_file.parent.name
        output_path = Path(args.output) / species / f"{audio_file.stem}.png"
        output_path.parent.mkdir(exist_ok=True)
        
        audio_to_spectrogram(str(audio_file), str(output_path), args.duration)