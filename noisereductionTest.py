import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
import gc
from pydub import AudioSegment
import sounddevice as sd
signal, sr = librosa.load('audio-dataset/Baudo Guan_sound/Baudo Guan7.mp3', sr=None)
#new method
# FFT parameters
n_fft = 2048  # number of samples per FFT window
hop_length = 512  # number of samples between successive frames

# Short-time Fourier Transformation (STFT)
audio_stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)

# Convert to absolute values (magnitude)
spectrogram = np.abs(audio_stft)
reduced_signal = nr.reduce_noise(y=signal, sr=sr)
# Compute Mel-Spectrogram directly from the signal
mel_signal = librosa.feature.melspectrogram(
    y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft
)

# Convert power spectrogram (amplitude squared) to decibel (dB) units
power_to_db = librosa.power_to_db(mel_signal, ref=np.max)

sd.play(reduced_signal, sr)
sd.wait()
# Plotting
plt.figure(figsize=(8, 7))
librosa.display.specshow(
    power_to_db, 
    sr=sr, 
    x_axis='time', 
    y_axis='mel', 
    cmap='magma', 
    hop_length=hop_length
)
plt.colorbar(label='dB')
plt.title('Mel-Spectrogram (dB)', fontsize=18)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

# Optional: Clean up
gc.collect()
