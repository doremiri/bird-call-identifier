import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

audio_dir = "output-dataset/Baudo Guan_sound"  
audio_file = "concatenated.wav"  
audio_path = os.path.join(audio_dir, audio_file)

signal, sr = librosa.load(audio_path, sr=None)  # sr=None to preserve the original sample rate

# Plot the waveform
plt.figure(figsize=(20, 5))
librosa.display.waveshow(signal, sr=sr)
plt.title('Waveform', fontsize=18)
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.show()
