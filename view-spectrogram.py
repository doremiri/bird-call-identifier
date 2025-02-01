

import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Load a spectrogram file

spectrogram = np.load("output-dataset/Cinereous Tinamou_sound/spectrogram_90.npy")

# Plot the spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram, sr=22050, hop_length=512, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (dB)')
plt.show()
