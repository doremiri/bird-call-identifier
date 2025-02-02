TODO: add images to githubb README

# Bird song classifier using Mel-Spectrograms and CNN model

For our implementation we utilized **audio to mel-spectrogram** to process the audio data, and used that to train a **CNN** classification model. 
### Dataset

The dataset was acquired from:
https://www.kaggle.com/datasets/soumendraprasad/sound-of-114-species-of-birds-till-2022
The dataset includes many species, with recordings from various sources and locations, with varying degrees of background noise.
These conditions are optimal for training the CNN model.
#### Dataset limitations

The full dataset includes 114 species, however many of these species contain very little audio, resulting in a dataset that is very wide with many classes, but limited data for some of those classes. 

**Additionally**, due to time and hardware limitations, we could not train on all of these species.

Due to these limitations, in this study we hand selected 6 species which included atleast 3 minutes of **trimmed audio**.

![[Figure_5.png]]
\- **Fig (1):** Total duration of audio of species used for training and validating this model 

---
### Preprocessing

#### Managing audio files:

**Naive approach:**

simply
1. Run noise reduction and silence removal 
2. Take a fixed size part of the audio (for example 30 seconds, **CNNs** need consistent dimensions across their data)
3. If shorter than decided size, pad with zeroes
4. Transform into Mel-Spectrogram

However this would cause various issues:
- Cutting audio files that're above 30 seconds would lose valuable data
- Padding audio files that're under 30 seconds would result in spectrograms with blank space to the right, distorting CNN results


**Our Approach:**

Steps:
- Concatenate audio files of each species into 1 large audio file
- Noise reduction and Silence removal are ran on this 1 concatenated audio file
- The resulting audio file is split into chunks of equal size ( e.g. `chunk_length` = 10)
- These chunks are each turned into the Mel-Spectrogram
- If the final chunk does not fit neatly into the `chunk_length`, discard it to maintain dataset consistency 

This results in many equal sized chunks that contain the bird call data. Without much data loss or half-empty Spectrograms.

![[Pasted image 20250201180911.png]]
\- **Fig (2):** Diagram showcasing the steps for the audio of each species: concatenation, audio enhancement, chunking into **spectrograms**.

This presents another issue, the concatenated audio, takes up alot of **memory**, which is a limited resource as only have access to our personal laptop devices.
To overcome this, we implemented **batching**.
Divide the **concated** audio file into smaller batches decided by **batch_size**, and run the pre-processing on those

![[Pasted image 20250201190219.png]]
\- **Fig (3):** Incorporating **Batching** to reduce **memory usage** at the cost of increased **CPU usage** and **processing time**
#### Noise Reduction and Silence Removal

**1. Short-Time Fourier Transform (STFT)**
- Converts the audio signal from the time domain (waveform) to the frequency domain (spectrum).
**2. Spectral Gating (Noise Reduction)**
- The noise profile is estimated by analyzing the first few frames of the audio (assuming they contain only background noise).
- A threshold is set based on the noise profile.
- Any frequency component below this threshold is considered noise and removed.
**3. Inverse STFT (I-STFT)**
- Converts the cleaned frequency-domain representation back into a time-domain waveform.
**4. Trimming (Silence Removal)**
-  Removes unnecessary silent parts from the beginning and end of the audio.

The Threshold used in our training and evaluation is relatively low, as the original audio from the dataset is clean with the bird calls being distinct among any background noise.
#### Mel-Spectrogram

We used the function `melspectrogram()` from the `librosa` library

This function applies:
- **STFT (Short-Time Fourier Transform)** to break the audio into small overlapping windows and compute frequency information over time.
- A **Mel filterbank** to transform the frequency bins into a Mel scale.

Since raw Spectrogram values represent **power**, which spans many orders of magnitude, we convert it to decibels (dB). Therefore, a logarithmic scale called the mel scale
is used to represent frequencies. **Decibel scaling** helps highlight small but important frequency changes.

![[spectro-after.png]]
![[wave-after.png]]
\- **Fig (4):** The waveform and Spectrogram of the concatenated audio of one of the species after noise reduction is applied. Loaded audio with `sample rate`: 48000 Hz, `duration`: 588.58 seconds. `Threshold` = 0.1 and `noise_frames` = 5.

#### Examples of Spectrograms post processing

After processing the audio into spectrograms we can convert the data from .npy format into an image format using the `view-spectrogram.py`script. 

After processing we can easily distinguish patterns by species, in this section we show some Spectrogram examples from 2 of the species.

**Andean Guan**
Andean Guan is characterized by quick bursts around the 512 Hz to 1024 Hz region

![[Pasted image 20250131123747.png]]
![[Pasted image 20250131123801.png]]
\- **Fig (5):** Spectrograms of the Andean Guan post processing

**Barletts Tinamou**
Barletts Tinamou is characterized by quick narrow bursts around the 1500 Hz region

![[Pasted image 20250131124020.png]]
![[Pasted image 20250131124035.png]]
![[Pasted image 20250131124048.png]]
\- **Fig (6):** Spectrograms of the Barletts Tinamou post processing


#### Pre-processing pipeline summary:

![[Pasted image 20250202140950.png]]

--- 

### Training and Evaluating the Model

#### CNN model Parameters

Species used to train:
- Andean Guan ~5 mins audio
- Band Tailed Guan ~5 mins audio
- Bartletts Tinamou ~5 mins audio
- Cinereous Tinamou ~11 mins audio
- Dusky Legged Guan ~3 mins audio
- West Mexican Chachalaca ~7 mins audio

Constants used:
- **Noise reduction threshold** = 0.1
- **batch_size** = 32
- **epochs** = 20

| Layer (type)                   | Output Shape         | Param #    |
| ------------------------------ | -------------------- | ---------- |
| conv2d (Conv2D)                | (None, 126, 936, 32) | 320        |
| max_pooling2d (MaxPooling2D)   | (None, 63, 468, 32)  | 0          |
| conv2d_1 (Conv2D)              | (None, 61, 466, 64)  | 18,496     |
| max_pooling2d_1 (MaxPooling2D) | (None, 30, 233, 64)  | 0          |
| conv2d_2 (Conv2D)              | (None, 28, 231, 128) | 73,856     |
| max_pooling2d_2 (MaxPooling2D) | (None, 14, 115, 128) | 0          |
| flatten (Flatten)              | (None, 206080)       | 0          |
| dense (Dense)                  | (None, 128)          | 26,378,368 |
| dropout (Dropout)              | (None, 128)          | 0          |
| dense_1 (Dense)                | (None, 6)            | 774        |
**Model:** "sequential"
**Total params:** 26,471,814 (100.98 MB)  
**Trainable params:** 26,471,814 (100.98 MB)  

```
Training data shape: 1834
Validation data shape: 459
```

1834 spectrograms used for training and 459 used for validation, each Spectrogram had 3 consistently sized channels.

#### Results

After training for 20 epochs the final evaluation reached these results:

![[Figure_1.png]]\- **Fig (7):** The confusion matrix of the of the validation data, showcasing a high true positive rate

![[Figure_4.png]]
\- **Fig (8):** shows the both validation and training accuracy over the 20 epochs

The final evaluation accuracy was around 93.6%

This unexpectedly high percentage made us suspect that the model was experiencing **over-fitting**, where the model memorizes training data but doesn't generalize well.

One method to determine whether over-fitting was present or not is to compare **Validation loss** and **Training loss**

If validation loss is much **higher** than training loss, it could indicate **over-fitting**

![[Figure_3.png]]
\- **Fig (9):** Graphing the validation and training loss over the 20 epochs

Both Training loss and Validation loss are roughly equal at the end of the epochs and throughout, this lends credence to the fact that the model is not experiencing **over-fitting**.

### Conclusion and Possible improvements

This project provides a straightforward approach to process audio files, Categorizing them by, species, and converting that data into a suitable format to be processed by a CNN. Using this method we successfully produced a model that produced high accuracy.

To further enhance the model’s robustness and generalization, we could incorporate data augmentation techniques. Techniques such as **time stretching**, **pitch shifting**, and adding **background noise**, would help the model learn more diverse inputs and improve its performance on real-world data.
