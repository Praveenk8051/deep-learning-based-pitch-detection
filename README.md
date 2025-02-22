## Deep Learning Based Pitch Detection

---

### What is Pitch?

- **Definition:**  
  Pitch is defined as the fundamental frequency—the lowest frequency component of a sound. It determines the “highness” or “lowness” of a tone and is essential for both speech and music.

---

### Psychoacoustic Factors

- **Perception:**  
  - The way our brain and ear perceive sound influences how we experience pitch.
  - Musicians might describe birds chirping as a series of precise, high-pitched notes.
  - Non-musicians tend to describe such sounds subjectively—as pleasant, lively, or soothing.

---

### MIDI – Musical Instrument Digital Interface

- **Overview:**  
  MIDI provides digital music data.  
- **Attributes:**  
  - **Start/Stop Times:** When a note begins and ends.
  - **Note Number:** For example, the note C is represented by 60.
  - **Velocity:** Indicates how hard a note is pressed, affecting the loudness.
  
- **Conversion to Audio:**  
  - The MIDI note number is first converted to a frequency using the formula:  
    f₀ = 440 × 2^((n – 69)/12)  
    - **440 Hz** is the frequency of the A4 note.
    - The exponent **1/12** reflects the equal-tempered scale used in Western music.
    
  - This frequency is then used to generate a sinusoidal wave with the equation:  
    x(t) = A · sin(2πf₀t + φ)  
    - Here, **φ = 0** since the focus is on pitch and frequency.
    - The amplitude **A** is derived from the MIDI velocity.

---

### From Time-Domain Audio to Frequency Domain: Constant Q Transform (CQT)

- **Process:**  
  - Once a time-domain audio signal is generated, the Constant Q Transform (CQT) is applied (typically every second) to produce spectrograms.
  
- **Intuitive Explanation:**  
  - In CQT, lower notes (which change more slowly) receive fine, detailed frequency markings, while higher notes (which change more quickly) have broader markings.
  - This transform is applied because it provides a time-frequency representation that better captures the nuances of musical signals than a standard Fourier Transform.

- **Key Formula:**  
  - The quality factor (Q) in CQT is given by:  
    Q = f₀ / Δf  
    where Δf is the bandwidth of the filter centered at frequency f₀.

---

### Denoising with Convolutional Neural Network (CNN)

- **Noise Addition:**  
  - Noise is added to the spectrograms to simulate real-world conditions.
  
- **CNN Architecture:**  
  - A Convolutional Neural Network with 4 layers is used, with the following number of neurons in each layer: 32, 64, 64, and 128.
  - The network is trained to denoise the noisy spectrograms.

---

### Pitch Detection with LSTM

- **Inputs and Labels:**  
  - **Inputs:** Denoised spectrograms (with dimensions like 96×1).
  - **Labels:** One-hot encoded pitch values corresponding to the MIDI scale.
  
- **Architecture:**  
  - The Long Short-Term Memory (LSTM) network processes the sequence of denoised features to detect pitch over time.

---

### Evaluation

- **Loss Functions Used:**  
  - **Mean Squared Error (MSE):** Used for the CNN denoiser to measure the difference between the denoised output and the clean target.
  - **Cross Entropy:**  
    - **Definition:** Cross entropy measures the difference between two probability distributions.  
    - **Formula:**  
      H(p, q) = – Σₓ p(x) · log q(x)  
    - In classification, p(x) is usually a one-hot encoded vector representing the true label, and q(x) is the predicted probability distribution.
  
---

### Conclusion

The experiment demonstrates that synthetic audio data—generated from MIDI—can effectively train deep neural networks for pitch detection. The combined CNN-LSTM architecture successfully denoises the time-frequency representations and accurately detects pitch in real-world audio signals. This approach minimizes the need for labor-intensive manual labeling and shows that a model trained on synthetic data can generalize well to natural sounds.

---

Blog about this project can be found [here](https://medium.com/@praveenkrishna/deep-learning-based-pitch-detection-cnn-lstm-3a2c5477c4e6)

---

### Folder Structure

The folder consists of 5 directories:

#### dataset_creation:  
- Contains scripts used to:
  - Convert MIDI data to time domain signals 
  - Create LSTM Pickle files
  - Create LSTM Non-Overlapping time steps
  - Create LSTM Overlapping time steps

#### evaluation_matlabFiles: 
- Contains scripts used to:
  - Reconstruct the audio files
  - Visualize the files created for real audio
  - Visualize the files created for synthetic audio

#### test_files:
- Contains 2 directories:
  - **cnn:** 
    - Feed spectrogram as .png file
    - Feed spectrogram as .mat file 
  - **lstm:** 
    - File to create Pickle file
    - File to create Timesteps 
    - Test using trained LSTM network

#### trained_nets:
- Contains:
  - Overlapping (LSTM)
  - Non-overlapping (LSTM)
  - CNN network (CNN)

#### training_files:
- Contains:
  - CNN architecture
  - LSTM architecture

---

### Procedure

#### CNN
1. Create datasets Inputs and Labels as .png files.
2. Provide the path to the CNN architecture file.
3. Train the network.
4. Test the network by passing the image in Matlab (applying CQT) and pass the saved image through the CNN network.

#### LSTM
1. Create datasets Inputs and Labels using Matlab. (Matlab)
   - Inputs are the spectrograms as .png files and labels are .mat files.
2. Create a pickle of 96xN by concatenating all spectrograms and .mat files. (Python)
3. Create the timesteps 96x216 using Python.
4. Provide the path of timesteps inputs and labels to the LSTM architecture.
5. Train the network.

#### Test the Network
1. Create 96xN of spectrograms using pickle.
2. Convert this to 96x216 timesteps.
3. Pass the file to the test LSTM script.
4. Save the network outputs.
5. Visualize the network outputs using Matlab.