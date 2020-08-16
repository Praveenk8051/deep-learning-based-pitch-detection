## Deep learning based pitch detection
### Abstract
The usage of deep learning models in audio signal domain tasks has substantially increased over the time. The success of these models, however, depends largely on the labeled dataset and their availability. Using digital data to render sufficient datasets for training have gained the focus as it reduces dependencies on the real world data. However, the realism of such digital data with corresponding real world data needs great human effort.

In this theoretical explanation, digital data based alternative prototype is proposed for pitch detection. As it is established, the relevant information of the signal is important for pitch detection, a synthesized audio signal is generated using MIDI data. Instead of processing the audio signal using conventional methods for pitch values, pitch tracker does the needful even in presence of noise using deep neural network. The proposed work has efficient approach to achieve the requirement. The evaluation on real world data shows the promising results. Finally, analysis on real world data shows the criteria that dataset needs to have for successful detection.

The appropriateness of the approach is demonstrated by training the state-of-the-art neural network architectures for pitch detection. Monophonic recordings under LMD (Lakh Midi Datasets) are considered. This scenario is evaluated on the Bach datasets to check the performance of model on real world data. The experiments conducted shows that the synthetic data helps the model in training and detecting the pitches when the real world data is passed. Instead of creating the real world datasets for statistical method based pitch detection, which can be complex, synthetic data can be used along with consideration of sound properties.

Blog about this project can be found [here](https://medium.com/@praveenkrishna/deep-learning-based-pitch-detection-cnn-lstm-3a2c5477c4e6)


The folder consists of 5 directories

#### dataset_creation:  
		Contains script used to:
		Convert MIDI data to time domain signals 
		LSTM Pickle file creation
		LSTM Non-Overlapping time steps
		LSTM Overlapping time steps
		
#### evaluation_matlabFiles: 
		Contains script used to:
		Reconstruct the audio files
		Visualize the files created for real audio
		Visualize the files created for synthetic audio
		
#### test_files:
		2 directories,
		cnn, contains
			feed spectrogram as .png file
			feed spectrogram as .mat file 
		lstm, contains
			file to create Pickle file
			file to create Timesteps 
			test using traied LSTM network
	
#### trained_nets:
		overlapping (LSTM)
		non-overlapping(LSTM)
		cnn network(CNN)
		
#### training_files:
		CNN architecture
		LSTM architecture
		
		
### Procedure:

##### CNN
	Create datasets Inputs and Labels as .png file
	Provide the path to CNN architecture file
	Train the network
	Test the network by passing the image in Matlab(applying CQT) and pass the saved image through CNN network.


##### LSTM
	Create datasets Inputs and Labels using Matlab. (Matlab)
	Inputs are the spectrogram as .png file and labels are mat files.
	Create a pickle of 96xN by concatenating all spectrograms and mat files.(python)
	Now create the timesteps 96x216 using python.
	Provide the path of timesteps inputs and labels to LSTM architecture
	Train the network

##### Test the network by passing spectrograms 
	Create 96xN of spectrograms using pickle
	Convert this to 96x216 timesteps
	Pass the file to test LSTM script
	save the network outputs
	Visulize the network outputs using Matlab.
	
	
	
	





