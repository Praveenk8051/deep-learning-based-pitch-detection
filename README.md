# deep-learning-based-pitch-detection
Deep learning based pitch detection

The folder consists of 5 directories

	dataset_creation:  contains,
		Files used to convert MIDI data to time domain signals 
		LSTM Pickle file creation
		LSTM Non-Overlapping time steps
		LSTM Overlapping time steps
		
	evaluation_matlabFiles: contains,
		script used to reconstruct the audio files
		Visualize the files created for real audio
		Visualize the files created for synthetic audio
		
	test_files: contains 2 directories,
		cnn, contains
			feed spectrogram as .png file
			feed spectrogram as .mat file 
		lstm, contains
			file to create Pickle file
			file to create Timesteps 
			test using traied LSTM network
	
	trained_nets: contains,
		overlapping 
		non-overlapping
		cnn network
		
	training_files:
		cnn architecture
		lstm architecture
		
		
Procedure:
CNN
Create datasets Inputs and Labels as .png file
Provide the path to CNN architecture file
Train the network
Test the network by passing the image in Matlab(applying CQT) and pass the saved image through CNN network.


LSTM
Create datasets Inputs and Labels using Matlab. (Matlab)
Inputs are the spectrogram as .png file and labels are mat files.
Create a pickle of 96xN by concatenating all spectrograms and mat files.(python)
Now create the timesteps 96x216 using python.
Provide the path of timesteps inputs and labels to LSTM architecture
Train the network

Test the network by passing spectrograms 
	Create 96xN of spectrograms using pickle
	Convert this to 96x216 timesteps
	Pass the file to test LSTM script
	save the network outputs
	Visulize the network outputs using Matlab.
	
	
	
	





