clear all

%% Load the paths
originalMidi='Enter the path';
loadLstmdata='Enter the path';
loadSpectrumData='Enter the path';


filePattern1 = fullfile(loadLstmdata, '*.mat');
filesList1 = dir(filePattern1);


filePattern2 = fullfile(originalMidi, '*.mat');
filesList2 = dir(filePattern2);

filePattern3 = fullfile(loadSpectrumData, '*.png');
filesList3 =dir(filePattern3);
x=0;
for i=1:50
    
    %% Load the data
    lstm=load((fullfile(loadLstmdata, filesList1(i).name)));
    origMidi=load((fullfile(originalMidi, filesList2(i).name)));
    image=imread((fullfile(loadSpectrumData, filesList3(i).name)));
    
    
    
    disp((fullfile(loadLstmdata, filesList1(i).name)))
    disp((fullfile(loadSpectrumData, filesList3(i).name)))
    disp((fullfile(originalMidi, filesList2(i).name)))
    
    
    lstm_=lstm.variable;
    midi_=origMidi.variable;
    
    
    %% Plots for comparison
    figure(1);
    imagesc((midi_))
    
    figure(2);
    imagesc(((lstm.variable)))
    
    figure(3);
    imagesc(image)
    
end







