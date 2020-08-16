close all;
clear all;
minFreq = 8.1757989156;
maxFreq = 1760.0000000000;
Fs = 44100;

saveSpectrogam = 'Enter the path';
saveAudio = 'Enter the path';
saveMidi='Enter the path';

pickMidiformats = 'Enter the path';
filePattern = fullfile(pickMidiformats, '*.mid');
filesList = dir(filePattern);

pickAudioformats = 'Enter the path';
filePattern1 = fullfile(pickAudioformats, '*.wav');
filesList1 = dir(filePattern1);

num = 0;

for k=1:26
    %% Convert to time domain samples
    midi = readmidi((fullfile(pickMidiformats, filesList(k).name)));
    Notes = midiInfo(midi,0);
    Notes = ReduceSilence(Notes);
    Nnotes = size(Notes,1);
    n1 = Notes(:,5);
    n2 = Notes(:,6);
    N = Notes(:,3);
    
      
    
    [y,Fs]=midi2audio(Notes(1:end,:),Fs);
    y_write=y;
    %% Save the time domain samples
    audiowrite(fullfile(saveAudio,['track' num2str(num) '.wav']),y_write,Fs);
    
    
    %% Read the corresponding audio for duration
    aud=audioinfo(fullfile(saveAudio,['track' num2str(num) '.wav']));
    
    
    %% Find CQT and form Midi labels for chunks of audio
    for j = 1:floor(aud.Duration)
        num = num+1;
        y_=y((j-1)*Fs+1:j*Fs);
        [cfs,f,g,fshifts] = cqt(yNoise,'SamplingFrequency',Fs,'FrequencyLimits',[minFreq maxFreq],'Window','hamming','BinsPerOctave',12);
        cfss = cfs.c;
        acfs = abs(cfss(1:size(cfss,1)/2+1,:));
        
        for i = 1:numel(n1)
            t1 = ceil(n1(i)*216)+1;
            t2 = ceil(n2(i)*216);
            tempvec(1,t1:t2) = N(i);
        end
        
        tempvec = tempvec(1,(j-1)*size(acfs,2)+1:j*size(acfs,2));
        
        imwrite(flipud(acfs),fullfile(saveSpectrogam,['image' num2str(num) '.png']))
        figure(1);
        imagesc(flipud(acfs))
        
        temp = zeros(size(acfs,1),size(acfs,2));
        for i = 1:size(acfs,2)
            temp(tempvec(i)+1,i) = 1;
        end
        figure(2);imagesc(flipud(temp))
        variable = flipud(temp);
        save(fullfile(saveMidi,['midi' num2str(num) '.mat']),'variable');
    end
end


