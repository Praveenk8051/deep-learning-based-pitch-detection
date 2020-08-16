close all;clear all;
saveAudio='E:\zThesisCodes_FreshImport\MatlabFiles\MIDIThings\saveAudio';
saveMidi='E:\zThesisCodes_FreshImport\MatlabFiles\MIDIThings\saveMatmidi';
minFreq = 8.1757989156;%C0
maxFreq = 3951.0664100490;%F8

Fs = 44100;
pickMidiformats = 'E:\zThesisCodes_FreshImport\MatlabFiles\MIDIThings\MIDI_renamedFiles';
filePattern = fullfile(pickMidiformats, '*.mid');
filesList = dir(filePattern);

for j = 9:length(filesList)

    midi = readmidi((fullfile(pickMidiformats, filesList(j).name)));
    Notes = midiInfo(midi,0);
    
%     Notes = Monomize(Notes);
    Notes = ReduceSilence(Notes);
    [filepath,name,ext] = fileparts(filesList(j).name);
    %save the notes as mat file, access later to create midi labels
    variable = Notes;
    
    %save(fullfile(saveMidi,[name '.mat']),'variable');
    
    L = 30;%length(Notes);
    K = 0;
    Notes = Notes(K+1:end,:);
    base = Notes(1,5);
    Notes(:,5) = Notes(:,5) - base;
    Notes(:,6) = Notes(:,6) - base;
    F = 440*(2.^((Notes(1:L,3)-69)/12));
    Md = Notes(1:L,3);
    n1 = round(Notes(1:L,5)*Fs/2048);
    n2 = round(Notes(1:L,6)*Fs/2048);
    
    vec = zeros(1,n2(end));
    vecmid = zeros(1,n2(end));
    for i = 1:length(n1)
        vec(n1(i)+1:n2(i)) = F(i);
    end
    [y,Fs]=midi2audio(Notes(1:L,:),Fs);
    
    
    [cfs,f,g,fshifts] = cqt(y(1:end),'SamplingFrequency',Fs,'FrequencyLimits',[minFreq maxFreq]);
    cfss = cfs.c;
    acfs = abs(cfss(1:size(cfss,1)/2+1,:));
    figure(1)
    imagesc(acfs);hold on;
    plot(Md);axis xy;hold off;
    %save the audio track, later slice it to 
    %audiowrite(fullfile(saveAudio,[name '.wav']),y,Fs)
    
%     figure(20);plot(vec);
    
    % [y,Fs]=midi2audio(midi,Fs,'sine');
%     len = length(vec);
%     window = 4096;
%     overlap = 2055;
%     nfft = 4096;
%     vec2 = vec(vec~=0);
%     
    
    
%     [C,F,T] = spectrogram(y,window,overlap,nfft,Fs);%ylim([0 1]);
%     while length(T)~=length(vec)
%         if length(T)<length(vec)
%             overlap = overlap+1;
%             [C,F,T] = spectrogram(y,window,overlap,nfft,Fs);%ylim([0 1]);
%         else
%             overlap = overlap-1;
%             [C,F,T] = spectrogram(y,window,overlap,nfft,Fs);%ylim([0 1]);
%         end
%     end
%     
%     
%     Fr = 4096/Fs;
%     K = round(3000*Fr);
%     F_K = F(2:K);
%     M_K = abs(C(2:K,:));
%     figure(2);image(T,F_K,M_K);hold on;plot(T,vec,'g'); axis xy;hold off;
end
% soundsc(y,Fs);