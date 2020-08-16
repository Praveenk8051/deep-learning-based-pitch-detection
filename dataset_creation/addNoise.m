function y_noise = addNoise(Signal, noiseLevel)

%% Create uniform noise along the length of the signal
Npts = length(Signal);
Noise = rand(1,Npts);

figure(98)
[Signal_,t]=Normalize(Signal,44100);
plot(t,Signal);

%% Calculate the power and add noise 
Signal_Power = sum(abs(Signal).*abs(Signal))/Npts;
Noise_Power = sum(abs(Noise).*abs(Noise))/Npts;
Noise = Noise.*sqrt(Signal_Power/Noise_Power);
Noise_Power = sum(abs(Noise).*abs(Noise))/Npts;

K = (Signal_Power/Noise_Power)*10^((-noiseLevel)/10);
%% Calculate the OLD SNR
Old_SNR = 10*(log10(Signal_Power/Noise_Power));
disp(Old_SNR)

%% Calculate the NEW SNR (verification)
New_Noise = sqrt(K)*Noise;
New_Noise_Power = sum(abs(New_Noise).*abs(New_Noise))/Npts;
New_SNR = 10*(log10(Signal_Power/New_Noise_Power));
disp(New_SNR)

Noisy_Signal = Signal + New_Noise;
[Noisy_Signal_,t]=Normalize(Signal,44100);

%% FFT plots for the verfication
%X=abs(fft(Signal));
% figure(99)
% plot(t,Noisy_Signal);
% figure(99);spectrogram(Signal,2048,2048-256,2048,44100,'yaxis');
%figure(100);spectrogram(Noisy_Signal,2048,2048-256,2048,44100,'yaxis');
%Y=abs(fft(Noisy_Signal));
%figure(7);plot(Y,'b','LineWidth',1.5);hold on;plot(X,'r','LineWidth',0.5);hold off;
% title('Noisy')

%% Return the noisy signal 
y_noise=Noisy_Signal;
end