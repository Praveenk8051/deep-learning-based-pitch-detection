close all
clear all
%% read the audio/cqt
minFreq = 8.1757989156;
maxFreq = 1760.0000000000;
[y,fs] = audioread('Enter the path');
y_tempc=y(1*fs:2*fs);
X=abs(fft(y_tempc));

%% Cqt for clean signal
[cfs1,~,g1,fshifts1] = cqt(y_tempc,'SamplingFrequency',fs,'FrequencyLimits',[minFreq maxFreq]);

%% Add noise
y_temp=addNoise(y_tempc,10);

%% Icqt for noisy signal
[cfs,~,g,fshifts] = cqt(y_temp,'SamplingFrequency',fs,'FrequencyLimits',[minFreq maxFreq]);
ytemp2=icqt(cfs,g,fshifts);

%% Save the noisy signal
audiowrite('Enter the path',ytemp2,fs);


%% FFT plots for noisy and clean
Xn=abs(fft(ytemp2));
figure(8);plot(Xn(1:(2.2051e+04)),'b','LineWidth',1.5);hold on;plot(X(1:(2.2051e+04)),'r','LineWidth',0.5);
legend('Noisy Signal','Clean Signal');
hold off;
figure(75);
plot(t,ytemp2);
%% save it as png
cfss = cfs.c;
posCoeff_freq = abs(cfss(1:size(cfss,1)/2+1,:));
negCoeff_freq = abs(cfss(size(cfss,1)/2+1:end,:));
a=(flipud(posCoeff_freq));
figure(1)
% imshow(a)

imagesc(a)
% imwrite((flipud(posCoeff_freq)),'E:\zThesisCodes_FreshImport\MatlabFiles\test_scripts\image1.png')
a=a*255;
save('a.mat','a');
disp('done')
%% Load the data using mat or image file
% image = imread('Enter the path');
image=load('matfile.mat');
image=image.a;


image = flipud(image);
image = image(1:end-1,:)/255;

%% Multiply the phase of the signal as x+iy
figure(2);
imagesc((flipud(image)));

resize = image;
phase = angle(cfs1.c);

resize1 = resize.*exp(1i*phase(1:size(cfss,1)/2,:));
resize2 = conj(resize1);
resize = [resize1;flipud(resize2)];

rec = resize;
cfs1.c = rec;

xrec = icqt(cfs1,g1,fshifts1);

figure(3);
plot(t,xrec);
Y=abs(fft(xrec));
figure(7);plot(Y(1:(2.2051e+04)),'b','LineWidth',1.5);hold on;plot(X(1:(2.2051e+04)),'r','LineWidth',0.5);
legend('Denoised Signal','Clean Signal');
hold off;

%% Save the noisy signal
audiowrite('E:\zThesisCodes_FreshImport\MatlabFiles\test_scripts\clean3_.wav',xrec,fs);


