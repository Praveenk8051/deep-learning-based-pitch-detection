function y=fmsynth(freq,dur,amp,Fs,filt)
% y=fmsynth(freq,dur,amp,Fs,type)
%
% Synthesize a single note
%
% Inputs:
%  freq - frequency in Hz
%  dur - duration in seconds
%  amp - Amplitude in range [0,1]
%  Fs -  sampling frequency in Hz

if nargin<5
  error('Five arguments required for synth()');
end

N = floor(dur*Fs);

if N == 0
  warning('Note with zero duration.');
  y = [];
  return;

elseif N < 0
  warning('Note with negative duration. Skipping.');
  y = [];
  return;
end

n=0:N-1;


rn = randn(1);
t = 0:(1/Fs):dur;
envel = interp1([0 dur/2 dur/4 dur/8 dur/16 dur/9 dur/7 dur/3 dur], [0 1 .9 .8 .7 .8 .9 1 0], 0:(1/Fs):dur);
I_env = 5.*envel;%5
s = 0.9;%0.7+randn(1)/5;
%y = s.*sin(2.*pi.*freq.*t);% 
y = s*envel.*sin(2.*pi.*freq.*t + I_env.*sin(2.*pi.*freq.*t));% + (amp/10)*sin(2*pi.*(freq+rn).*t) + (amp/10)*sin(2*pi.*(freq-rn).*t);
%y = s*envel.*sin(2.*pi.*freq.*t);
%% Harmonic Parts %%
row = [2,4,8,16,32,64,128];
%     row = [2,4,6,8,12,16,20,24,32,48,64,80,96,128,256];

for i = row(1:end)
    rn = max(-1,randn(1));
    rn = min(1,rn);
    hars = 0;
    y = y + (1/i)*envel.*(sin(2.*pi.*i*freq.*t + I_env.*sin(2.*pi.*i*freq.*t)));
    s = s + (1/i);
end
y = y/s;

% smooth edges w/ 10ms ramp
if (dur > .015)
  L = fix((9+randn(1)/10)*dur/10*Fs)+1;  % L odd
  if rand(1)>0.1
      ramp = kaiser(L)';  % odd length
  else
      ramp = bartlett(L)';
  end
%   L = ceil(L/2);
  y(1:L) = y(1:L) .* ramp(1:L);
  y(end-L+1:end) = y(end-L+1:end) .* ramp(end-L+1:end);
end
% figure;plot(y);
% y = filtfilt(filt.b,filt.a,y);
% figure(1);plot(y);

y = filter(filt.b,filt.a,y);
% figure(31);plot(y);
% figure(30);freqz([filt.b,filt.a]);
% disp(1);
