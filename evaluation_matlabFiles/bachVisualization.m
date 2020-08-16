clear all

%% Load the paths
loadLstmdata='Enter the path';
loadSpectrumData='Enter the path';


saveFiles='Enter the path';

filePattern1 = fullfile(loadLstmdata, '*.mat');
filesList1 = dir(filePattern1);


filePattern2 = fullfile(loadSpectrumData, '*.png');
filesList2 =dir(filePattern2);
num=0;
for i=1:50
    close all
    
    num=num+1;
    
    %% Load the data
    lstm=load((fullfile(loadLstmdata, filesList1(i).name)));
    image=imread((fullfile(loadSpectrumData, filesList2(i).name)));
    
    disp((fullfile(loadLstmdata, filesList1(i).name)))
    disp((fullfile(loadSpectrumData, filesList2(i).name)))
    
    
    lstm_=(lstm.variable);
    
    %% Replace the silence path in order to flip up side down
    for i =1:216
        math(1,i) =find(lstm_(:,i),1);
    end
    for i =1:216
        if math(1,i)==1
            math(1,i)=96;
        end
    end
    
    %% Plots and save the image
    figure(1)
    imagesc(imadjust(image))
    
    
    hold on
    
    plot(math,'r--','LineWidth',2.5)
    hold off
    axis off
    
    set(gca,'LooseInset',get(gca,'TightInset'),'XLimSpec', 'Tight','YLimSpec', 'Tight')
    saveas(gcf, fullfile(saveFiles,['noisy' num2str(num) '.png']))
end







