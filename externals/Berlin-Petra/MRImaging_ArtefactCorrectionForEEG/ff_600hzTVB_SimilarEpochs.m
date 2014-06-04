function [Xx,EEG]= ff_600hzTVB_SimilarEpochs(A,Xx,EEG)

h=waitbar(0,'calculating similarity measure (PSD)');
 
for a = 1:size(A,1)
    
    waitbar(a/size(A,1),h);
      
    A_diff =(A-repmat(A(a,:),size(A,1),1));
    A_diff_demeaned = A_diff - repmat(mean(A_diff'),size(A_diff,2),1)';
    
    A_diff_PSD = zeros(size(A_diff_demeaned,1),Xx.ppwelch.nfft/2+1);
    for a2 = 1:size(A_diff_demeaned,1)
        [A_diff_PSD(a2,:),F_psd] = pwelch(A_diff_demeaned(a2,Xx.interval_scan)',Xx.ppwelch.window,Xx.ppwelch.noverlap,Xx.ppwelch.nfft,Xx.srate);
    end
    
    [ss,ii] = sort(mean(A_diff_PSD(:,Xx.fft_index),2));
    
    Xx.simepindx(a,:) = ii(2:end);
    
end

close(h)

EEG.simepindx = Xx.simepindx;

EEG = pop_saveset(EEG,'filename',[Xx.file.eeg_original '.set'],'filepath',Xx.path.eeg_original,'check','on','savemode','twofiles');


