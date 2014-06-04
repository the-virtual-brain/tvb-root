% ff_600hzTVB_MRCorr
% MR artifact correction pipeline for EEG data
% EEG data needs to support EEGLab files structure!
% needs EEGLab toolbox --> http://sccn.ucsd.edu/eeglab/

function [EEG,EEG_corr,Xx] = ff_600hzTVB_MRCorr(Xx)

disp(' ');
disp('---------------------- MR artifact correction -----------------------')
disp('(see Freyer et al., Neuroimage. 2009 Oct 15;48(1):94-108) for details)');
disp(' ');

disp(['correcting file ' Xx.file.eeg]);
disp(' ');

% Load EEG from EEGLab-File
EEG = pop_loadset([Xx.file.eeg '.set'],Xx.path.eeg_original);
EEG_corr = EEG;

% extract MR-onset Markers
MRmarker = [];
for E=1:length(EEG.event)
    if strcmp(EEG.event(E).type,'Scan Start')
        MRmarker(end+1)=EEG.event(E).latency;
    end
end

freqs = EEG.srate/2*linspace(0,1,Xx.nfft/2);
Xx.fft_index = find(freqs>Xx.freqrange(1) & freqs<Xx.freqrange(2));
Xx.srate = EEG.srate;
Xx.nepochs = length(MRmarker);
Xx.interval = [Xx.offset round(mean(diff(MRmarker)))+Xx.offset];

%iterate EEG channels
for ch = 1:size(EEG.data,1)
    
    %Channels
    disp(['Channel #' num2str(ch) ' - ' num2str(EEG.chanlocs(ch).labels)]);

    A = zeros(Xx.nepochs,(MRmarker(2)-MRmarker(1)));
    
    h=waitbar(0,'segmentation');
    for a=1:Xx.nepochs
        waitbar((a/Xx.nepochs),h);
        A(a,:)=EEG.data(ch,MRmarker(a)+Xx.interval(1):(MRmarker(a)+Xx.interval(2))-1);
        %A_demeaned(a,:)=(A(a,:)-mean(A(a,:)));
    end
    close(h)
    
    [A_base,base_A] = ff_600hzTVB_BaselineCorr(A,[1 size(A,2)]);
    
    % calculate similarity measure (PSD) to create epoch index for template
    % if epoch index EEG.simepindx already exists, this step is skipped
    if ~isfield(EEG,'simepindx')
        [Xx,EEG]= ff_600hzTVB_SimilarEpochs(A,Xx,EEG);
    end
    
    B=zeros(size(A));
    C=zeros(size(A));
    EEG_corr.data(ch,:) = EEG.data(ch,:);
    
    h=waitbar(0,'template subtraction)');
    
    w = Xx.weighting.^(1:Xx.nepochstempl);
    
    for a = 1:Xx.nepochs
        
        waitbar(a/size(A,1),h);
        B_fore=(w*A_base(EEG.simepindx(a,1:Xx.nepochstempl),:));
        B(a,:)=B_fore';
        B(a,:)=B(a,:)./sum(w);
        C(a,:)=A(a,:)-B(a,:);
        
        EEG_corr.data(ch,MRmarker(a)+Xx.interval(1):(MRmarker(a)+Xx.interval(2))-1) =  C(a,:);
        
    end
    
    close(h);
    
    %PCA subraction START (only if flag is on)
    if(strcmp(Xx.pca,'yes')) == 1;
        
        options.numpc = 1;
        
        [C_base,base_C] = ff_600hzTVB_BaselineCorr(C,[1 size(A,2)]);
        
        [ppca.coeff, ppca.scores, ppca.latent] = princomp(C_base','econ');
        
        for a = 1:size(ppca.scores,2)
            [ppca.scores_psd(a,:),F_psd] = pwelch(ppca.scores(Xx.interval_scan,a),Xx.ppwelch.window,Xx.ppwelch.noverlap,Xx.ppwelch.nfft,Xx.srate);
        end
        
        [ss,ii] = sort(mean(ppca.scores_psd(:,Xx.fft_index),2),'descend');
        
        ppca.pcs = ppca.scores(:,ii(1:options.numpc))*ppca.coeff(:,ii(1:options.numpc))';
        
%         clear ii ss ppca
        
        for a = 1:Xx.nepochs
            
            C_pca(a,:)= C(a,:)-ppca.pcs(:,a)';
            
            EEG_corr.data(ch,MRmarker(a)+Xx.interval(1):(MRmarker(a)+Xx.interval(2))-1) =  C_pca(a,:);
            
        end
        
    end
    %PCA subraction END
    
    clear A* B* C*
    
end

EEG_corr = pop_saveset(EEG_corr,'filename',[Xx.file.eeg '_corr.set'],'filepath',Xx.path.eeg_corrected,'check','on','savemode','twofiles');

%plot results (only if flag is on)
if(strcmp(Xx.plot,'yes')) == 1;
    eegplot(EEG.data,'srate',EEG.srate,'eloc_file',EEG.chanlocs,'color',{'r' 'g' 'b' 'k'});
    set(gcf,'Name',[Xx.file.eeg '_UNCORRECTED'] );
    eegplot(EEG_corr.data,'srate',EEG.srate,'eloc_file',EEG.chanlocs,'color',{'r' 'g' 'b' 'k'});
    set(gcf,'Name',[Xx.file.eeg '_CORRECTED'] );
end
