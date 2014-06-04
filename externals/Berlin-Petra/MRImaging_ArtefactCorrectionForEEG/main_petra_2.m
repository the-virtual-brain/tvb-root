function [EEG, EEG_corr, Xx] = main_petra_2(eeg_original, eeg_corrected)
%UNTITLED main function to call second petra algorithm
%   Detailed explanation goes here
    eeglab_path = which('eeglab');
    
    eeg_original = '/Users/bogdan/Documents/TVB/tvb-root/tvb/trunk/externals/Berlin-Petra/MRImaging_ArtefactCorrectionForEEG/testdata/original';
    eeg_corrected = '/Users/bogdan/Documents/TVB/tvb-root/tvb/trunk/externals/Berlin-Petra/MRImaging_ArtefactCorrectionForEEG/testdata/corrected';
    
    if isempty(eeglab_path)
        disp('  ');
        disp('SCRIPT ABORTED - EEGLab toolbox is missing')
        disp('please download EEGLab on http://sccn.ucsd.edu/eeglab/) and add to your Matlab search path');
        return;       
    end;
    
    ff_600hzTVB_Init;
    Xx.path.eeg_original = eeg_original;
    Xx.path.eeg_corrected = eeg_corrected;
    Xx.plot = 'no';
    
    addpath(strrep(eeglab_path, 'eeglab.m', 'functions/adminfunc'));
    addpath(strrep(eeglab_path, 'eeglab.m', 'functions/guifunc'));
    addpath(strrep(eeglab_path, 'eeglab.m', 'functions/javachatfunc'));
    addpath(strrep(eeglab_path, 'eeglab.m', 'functions/miscfunc'));
    addpath(strrep(eeglab_path, 'eeglab.m', 'functions/octavefunc'));
    addpath(strrep(eeglab_path, 'eeglab.m', 'functions/popfunc'));
    addpath(strrep(eeglab_path, 'eeglab.m', 'functions/resources'));
    addpath(strrep(eeglab_path, 'eeglab.m', 'functions/sigprocfunc'));
    addpath(strrep(eeglab_path, 'eeglab.m', 'functions/statistics'));
    addpath(strrep(eeglab_path, 'eeglab.m', 'functions/studyfunc'));
    addpath(strrep(eeglab_path, 'eeglab.m', 'functions/timefreqfunc'));
    
    [EEG,EEG_corr,Xx] = ff_600hzTVB_MRCorr(Xx);

end

