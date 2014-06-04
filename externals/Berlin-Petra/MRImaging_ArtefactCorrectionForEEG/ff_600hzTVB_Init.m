Xx.path.eeglab = which('eeglab');

if isempty(Xx.path.eeglab)
    disp('  ');
    disp('SCRIPT ABORTED - EEGLab toolbox is missing')
    disp('please download EEGLab on http://sccn.ucsd.edu/eeglab/) and add to your Matlab search path');
    break    
end

Xx.path.eeg_original = 'testdata/original';
Xx.path.eeg_corrected = 'testdata/corrected';

% EEGLAB set must contain data and 'Scan Start' markers for the start of each
% volume.
Xx.file.eeg = 'VP1';

Xx.pca = 'no';
Xx.plot = 'yes';



%parameters for MR artifact correction
Xx.offset = -250;
Xx.interval_scan = [1:8192];
Xx.interval_nonscan = [9001:17192];
Xx.nepochstempl = 30;
Xx.weighting = 0.9;
Xx.freqrange = [400 600];
Xx.nfft = 2^nextpow2(length(Xx.interval_scan))/2;
clear nfft_temp freqs freqrange;
Xx.ppwelch.window = length(Xx.interval_scan)/2;
Xx.ppwelch.noverlap = length(Xx.interval_scan)/4;
Xx.ppwelch.nfft = Xx.nfft;