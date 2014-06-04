%add path to scripts
addpath('\\naas\synchrony\_Projects\TVB\SkripteFrank\600hzTVB')

%set parameters, paths, filenames
ff_600hzTVB_Init;

%execute MR artifact correction 
[EEG,EEG_corr,Xx] = ff_600hzTVB_MRCorr(Xx);
