%%%%% This function takes output from Erin's DTI pipeline 
%%%%% and generates connectivity matrices for Viktor.
addpath('/phocaea/mcintosh_lab/natasa/copy_plsgui/');

% subject directories containing all DTI pipeline outputs
indirs = {'/flora/mcintosh_lab/erin/sb/processed_at_rri/dti1/E1751/','/flora/mcintosh_lab/erin/sb/processed_at_rri/dti1/E2460/','/flora/mcintosh_lab/erin/sb/processed_at_sb/dti1/processed/E1788/','/flora/mcintosh_lab/erin/sb/processed_at_sb/dti1/processed/E1813/','/flora/mcintosh_lab/erin/sb/processed_at_sb/dti1/processed/E2237/','/flora/mcintosh_lab/erin/sb/processed_at_sb/dti1/processed/E2322/','/flora/mcintosh_lab/erin/sb/processed_at_sb/dti1/processed/E2562/'};
prefixes = {'E1751','E2460','E1788','E1813','E2237','E2322','E2562'};

% this is where we'll write connectivity matrices for all subjects
outdir = '/rri_disks/phocaea/mcintosh_lab/natasa/dti_for_viktor/matrices_for_viktor/';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% read in roi_lables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% in order in which Gleb has them in the cocomac connectivity mtarix. 
% That is the order in which we want to assemble erin's data for viktor

roi_labels = dlmread('roi_labels.txt');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% for each subject assemble connectivity matrices (pathlength and capacity)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_subjects = numel(indirs);
for s=1:num_subjects
  num_brain_voxels = generate_connectivity_matrix(roi_labels,indirs{s},outdir,prefixes{s});
  disp([prefixes{s} ' ' num2str(num_brain_voxels)]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% visulize matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
for s=1:num_subjects
  subplot(3,3,s);
  a = csvread([outdir prefixes{s} '_PathlengthMatrix.csv']);
  a = a(2:end,:); % remove header line
  imagesc(a,[0 160]);colorbar;
  axcopy(gcf);
end

figure;
for s=1:num_subjects
  subplot(3,3,s);
  a = csvread([outdir prefixes{s} '_CapacityMatrix.csv']);
  a = 10000*a(2:end,:); % remove header line
  imagesc(a,[0 12000]);colorbar;
  axcopy(gcf);
end



