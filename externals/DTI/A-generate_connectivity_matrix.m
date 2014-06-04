function num_brain_voxels = generate_connectivity_matrix(roi_labels,indir,outdir,prefix)
%%%
%%% INPUTS: 
%%%        indir - directory containing bunch of .txt files 
%%%                for all pairwise connections, and segmented volumetric 
%%%                brain image - to be used for estimating 
%%%                total barain volume and resolution - these two will be used 
%%%                 for normalization of capacity (flow) of cennections
%%%      
%%%        roi_labels - roi lebels in Gleb's order
%%%        prefix - subj name
%%%
%%% OUPUTS:
%%%        write matrix of pairwise path lengths as csv file
%%%
%%%        write matrix of pairwise path capacities as csv file, normalized


num_rois = numel(roi_labels);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% write path length matrix (symmetric)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% put -1 if there is no cocomac connection, so erin didn't look for it
%% put 0 if there is a cocomac con3enction, but erins pipleine didn't find it
pathlength_matrix = zeros(num_rois) - 1;
for r1=1:(num_rois-1)
  [r1_str,err] = sprintf('%03d',roi_labels(r1)); % zeropaded string for r1
  for r2=(r1+1):num_rois
    [r2_str,err] = sprintf('%03d',roi_labels(r2)); % zeropaded string for r2
    fname1 = [indir prefix '_' r1_str '_' r2_str '_AvgLength.txt'];
    fname2 = [indir prefix '_' r2_str '_' r1_str '_AvgLength.txt'];
    fname = [];
    if exist(fname1,'file'), fname = fname1; 
    elseif exist(fname2,'file'), fname = fname2; 
    end
    if ~isempty(fname)
      pathlength_matrix(r1,r2) = dlmread(fname);
      pathlength_matrix(r2,r1) = pathlength_matrix(r1,r2);
    else
    end
  end
end

%% add first row = header = roi_labels
pathlength_matrix = [roi_labels'; pathlength_matrix];

%% save
csvwrite([outdir prefix '_PathlengthMatrix.csv'],pathlength_matrix);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% write capacity matrix (symmetric)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% put -1 if there is no cocomac connection, so erin didn't look for it
%% put 0 if there is a cocomac connenction, but erins pipleine didn't find it
capacity_matrix = zeros(num_rois) - 1;
for r1=1:(num_rois-1)
  [r1_str,err] = sprintf('%03d',roi_labels(r1)); % zeropaded string for r1
  for r2=(r1+1):num_rois
    [r2_str,err] = sprintf('%03d',roi_labels(r2)); % zeropaded string for r2
    fname1 = [indir prefix '_' r1_str '_' r2_str '_AvgFlow.txt'];
    fname2 = [indir prefix '_' r2_str '_' r1_str '_AvgFlow.txt'];
    fname = [];
    if exist(fname1,'file'), fname = fname1; 
    elseif exist(fname2,'file'), fname = fname2; 
    end
    if ~isempty(fname)
      capacity_matrix(r1,r2) = dlmread(fname);
      capacity_matrix(r2,r1) = capacity_matrix(r1,r2);
    else
    end
  end
end

%% Capacity matrix must be normalized because number of fibers depends on the number of voxels
nii = load_nii([indir prefix '_seg_to_dti.img']); % load segmented brain
num_brain_voxels = numel(find(nii.img>0));
capacity_matrix = capacity_matrix / log(num_brain_voxels);

%% add first row = header = roi_labels
capacity_matrix = [roi_labels'; capacity_matrix];

%% save
csvwrite([outdir prefix '_CapacityMatrix.csv'],capacity_matrix);
