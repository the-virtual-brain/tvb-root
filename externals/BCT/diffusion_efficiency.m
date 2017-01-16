function [GEdiff,Ediff] = diffusion_efficiency(adj)
% DIFFUSION_EFFICIENCY      Global mean and pair-wise diffusion efficiency
%
%   [GEdiff,Ediff] = diffusion_efficiency(adj);
%
%   The diffusion efficiency between nodes i and j is the inverse of the
%   mean first passage time from i to j, that is the expected number of
%   steps it takes a random walker starting at node i to arrive for the
%   first time at node j. Note that the mean first passage time is not a
%   symmetric measure -- mfpt(i,j) may be different from mfpt(j,i) -- and
%   the pair-wise diffusion efficiency matrix is hence also not symmetric.
%
%
%   Input:
%       adj,    Weighted/Unweighted, directed/undirected adjacency matrix
%
%
%   Outputs:
%       GEdiff, Mean Global diffusion efficiency (scalar)
%       Ediff,  Pair-wise diffusion efficiency (matrix)
%
%
%   References: Goñi J, et al (2013) PLoS ONE
%
%   Joaquin Goñi and Andrea Avena-Koenigsberger, IU Bloomington, 2012


n = size(adj,1);
mfpt = mean_first_passage_time(adj);
Ediff = 1./mfpt;
Ediff(eye(n)>0) = 0;
GEdiff = sum(Ediff(~eye(n)>0))/(n^2-n);

