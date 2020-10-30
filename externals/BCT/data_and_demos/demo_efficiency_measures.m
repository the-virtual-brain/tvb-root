% This script loads 7 unweighted, undirected adjacency matrices
% corresponding to a clique, chain, ring, 1D lattice, star, rich-club, and
% bi-modular toy networks (all graphs have 50 nodes). The following
% efficiency measures are computed for each graph:
%
%  - prob_SPL: probability of one particle traveling through shortest-paths
%  - Erout: efficiency of routing -> based on shortest-paths
%  - Ediff: efficiency of diffusion -> based on mean-first-passage-times
%  - Eres: efficiency of resources -> based on number of particles
%    necessary so that at least one particle taking shortest-paths with
%    certain probability (lambda).
%
%  If you are using this efficiency package for your research, plase kindly
%  cite the paper:
%
%  "Exploring the Morphospace of Communication Efficiency in Complex
%  Networks" Goñi J, Avena-Koenigsberger A, Velez de Mendizabal N, van den
%  Heuvel M, Betzel RF and Sporns O. PLoS ONE. 2013
%
%  These examples and results correspond to Table 1 in the paper.
%
%  Joaquin Goñi and Andrea Avena-Koenigsberger, IU Bloomington, 2012

close all;
clear all;
clc;

load demo_efficiency_measures_data.mat; % 7 adjacency matrices corresponding to the examples shown in Table 1 are loaded.
lambda = 0.5; % this parameter is an input for the computation of Eres.

% run and display efficiency measures for the 7 graphs
disp(['    prob_SPL  ','  Erout  ','  Ediff  ','  Eres  '])

fprintf('----- clique ----- \n')
adj = clique;
N = size(adj,1);
EYE = logical(eye(N,N));
SPL = distance_wei_floyd(adj);
Erout = rout_efficiency(adj);
Ediff = diffusion_efficiency(adj);
[Eres,prob_SPL] = resource_efficiency_bin(adj,lambda,SPL);
prob_SPL = mean(prob_SPL(~EYE));
Eres = mean(Eres(~EYE));
disp([prob_SPL,Erout,Ediff,Eres])

fprintf('----- chain ----- \n')
adj = chain;
N = size(adj,1);
EYE = logical(eye(N,N));
SPL = distance_wei_floyd(adj);
Erout = rout_efficiency(adj);
Ediff = diffusion_efficiency(adj);
[Eres,prob_SPL] = resource_efficiency_bin(adj,lambda,SPL);
prob_SPL = mean(prob_SPL(~EYE));
Eres = mean(Eres(~EYE));
disp([prob_SPL,Erout,Ediff,Eres])

fprintf('----- ring ----- \n')
adj = ring;
N = size(adj,1);
EYE = logical(eye(N,N));
SPL = distance_wei_floyd(adj);
Erout = rout_efficiency(adj);
Ediff = diffusion_efficiency(adj);
[Eres,prob_SPL] = resource_efficiency_bin(adj,lambda,SPL);
prob_SPL = mean(prob_SPL(~EYE));
Eres = mean(Eres(~EYE));
disp([prob_SPL,Erout,Ediff,Eres])

fprintf('----- lattice1D ----- \n')
adj = lattice1D;
N = size(adj,1);
EYE = logical(eye(N,N));
SPL = distance_wei_floyd(adj);
Erout = rout_efficiency(adj);
Ediff = diffusion_efficiency(adj);
[Eres,prob_SPL] = resource_efficiency_bin(adj,lambda,SPL);
prob_SPL = mean(prob_SPL(~EYE));
Eres = mean(Eres(~EYE));
disp([prob_SPL,Erout,Ediff,Eres])

fprintf('----- star ----- \n')
adj = star;
N = size(adj,1);
EYE = logical(eye(N,N));
SPL = distance_wei_floyd(adj);
Erout = rout_efficiency(adj);
Ediff = diffusion_efficiency(adj);
[Eres,prob_SPL] = resource_efficiency_bin(adj,lambda,SPL);
prob_SPL = mean(prob_SPL(~EYE));
Eres = mean(Eres(~EYE));
disp([prob_SPL,Erout,Ediff,Eres])

fprintf('----- rich-club ----- \n')
adj = rich_club;
N = size(adj,1);
EYE = logical(eye(N,N));
SPL = distance_wei_floyd(adj);
Erout = rout_efficiency(adj);
Ediff = diffusion_efficiency(adj);
[Eres,prob_SPL] = resource_efficiency_bin(adj,lambda,SPL);
prob_SPL = mean(prob_SPL(~EYE));
Eres = mean(Eres(~EYE));
disp([prob_SPL,Erout,Ediff,Eres])

fprintf('----- bi-modular ----- \n')
adj = bi_modular;
N = size(adj,1);
EYE = logical(eye(N,N));
SPL = distance_wei_floyd(adj);
Erout = rout_efficiency(adj);
Ediff = diffusion_efficiency(adj);
[Eres,prob_SPL] = resource_efficiency_bin(adj,lambda,SPL);
prob_SPL = mean(prob_SPL(~EYE));
Eres = mean(Eres(~EYE));
disp([prob_SPL,Erout,Ediff,Eres])
