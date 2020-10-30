function [GErout,Erout,Eloc] = rout_efficiency(D,transform)
% ROUT_EFFICIENCY       Mean, pair-wise and local routing efficiency
%
%   [GErout,Erout,Eloc] = rout_efficiency(D,transform);
%
%   The routing efficiency is the average of inverse shortest path length.
%
%   The local routing efficiency of a node u is the routing efficiency
%   computed on the subgraph formed by the neighborhood of node u
%   (excluding node u).
%
%
%   Inputs:
%
%       D,
%           Weighted/unweighted directed/undirected
%           connection *weight* OR *length* matrix.
%
%       transform,
%           If the input matrix is a connection *weight* matrix, specify a
%           transform that map input connection weights to connection
%           lengths. Two transforms are available.
%               'log' -> l_ij = -log(w_ij)
%               'inv' -> l_ij =    1/w_ij
%
%           If the input matrix is a connection *length* matrix, do not
%           specify a transform (or specify an empty transform argument).
%
%
%   Outputs:
%
%       GErout,
%           Mean global routing efficiency (scalar).
%
%   	Erout,
%           Pair-wise routing efficiency (matrix).
%
%    	Eloc,
%           Local efficiency (vector)
%
%
%   Note:
%
%       The input matrix may be either a connection weight matrix, or a
%       connection length matrix. The connection length matrix is typically
%       obtained with a mapping from weight to length, such that higher
%       weights are mapped to shorter lengths (see above).
%
%
%   Algorithm:  Floyd–Warshall Algorithm
%
%
%   References:
%       Latora and Marchiori (2001) Phys Rev Lett
%       Goñi et al (2013) PLoS ONE
%       Avena-Koenigsberger et al (2016) Brain Structure and Function
%
%
%   Andrea Avena-Koenigsberger and Joaquin Goñi, IU Bloomington, 2012
%

%   Modification history
%   2012 - original
%   2016 - included comutation of local efficiency
%   2016 - included transform variable that maps strengths onto distances


if ~exist('transform','var')
    transform = [];
end

n=length(D);                                            % number of nodes

Erout = distance_wei_floyd(D,transform);               	% pair-wise routing efficiency
Erout = 1./Erout;
Erout(eye(n)>0) = 0;
GErout = sum(Erout(~eye(n)>0))/(n^2-n);                 % global routing efficiency

if nargout == 3
    Eloc = zeros(n,1);
    for u = 1:n
        Gu = find(D(u,:) | D(:,u).');                 	% u's neighbors
        nGu = length(Gu);
        e = distance_wei_floyd(D(Gu,Gu),transform);
        Eloc(u) = sum(sum(1./e(~eye(nGu)>0)))/nGu;     	% efficiency of subgraph Gu
    end
end
