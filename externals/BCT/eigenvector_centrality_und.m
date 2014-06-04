function   v = eigenvector_centrality_und(CIJ)
%EIGENVECTOR_CENTRALITY_UND      Spectral measure of centrality
%
%   v = eigenvector_centrality_und(CIJ)
%
%   Eigenector centrality is a self-referential measure of centrality:
%   nodes have high eigenvector centrality if they connect to other nodes
%   that have high eigenvector centrality. The eigenvector centrality of
%   node i is equivalent to the ith element in the eigenvector 
%   corresponding to the largest eigenvalue of the adjacency matrix.
%
%   Inputs:     CIJ,        binary/weighted undirected adjacency matrix.
%
%   Outputs:      v,        eigenvector associated with the largest
%                           eigenvalue of the adjacency matrix CIJ.
%
%   Reference: Newman, MEJ (2002). The mathematics of networks.
%
%   Xi-Nian Zuo, Chinese Academy of Sciences, 2010
%   Rick Betzel, Indiana University, 2012

n = length(CIJ) ;
if n < 1000
    [V,~] = eig(CIJ) ;
    ec = abs(V(:,n)) ;
else
    [V, ~] = eigs(sparse(CIJ)) ;
    ec = abs(V(:,1)) ;
end
v = reshape(ec, length(ec), 1);