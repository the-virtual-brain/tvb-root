function [SPL,hops,Pmat] = distance_wei_floyd(D,transform)
% DISTANCE_WEI_FLOYD        Distance matrix (Floyd-Warshall algorithm)
%
%   [SPL,hops,Pmat] = distance_wei_floyd(D,transform)
%
%   Computes the topological length of the shortest possible path
%   connecting every pair of nodes in the network.
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
%       SPL,
%           Unweighted/Weighted shortest path-length matrix.
%           If W is directed matrix, then SPL is not symmetric.
%
%       hops,
%           Number of edges in the shortest path matrix. If W is
%           unweighted, SPL and hops are identical.
%
%       Pmat,
%           Elements {i,j} of this matrix indicate the next node in the
%           shortest path between i and j. This matrix is used as an input
%           argument for function 'retrieve_shortest_path.m', which returns
%           as output the sequence of nodes comprising the shortest path
%           between a given pair of nodes.
%
%
%   Notes:
%
%       There may be more than one shortest path between any pair of nodes
%       in the network. Non-unique shortest paths are termed shortest path
%       degeneracies, and are most likely to occur in unweighted networks.
%       When the shortest-path is degenerate, The elements of matrix Pmat
%       correspond to the first shortest path discovered by the algorithm.
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
%   Andrea Avena-Koenigsberger, IU, 2012

%   Modification history
%   2016 - included transform variable that maps weights to lengths

if exist('transform','var') && ~isempty(transform)
    
    switch transform
        
        case 'log'
            
            if any((D<0) & D>1)
                error('connection-strengths must be in the interval [0,1) to use the transform -log(w_ij) \n')
            else
                SPL = -log(D);
            end
            
        case 'inv'
            
            SPL = 1./D;
            
        otherwise
            
            error('Unexpected transform type. Only "log" and "inv" are accepted \n')
    end
    
else    % the input is a connection lengths matrix.
    SPL = D;
    SPL(SPL == 0) = inf;
end

n=size(D,2);

if nargout > 1
    flag_find_paths = true;
    hops = double(D ~= 0);
    Pmat = 1:n;
    Pmat = Pmat(ones(n,1),:);
else
    flag_find_paths = false;
end

for k=1:n
    i2k_k2j = bsxfun(@plus, SPL(:,k), SPL(k,:));
    
    if flag_find_paths
        path = bsxfun(@gt, SPL, i2k_k2j);
        [i,j] = find(path);
        hops(path) = hops(i,k) + hops(k,j)';
        Pmat(path) = Pmat(i,k);
    end
    
    SPL = min(SPL, i2k_k2j);
end

SPL(eye(n)>0)=0;

if flag_find_paths
    hops(eye(n)>0)=0;
    Pmat(eye(n)>0)=0;
end
