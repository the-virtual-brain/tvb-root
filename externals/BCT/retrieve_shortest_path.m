function path = retrieve_shortest_path(s,t,hops,Pmat)
% RETRIEVE_SHORTEST_PATH        Retrieval of shortest path
%
%   This function finds the sequence of nodes that comprise the shortest
%   path between a given source and target node.
%
%   Inputs:
%       s,
%           Source node: i.e. node where the shortest path begins. 
%    	t,
%           Target node: i.e. node where the shortest path ends.
%       hops,
%           Number of edges in the path. This matrix may be obtained as the
%           second output argument of the function "distance_wei_floyd.m".
%       Pmat,
%           Pmat is a matrix whose elements {k,t} indicate the next node in
%           the shortest path between k and t. This matrix may be obtained
%           as the third output of the function "distance_wei_floyd.m"
%
%   Output:
%       path,
%           Nodes comprising the shortest path between nodes s and t.
%
%
%   Andrea Avena-Koenigsberger and Joaquin Goñi, IU, 2012

path_length = hops(s,t);
if path_length ~= 0
    path = nan(path_length+1,1);
    path(1) = s;
    for ind = 2:length(path)
        s = Pmat(s,t);
        path(ind) = s;
    end
else
    path = [];
end