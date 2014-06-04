function [D B]=distance_wei(G)
%DISTANCE_WEI       Distance matrix
%
%   D = distance_wei(L);
%   [D B] = distance_wei(L);
%
%   The distance matrix contains lengths of shortest paths between all
%   pairs of nodes. An entry (u,v) represents the length of shortest path 
%   from node u to node v. The average shortest path length is the 
%   characteristic path length of the network.
%
%   Input:      L,      Directed/undirected connection-length matrix.
%
%   Output:     D,      distance (shortest weighted path) matrix
%               B,      number of edges in shortest weighted path matrix
%
%   Notes:
%       The input matrix must be a connection-length matrix, typically
%   obtained via a mapping from weight to length. For instance, in a
%   weighted correlation network higher correlations are more naturally
%   interpreted as shorter distances and the input matrix should
%   consequently be some inverse of the connectivity matrix. 
%       The number of edges in shortest weighted paths may in general 
%   exceed the number of edges in shortest binary paths (i.e. shortest
%   paths computed on the binarized connectivity matrix), because shortest 
%   weighted paths have the minimal weighted distance, but not necessarily 
%   the minimal number of edges.
%       Lengths between disconnected nodes are set to Inf.
%       Lengths on the main diagonal are set to 0.
%
%   Algorithm: Dijkstra's algorithm.
%
%
%   Mika Rubinov, UNSW/U Cambridge, 2007-2012.
%   Rick Betzel and Andrea Avena, IU, 2012

%Modification history
%2007: original
%2009-08-04: min() function vectorized
%2012 added number of edges in shortest path as additional output

n=length(G);
D=zeros(n); D(~eye(n))=inf;                 %distance matrix
B=zeros(n);                                 %number of edges matrix

for u=1:n
    S=true(1,n);                            %distance permanence (true is temporary)
    G1=G;
    V=u;
    while 1
        S(V)=0;                             %distance u->V is now permanent
        G1(:,V)=0;                          %no in-edges as already shortest
        for v=V
            W=find(G1(v,:));                %neighbours of shortest nodes
            [d wi]=min([D(u,W);D(u,v)+G1(v,W)]);
            D(u,W)=d;                       %smallest of old/new path lengths
            ind=W(wi==2);                   %indices of lengthened paths
            B(u,ind)=B(u,v)+1;              %increment no. of edges in lengthened paths
        end

        minD=min(D(u,S));
        if isempty(minD)||isinf(minD),      %isempty: all nodes reached;
            break,                          %isinf: some nodes cannot be reached
        end;

        V=find(D(u,:)==minD);
    end
end