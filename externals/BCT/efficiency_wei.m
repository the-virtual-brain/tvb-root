function E=efficiency_wei(Gw,local)
%EFFICIENCY_WEI     Global efficiency, local efficiency.
%
%   Eglob = efficiency_wei(W);
%   Eloc = efficiency_wei(W,1);
%
%   The global efficiency is the average of inverse shortest path length, 
%   and is inversely related to the characteristic path length.
%
%   The local efficiency is the global efficiency computed on the
%   neighborhood of the node, and is related to the clustering coefficient.
%
%   Inputs:     W,              undirected weighted connection matrix
%                               (all weights in W must be between 0 and 1)
%               local,          optional argument
%                               (local=1 computes local efficiency)
%
%   Output:     Eglob,          global efficiency (scalar)
%               Eloc,           local efficiency (vector)
%
%   Notes:
%       The  efficiency is computed using an auxiliary connection-length
%   matrix L, defined as L_ij = 1/W_ij for all nonzero L_ij; This has an
%   intuitive interpretation, as higher connection weights intuitively
%   correspond to shorter lengths.
%       The weighted local efficiency broadly parallels the weighted
%   clustering coefficient of Onnela et al. (2005) and distinguishes the
%   influence of different paths based on connection weights of the
%   corresponding neighbors to the node in question. In other words, a path
%   between two neighbors with strong connections to the node in question
%   contributes more to the local efficiency than a path between two weakly
%   connected neighbors. Note that this weighted variant of the local
%   efficiency is hence not a strict generalization of the binary variant.
%
%   Algorithm:  Dijkstra's algorithm
%
%   References: Latora and Marchiori (2001) Phys Rev Lett 87:198701.
%               Onnela et al. (2005) Phys Rev E 71:065103
%               Rubinov M, Sporns O (2010) NeuroImage 52:1059-69
%
%
%   Mika Rubinov, U Cambridge, 2011-2012

%Modification history
%2011-09-17: original (based on efficiency.m and distance_wei.m)

if ~exist('local','var')
    local=0;
end

N=length(Gw);                               %number of nodes
Gl = Gw;
ind = Gl~=0;
Gl(ind) = 1./Gl(ind);                       %connection-length matrix

if local                                    %local efficiency
    E=zeros(N,1);                           %local efficiency

    for u=1:N
        V=find(Gw(u,:));                    %neighbors
        k=length(V);                        %degree
        if k>=2;                            %degree must be at least two
            e=( distance_inv_wei(Gl(V,V)) .* (Gw(V,u)*Gw(u,V)) ).^(1/3) ;
            E(u)=sum(e(:))./(k^2-k);        %local efficiency
        end
    end
else
    e=distance_inv_wei(Gl);
    E=sum(e(:))./(N^2-N);                   %global efficiency
end


function D=distance_inv_wei(G)

n=length(G);
D=zeros(n); D(~eye(n))=inf;                 %distance matrix

for u=1:n
    S=true(1,n);                            %distance permanence (true is temporary)
    G1=G;
    V=u;
    while 1
        S(V)=0;                             %distance u->V is now permanent
        G1(:,V)=0;                          %no in-edges as already shortest
        for v=V
            W=find(G1(v,:));                %neighbours of shortest nodes
            D(u,W)=min([D(u,W);D(u,v)+G1(v,W)]); %smallest of old/new path lengths
        end

        minD=min(D(u,S));
        if isempty(minD)||isinf(minD),      %isempty: all nodes reached;
            break,                          %isinf: some nodes cannot be reached
        end;

        V=find(D(u,:)==minD);
    end
end

D=1./D;                                     %invert distance
D(1:n+1:end)=0;