function T=path_transitivity(W,transform)
% PATH_TRANSITIVITY             Transitivity based on shortest paths
%
%   T = path_transitivity(W,transform)
%
%   This function computes the density of local detours (triangles) that
%   are available along the shortest-paths between all pairs of nodes.
%
%   Inputs:
%
%       W,
%           unweighted/weighted undirected connection *weight* OR *length*
%           matrix.
%
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
%   Output:
%
%       T,
%           matrix of pairwise path transitivity.
%
%
%   Olaf Sporns, Andrea Avena-Koenigsberger and Joaquin Goñi, IU Bloomington, 2014
%
%   References: Goñi et al (2014) PNAS doi: 10.1073/pnas.131552911
%

if ~exist('transform','var')
    transform = [];
end

n=length(W);
m=zeros(n,n);
T=zeros(n,n);

for i=1:n-1
    for j=i+1:n
        x=0;
        y=0;
        z=0;
        for k=1:n
            if W(i,k)~=0 && W(j,k)~=0 && k~=i && k~=j
                x=x+W(i,k)+W(j,k);
            end
            if k~=j
                y=y+W(i,k);
            end
            if k~=i
                z=z+W(j,k);
            end
        end
        m(i,j)=x/(y+z);
    end
end
m=m+m';

[~,hops,Pmat] = distance_wei_floyd(W,transform);

% --- path transitivity ---%%
for i=1:n-1
    for j=i+1:n
        x=0;
        path = retrieve_shortest_path(i,j,hops,Pmat);
        K=length(path);
        
        for t=1:K-1
            for l=t+1:K
                x=x+m(path(t),path(l));
            end
        end
        T(i,j)=2*x/(K*(K-1));
    end
end
T=T+T';
