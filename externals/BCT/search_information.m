function SI = search_information(adj,transform,has_memory)
% SEARCH_INFORMATION                    Search information
%
%   SI = search_information(adj,transform,has_memory)
%
%   Computes the amount of information (measured in bits) that a random
%   walker needs to follow the shortest path between a given pair of nodes.
%
%   Inputs:
%
%       adj,
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
%      	has_memory,
%           This flag defines whether or not the random walker "remembers"
%           its previous step, which has the effect of reducing the amount
%           of information needed to find the next state. If this flag is
%           not set, the walker has no memory by default.
%
%
%   Outputs:
%
%       SI,
%           pair-wise search information (matrix). Note that SI(i,j) may be
%           different from SI(j,i), hense, SI is not a symmetric matrix
%           even when adj is symmetric.
%
%
%   References: Rosvall et al. (2005) Phys Rev Lett 94, 028701
%               Goñi et al (2014) PNAS doi: 10.1073/pnas.131552911
%
%
%   Andrea Avena-Koenigsberger and Joaquin Goñi, IU Bloomington, 2014


%   Modification history
%   2014 - original
%   2016 - included SPL transform option and generalized for
%          symmetric/asymmetric networks


if ~exist('transform','var')
    transform = [];
end

if ~exist('has_memory','var')
    has_memory = false;
end

N = size(adj,1);

if sum(sum( triu(adj,1) + triu(adj,1)' - (adj))) < eps
    flag_triu = true;           % matrix is symmetric (undirected network)
else
    flag_triu = false;          % matrix is not symmetric (directed network)
end

T = diag(sum(adj,2))\adj;
[~,hops,Pmat] = distance_wei_floyd(adj,transform);

SI = zeros(N,N);
SI(eye(N)>0) = nan;

for i = 1:N
    for j = 1:N
        if (j > i && flag_triu) || (~flag_triu && i ~= j)
            path = retrieve_shortest_path(i,j,hops,Pmat);
            lp = length(path);
            if flag_triu
                if ~isempty(path)
                    pr_step_ff = nan(1,lp-1);
                    pr_step_bk = nan(1,lp-1);
                    if has_memory
                        pr_step_ff(1) = T(path(1),path(2));
                        pr_step_bk(lp-1) = T(path(lp),path(lp-1));
                        for z=2:lp-1
                            pr_step_ff(z) = T(path(z),path(z+1))/(1 - T(path(z-1),path(z)));
                            pr_step_bk(lp-z) = T(path(lp-z+1),path(lp-z))/(1 - T(path(lp-z+2),path(lp-z+1)));
                        end
                    else
                        for z=1:length(path)-1
                            pr_step_ff(z) = T(path(z),path(z+1));
                            pr_step_bk(z) = T(path(z+1),path(z));
                        end
                    end
                    prob_sp_ff = prod(pr_step_ff);
                    prob_sp_bk = prod(pr_step_bk);
                    SI(i,j) = -log2(prob_sp_ff);
                    SI(j,i) = -log2(prob_sp_bk);
                else
                    SI(i,j) = inf;
                    SI(j,i) = inf;
                end
            else
                if ~isempty(path)
                    pr_step_ff = nan(1,lp-1);
                    if has_memory
                        pr_step_ff(1) = T(path(1),path(2));
                        for z=2:lp-1
                            pr_step_ff(z) = T(path(z),path(z+1))/(1 - T(path(z-1),path(z)));
                        end
                    else
                        for z=1:length(path)-1
                            pr_step_ff(z) = T(path(z),path(z+1));
                        end
                    end
                    prob_sp_ff = prod(pr_step_ff);
                    SI(i,j) = -log2(prob_sp_ff);
                else
                    SI(i,j) = inf;
                end
            end
        end
    end
end
