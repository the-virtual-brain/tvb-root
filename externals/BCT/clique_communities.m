function M = clique_communities(A, cq_thr)
% CLIQUE_COMMUNITIES     Overlapping community structure via clique percolation
%
%   M = clique_communities(A, cq_thr)
%
%   The optimal community structure is a subdivision of the network into
%   groups of nodes which have a high number of within-group connections
%   and a low number of between group connections.
%
%   This algorithm uncovers overlapping community structure in binary
%   undirected networks via the clique percolation method.
%
%   Inputs:
%       A,          Binary undirected connection matrix.
%
%      	cq_thr,     Clique size threshold (integer). Larger clique size
%                   thresholds potentially result in larger communities.
%
%   Output:     
%       M,          Overlapping community-affiliation matrix
%                   Binary matrix of size CxN [communities x nodes]
%
%   Algorithms:
%       Bron–Kerbosch algorithm for detection of maximal cliques.
%       Dulmage-Mendelsohn decomposition for detection of components
%                   (implemented in get_components.m)
%
%
%   Note: This algorithm can be slow and memory intensive in large
%   matrices. The algorithm requires the function get_components.m
%
%   Reference: Palla et al. (2005) Nature 435, 814-818.
%
%   Mika Rubinov, Janelia HHMI, 2017

if ~isequal(A, A.')
    error('A must be undirected.')
end
if ~isequal(size(A, 1), size(A, 2))
    error('A must be square.')
end
if ~issparse(A)
    A = sparse(A);
end
if ~islogical(A)
    A = logical(A);
end

n = length(A);                                  % number of nodes
A(1:n+1:end) = 0;                               % clear diagonal
MQ = maximal_cliques(A, n);                     % get maximal cliques
Cq = double(cell2mat(MQ)).';                    % convert to matrix
Cq = Cq(sum(Cq, 2) >= cq_thr, :);               % remove subthreshold cliques
Ov = Cq * Cq.';                                 % compute clique overlap
Ov_thr = (Ov >= cq_thr - 1);                    % keep percolating cliques

Cq_components = get_components(Ov_thr);         % find components 

m = max(Cq_components);                         % get number of components
M = zeros(m, n);                                % collect communities
for i = 1:m
    M(i, any( Cq(Cq_components==i, :), 1)) = 1;
end

end

function MQ = maximal_cliques(A, n)             % Bron-Kerbosch algorithm

MQ = cell(1, 1000*n);

R = false(n, 1);               %current
P = true(n, 1);                %prospective
X = false(n, 1);               %processed
q = 0;

BK(R, P, X);

    function BK(R, P, X)
        if ~any(P | X)
            q = q + 1;
            MQ{q} = R;
        else
            U_p = find(any([P X], 2));
            [~, idx] = max(A(:,U_p).' * double(P));
            u_p = U_p(idx);
            
            U = find(all([P ~A(:,u_p)], 2)).';
            for u = U
                Nu = A(:,u);
                P(u) = 0;
                Rnew = R; Rnew(u) = 1;
                Pnew = all([P Nu],2);
                Xnew = all([X Nu],2);
                BK(Rnew, Pnew, Xnew)
                X(u) = 1;
            end
        end
    end

MQ=MQ(1:q);

end
