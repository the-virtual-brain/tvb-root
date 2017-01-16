function [VIn, MIn] = partition_distance(Cx, Cy)
%PARTITION_DISTANCE     Distance or similarity between community partitions
%
%   This function quantifies information-theoretic distance (normalized
%   variation of information) or similarity (normalized mutual information)
%   between community partitions.
%
%   VIn        = partition_distance(Cx);
%   VIn        = partition_distance(Cx, Cy);
%   [VIn, MIn] = partition_distance(Cx, Cy);
%
%   Inputs:
%       Cx,
%           Community partition vector or matrix of n rows and p columns,
%           n is the number of network nodes, and p is the number of input
%           community partitions (in the case of vector input p=1).
%
%       Cy (optional argument),
%           Community partition vector or matrix of n rows and q columns. n
%           is the number of nodes (must be equal to the number of nodes in
%           Cq) and q is the number of input community partitions (may be
%           different to the number of nodes in Cq). This argument may be
%           omitted, in which case, the partition distance is computed
%           between all pairwise partitions of Cx.
%
%   Outputs:
%       VIn,
%           Normalized variation of information ([p, q] matrix)
%
%       MIn,
%           Normalized mutual information ([p, q] matrix)
%
%   Notes:
%       Mathematical definitions.
%
%           VIn = [H(X) + H(Y) - 2MI(X, Y)]/log(n)
%           MIn = 2MI(X, Y) / [H(X) + H(Y)]
%
%           where H is the entropy and MI is the mutual information
%
%
%   Reference: Meila M (2007) J Multivar Anal 98, 873-895.
%
%
%   2011-2017, Mika Rubinov, UNSW, Janelia HHMI

%   Modification History:
%   Mar 2011: Original
%   Jan 2017: Added computation between input matrices.

s = (nargin==1);
if s
    Cy = Cx;
    d = 10.^ceil(log10(double(1 + max( Cx(:)) )));
else
    d = 10.^ceil(log10(double(1 + max([Cx(:);Cy(:)]) )));
end

if ~isequal([Cx(:);Cy(:)], int64([Cx(:);Cy(:)])) || min([Cx(:);Cy(:)])<=0
    error('Input partitions must contain only positive integers.')
end

[n, p] = size(Cx);
HX = zeros(p, 1);
for i = 1:p
    Px = nonzeros(accumarray(Cx(:, i), 1)) / n;                     % P(x)
    HX(i) = - sum(Px .* log(Px));                                   % H(x)
end

if s
    q = p;
    HY = HX;
else
    [n_, q] = size(Cy);
    assert(n == n_);
    HY = zeros(q, 1);
    for j = 1:q
        Py = nonzeros(accumarray(Cy(:, j), 1)) / n;                 % P(y)
        HY(j) = - sum(Py .* log(Py));                               % H(y)
    end
end

VIn = zeros(p, q);
MIn = zeros(p, q);
for i = 1:p
    j_idx = (s * (i - 1) + 1):q;
    for j = j_idx
        Pxy = nonzeros(accumarray(d*Cx(:, i) + Cy(:, j), 1)) / n; 	% P(x,y)
        Hxy = -sum(Pxy .* log(Pxy));                                % H(x,y)
        VIn(i, j) = (2 * Hxy - HX(i) - HY(j)) / log(n);             % VIn
        MIn(i, j) = 2 * (HX(i) + HY(j) - Hxy) / (HX(i) + HY(j));    % MIn
    end
    if s
        VIn(j_idx, i) = VIn(i, j_idx);
        MIn(j_idx, i) = MIn(i, j_idx);
    end
end
