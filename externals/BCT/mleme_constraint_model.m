function [W0, E0, P0, Delt0] = mleme_constraint_model(samp, W, M, Lo, Li, Lm, opts)
%MLEME_CONSTRAINT_MODEL     Unbiased sampling of networks with soft constraints
%
%   W0 = mleme_constraint_model(samp, W);
%   W0 = mleme_constraint_model(samp, W, M);
%   W0 = mleme_constraint_model(samp, W, M, Lo, Li, Lm);
%   [W0, E0, P0, Delt0] = mleme_constraint_model(samp, W, M, Lo, Li, Lm, opts);
%
%   This function returns an ensemble of unbiasedly sampled networks with
%   weighted node-strength and module-weight constraints. These constraints
%   are soft in that they are satisfied on average for the full network
%   ensemble but not, in general, for each individual network.
%
%   Inputs (for a network with n nodes, m modules and c constraints):
%
%       samp,   Number of networks to sample.
%
%       W,      (length n) square directed and weighted connectivity
%               matrix. All weights must be nonnegative integers. Note that
%               real-valued weights may be converted to integers with
%               arbitrary precision through rescaling and rounding, e.g.
%               W_int = round(10^precision * W_real).
%
%       M,      (length n) module affiliation vector. This vector is often
%               obtained as the output of a community detection algorithm.
%               The vector must contain nonnegative integers, with zeros
%               specifying nodes which are not part of any community. This
%               input may be left empty if there are no module constraints.
%
%       Lo,     (length n) out-strength constraint logical vector. This
%               vector specifies out-strength constraints for each node.
%               Alternatively, it is possible to specify 1 to constrain all
%               out-strengths or 0 for no constraints. Empty or no input
%               results in default behavour (no constraints).
%
%       Lo,     (length n) in-strength constraint logical vector. This
%               vector specifies in-strength constraints for each node.
%               Alternatively, it is possible to specify 1 to constrain all
%               in-strengths or 0 for no constraints. Empty or no input
%               results in default behavour (no constraints).
%
%       Lm,     (length m) module-weight constraint logical matrix. This
%               matrix specifies module-weight constraints for all pairs of
%               modules. Alternatively, it is possible to specify
%               2 to constrain all inter-module and intra-module weights,
%               1 to constrain all intra-module weights, or 0  for no
%               constraints. Empty or no input results in default behavour
%               (no constraints).
%
%       opts,   optional argument: pass optimization and display options with optimset.
%               Default: optimset('MaxFunEvals', 1e6*c, 'MaxIter', 1e6, 'Display', 'iter');
%
%
%   Outputs:
%       W0,     an ensemble of sampled networks with constraints.
%
%       E0,     expected weights matrix.
%
%       P0,     probability matrix.
%
%       Delt0,  algorithm convergence error.
%
%
%   Algorithm:
%               Maximum-likelihood estimation of network probability
%               distribution by numerical solution of systems of nonlinear
%               equations, and sampling of individual networks directly
%               from this distribution.
%
%
%   Notes:
%               Empirical connection weights are
%               not preserved. Constraint errors are guaranteed to vanish
%               in the limit of the full network ensemble.
%
%
%   Examples:
%               % get community structure of a weighted network W
%               M = community_louvain(W, 2);
%
%               % specify node and module constraints
%               n = length(W);              % number of nodes
%               m = max(M);                 % number of modules
%               Lo = true(n, 1);            % out-strength constraints
%               Li = true(n, 1);            % in-strength constraints
%               Lm = eye(m);                % module-weight constraints
%
%               % sample networks with the above constraints
%               [W0, E0, P0, Delt0] = mleme_constraint_model(samp, W, M, Lo, Li, Lm);
%
%               % equivalent formulation
%               [W0, E0, P0, Delt0] = mleme_constraint_model(samp, W, M, 1, 1, 1);
%
%               % alternative: sample networks with average weight constraints only
%               [W0, E0, P0, Delt0] = mleme_constraint_model(samp, W);
%
%
%   References: Squartini and Garlaschelli (2011) New J Phys 13:083001
%               Rubinov (2016) Nat Commun 7:13812
%
%
%   2016, Mika Rubinov, Janelia HHMI

%   Modification History
%   Dec 2016: Original.

n = length(W);                  % number of nodes

if ~exist('M', 'var') || isempty(M)
    if exist('Lm', 'var') && any(Lm)
        error('Need module affiliation vector for module constraints')
    else
        M = zeros(n, 1);
    end
end

m = max(M);                     % number of modules

if ~isequal(W, int64(W)) || min(W(:))<0
    error('W must only contain nonnegative integers.')
end
if ~isequal(M, int64(M)) || min(M(:))<0
    error('M must only contain nonnegative integers.')
end

% process node constraints
if ~exist('Lo','var') || isempty(Lo) || isequal(Lo,0)
    Lo = false(n, 1);
elseif isequal(Lo, 1)
    Lo = true(n, 1);
end
if ~exist('Li','var')
    Li = Lo;
elseif isempty(Li) || isequal(Li, 0)
    Li = false(n, 1);
elseif isequal(Li, 1)
    Li = true(n, 1);
end

% process module constraints
if ~exist('Lm','var') || isempty(Lm) || isequal(Lm,0)
    Lm = false(m);
elseif isequal(Lm, 2)
    Lm = true(m);
elseif isequal(Lm, 1)
    Lm = diag(true(m, 1));
end
if any(~M)
    m = m + 1;
    M(~M) = m;
    Lm(m, m) = 0;     % add a new row and column for nodes without modules
end

Lo = logical(Lo(:));
Li = logical(Li(:));
Lm = logical(Lm(:));
ao = numel(Lo);
ai = numel(Li);
am = numel(Lm);
uo = nnz(Lo);
ui = nnz(Li);
um = nnz(Lm);
Mij = bsxfun(@plus, M, (M.'-1)*m);

f_ex = @(V) system_equations(V, Mij, Lo, Li, Lm, ao, ai, am, uo, ui, um);
f_cx = @(W) system_constraints(W, M, Lo, Li, Lm, uo, ui, um);

C = f_cx(W);
c = 1 + uo + ui + um;
if ~exist('V','var')
    V = mean2(W)/(1+mean2(W))*ones(c,1);
end

assert(c == numel(C));
assert(c == numel(V));

if ~exist('opts', 'var') || isempty(opts)
    opts = optimset('MaxFunEvals', 1e6*c, 'MaxIter', 1e6, 'Display', 'iter');
end

V0 = fsolve(@(V) C - f_cx(f_ex(V)), V, opts);

[E0, P0] = f_ex(V0);
Delt0 = C - f_cx(f_ex(V0));

W0 = sample_networks(P0, samp);

end


function CellW0 = sample_networks(P0, samp)

if ~exist('samp', 'var')
    samp = 1;
end

n = length(P0);

CellW0 = cell(samp, 1);
for i = 1:samp
    W0 = zeros(n);
    L0 = ~eye(n);
    l0 = nnz(L0);
    while l0
        L0(L0) = P0(L0) > rand(l0,1);
        W0(L0) = W0(L0) + 1;
        l0 = nnz(L0);
    end
    CellW0{i} = W0;
end

end


function [W, P] = system_equations(V, Mij, Lo, Li, Lm, ao, ai, am, uo, ui, um)

X = ones(ao, 1);
Y = ones(ai, 1);
Z = ones(am, 1);

if uo
    offset = 1;
    X(Lo) = V(offset + (1:uo));
end
if ui
    offset = 1 + uo;
    Y(Li) = V(offset + (1:ui));
end
if um
    offset = 1 + uo + ui;
    Z(Lm) = V(offset + (1:um));
end

P = V(1) .* (X * Y.') .* Z(Mij);            % V(1) is the total weight
P(P>1) = 1 - eps;

W = P ./ (1 - P);
W(1:length(W)+1:end) = 0;

end


function C = system_constraints(W, M, Lo, Li, Lm, uo, ui, um)

if nargin == 0
    C = @block_density;
    return;
end

if uo
    So = sum(W(Lo,:), 2);
else
    So = [];
end
if ui
    Si = sum(W(:,Li), 1).';
else
    Si = [];
end
if um
    Wm = block_density(W, M, Lm);
else
    Wm = [];
end

C = [sum(sum(W)); So; Si; Wm];

end


function Wm = block_density(W, M, Lwm)

m = max(M);

Wm = zeros(m*m, 1);
for u = 1:m
    for v = 1:m
        Wm(u + (v-1)*m) = sum(sum(W(M==u, M==v)));
    end
end

Wm = Wm(Lwm);

end
