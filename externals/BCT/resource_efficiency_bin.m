function [Eres,prob_SPL] = resource_efficiency_bin(adj,lambda,SPL,M)
% RESOURCE_EFFICIENCY_BIN       Resource efficiency and shortest-path probability
%
%   [Eres,prob_SPL] = resource_efficiency_bin(adj,lambda,SPL,M)
%
%   The resource efficiency between nodes i and j is inversly proportional
%   to the amount of resources (i.e. number of particles or messages)
%   required to ensure with probability 0 < lambda < 1 that at least one of
%   them will arrive at node j in exactly SPL steps, where SPL is the
%   length of the shortest-path between i and j.
%
%   The shortest-path probability between nodes i and j is the probability
%   that a single random walker starting at node i will arrive at node j by
%   following (one of) the shortest path(s).
%
%   Inputs:
%
%       adj,
%           Unweighted, undirected adjacency matrix
%
%       lambda,
%           Probability (0 < lambda < 1)
%          	set to NAN if computation of Eres is not desired
%
%       SPL,
%           Shortest-path length matrix (optional)
%
%     	M,
%           Transition probability matrix (optional)
%
%
%   Outputs:
%
%       Eres,
%           Resource efficiency matrix.
%
%       prob_SPL,
%           Shortest-path probability matrix
%
%   Notes:
%
%       Global measures for both Eres and prob_SPL are well defined and can
%       be computed as the average across the off-diagonal elements:
%           GEres = mean(Eres(~eye(N)>0));
%           Gprob_SPL = mean(rob_SPL(~eye(N)>0));
%
%
%   Reference: Goñi J, et al (2013) PLoS ONE
%
%
%   Joaquin Goñi, IU Bloomington, 2012


N = size(adj,1);
EYE = logical(eye(N,N));

flagResources = ~isnan(lambda);

if flagResources
    if lambda<=0 || lambda>=1
        error('p_req_values must be non-zero probabilities')
    end
    z = zeros(N,N);
end
if nargin<4
    SPL = distance_wei_floyd(adj);
end
if nargin<5
    M = diag(sum(adj,2))\adj;
end

Lvalues =  unique(SPL(:));
Lvalues = Lvalues(~(Lvalues==0));

prob_SPL = zeros(N,N);         % a priori zero probability of going through SPL among nodes

for indexSPL=1:length(Lvalues) %for each possible value of SPL for current component
    SPLvalue = Lvalues(indexSPL);
    [~, hcols] = find(SPL==SPLvalue);
    hvector = unique(hcols); clear hrows hcols
    entries = SPL==SPLvalue;
    
    if flagResources  % compute Eres
        [prob_aux,z_aux] = prob_first_particle_arrival(M,SPLvalue,hvector,lambda);
    else              % not compute Eres
        [prob_aux] = prob_first_particle_arrival(M,SPLvalue,hvector,[]);
    end
    
    prob_aux(~entries) = 0;
    prob_SPL = prob_SPL + prob_aux;
    
    if flagResources
        z_aux(~entries) = 0;
        z = z + z_aux;
    end
end

prob_SPL(EYE) = 0;

if flagResources
    z(prob_SPL==1) = 1;
    Eres = 1./z;
    Eres(EYE) = 0;
else
    Eres =  nan;
end

function [prob,resources] = prob_first_particle_arrival(M,L,hvector,lambda)

N = size(M,1);
prob = zeros(N,N);

if nargin<4
    hvector=1:N;
end

flagResources = ~isnan(lambda);

if flagResources
    if lambda<=0 || lambda>=1
        error('p_req_values must be non-zero probabilities')
    end
    resources = zeros(N,N);
end

for hindex=1:length(hvector)          %for each destination node h
    h = hvector(hindex);
    B_h = M;
    B_h(h,:) = 0; B_h(h,h) = 1;       % h becomes absorbant state.
    
    B_h_L = B_h^L;
    
    term = 1-B_h_L(:,h);
    
    prob(:,h)= 1-term;
    
    if flagResources
        resources(:,h) = repmat(log(1-lambda),N,1)./repmat(log(term),1);
    end
end
