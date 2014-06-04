function Z=module_degree_zscore(W,Ci)
%MODULE_DEGREE_ZSCORE       Within-module degree z-score
%
%   Z=module_degree_zscore(W,Ci);
%
%   The within-module degree z-score is a within-module version of degree
%   centrality.
%
%   Inputs:     W,      binary/weighted, directed/undirected connection matrix
%               Ci,     community affiliation vector
%
%   Output:     Z,      within-module degree z-score.
%
%   Note: The output for directed graphs is the "out-neighbor" z-score.
%
%   Reference: Guimera R, Amaral L. Nature (2005) 433:895-900.
%
%
%   Mika Rubinov, UNSW, 2008-2010


n=length(W);                        %number of vertices
Z=zeros(n,1);
for i=1:max(Ci)
    Koi=sum(W(Ci==i,Ci==i),2);
    Z(Ci==i)=(Koi-mean(Koi))./std(Koi);
end

Z(isnan(Z))=0;