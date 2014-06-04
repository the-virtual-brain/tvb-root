function W = weight_conversion(W, wcm)
% WEIGHT_CONVERSION    Conversion of weights in input matrix
%
%   W_bin = weight_conversion(W, 'binarize');
%   W_nrm = weight_conversion(W, 'normalize');
%   L = weight_conversion(W, 'lengths');
%
%   This function may either binarize an input weighted connection matrix,
%   normalize an input weighted connection matrix or convert an input
%   weighted connection matrix to a weighted connection-length matrix.
%
%       Binarization converts all present connection weights to 1.
%
%       Normalization scales all weight magnitudes to the range [0,1] and
%   should be done prior to computing some weighted measures, such as the
%   weighted clustering coefficient.
%
%       Conversion of connection weights to connection lengths is needed
%   prior to computation of weighted distance-based measures, such as
%   distance and betweenness centrality. In a weighted connection network,
%   higher weights are naturally interpreted as shorter lengths. The
%   connection-lengths matrix here is defined as the inverse of the
%   connection-weights matrix. 
%
%   Inputs: W           weighted connectivity matrix
%           wcm         weight-conversion command - possible values:
%                           'binarize'      binarize weights
%                           'normalize'     normalize weights
%                           'lengths'       convert weights to lengths
%
%   Output: W_          connectivity matrix with converted weights
%
%
%   Mika Rubinov, U Cambridge, 2012

%   Modification History:
%   Sep 2012: Original

switch wcm
    case 'binarize'
        W=double(W~=0);         %binarize
    case 'normalize'
        W=W./max(abs(W(:)));    %scale by maximal weight
    case 'lengths'
        E=find(W); 
        W(E)=1./W(E);           %invert weights
    otherwise
        error('Unknown weight-conversion command.')
end
