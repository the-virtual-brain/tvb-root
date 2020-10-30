function [FCpre,pred_data,Fcorr] = generate_fc(SC,beta,ED,pred_var,model,FC)
%   GENERATE_FC     Generation of synthetic functional connectivity matrices
%
%   [FCpre,pred_data,Fcorr] = generate_fc(SC,beta,ED,{'SPLwei_log','SIwei_log'},FC)
%   [FCpre,pred_data] = generate_fc(SC,beta,[],{'SPLwei_log','SIwei_log'})
%
%   Uses a vector beta of regression coefficients from the model
%   FC = pred_var*beta to predict FC. pred_var are structural-based network
%   measures derived from the structural connectivity network.
%
%   Inputs:
%
%       SC,
%           Weighted/unweighted undirected NxN Structural Connectivity matrix.
%
%       beta,
%           Regression coefficients (vector). These may be obtained as an
%           output parameter from function predict_fc.m
%
%       ED,
%           Euclidean distance matrix or upper triangular vector of the
%           matrix (optional)
%
%       pred_var,
%           Set of M predictors. These can be given as an KxM array where
%           K = ((N*(N-1))/2) and M is the number of predictors.
%           Alternatively, pred_var can be a cell with the names of network
%           measures to be used as predictors. Accepted network measure
%           names are:
%               SPLbin        - Shortest-path length (binary)
%               SPLwei_inv    - Shortest-path length computed with an inv transform
%             	SPLwei_log    - Shortest-path length computed with a log transform
%             	SPLdist       - Shortest-path length computed with no transform
%              	SIbin         - Search Information of binary shortest-paths
%             	SIwei_inv     - Search Information of shortest-paths computed with an inv transform
%           	SIwei_log     - Search Information of shortest-paths computed with a log transform
%              	SIdist        - Search Information of shortest-paths computed with no transform
%              	T             - Path Transitivity
%              	deltaMFPT     - Column-wise z-scored mean first passage time
%               neighOverlap  - Neighborhood Overlap
%              	MI            - Matching Index
%
%           Predictors must be specified in the order that matches the
%           given beta values.
%
%      	model,
%           Specifies the order of the regression model used within
%           matlab's function regstats.m. 'model' can be any option
%           accepted by matlab's regstats.m function (e.g.'linear',
%           'interaction' 'quadratic', etc). If no model is specified,
%           'linear' is the default.
%
%    	FC,
%           Functional connections. FC can be a NxN symmetric matrix or a
%           ((N*(N-1))/2) x 1 vector containing the upper triangular
%           elements of the square FC matrix (excluding diagonal elements).
%           This argument is optional and only used to compute the
%           correlation between the predicted FC and empirical FC.
%
%
%   Outputs:
%
%       FCpre,
%           Predicted NxN Functional Connectivity matrix
%
%      	pred_data,
%           KxM array of predictors.
%
%       FCcorr,
%           Pearson Correlation between FCpred and FC
%
%
%   Reference: Goñi et al. (2014) PNAS 111: 833–838
%
%
%   Andrea Avena-Koenigsberger, Joaquin Goñi and Olaf Sporns; IU Bloomington, 2016


[b1,b2] = size(beta);
if b1 == 1 && b2 >= b1
    beta = beta';  % beta must be a column vector
elseif b1 > 1 && b2 > 1
    error('beta must be a vector of scalar regression coefficients')
end

pred_names = {'SPLbin','SPLwei_inv','SPLwei_log','SPLdist','SIbin',...
    'SIwei_inv','SIwei_log','SIdist','T','deltaMFPT','neighOverlap','MI'};

% select model
if ~exist('model','var') || isempty(model)
    model = 'linear';
end

N = size(SC,1);
indx = find(triu(ones(N),1));

if ~exist('pred_var','var') && ~isempty(ED)
    pred_var = {'ED','SPLwei_log','SI','T'};
    flag_var_names = true;
    flag_ED = true;
elseif ~exist('pred_var','var') && isempty(ED)
    pred_var = {'SPLwei_log','SI','T'};
    flag_var_names = true;
elseif exist('pred_var','var') && ~isnumeric(pred_var) && ~isempty(ED)
    flag_var_names = true;
    flag_ED = true;
elseif exist('pred_var','var') && ~isnumeric(pred_var) && isempty(ED)
    flag_var_names = true;
    flag_ED = false;
elseif exist('pred_var','var') && isnumeric(pred_var) && ~isempty(ED)
    flag_var_names = false;
    flag_ED = true;
elseif exist('pred_var','var') && isnumeric(pred_var) && isempty(ED)
    flag_var_names = false;
    flag_ED = false;
else
    err_str = '"pred_var" must be an KxM array of M predictors, or any of the following graph-measure names:';
    s1 = sprintf('SPLbin - Shortest-path length (binary) \n');
    s2 = sprintf('SPLwei_inv - Shortest-path length computed with an inv transform \n');
    s3 = sprintf('SPLwei_log - Shortest-path length computed with a log transform \n');
    s4 = sprintf('SPLdist - Shortest-path length computed with no transform \n');
    s5 = sprintf('SIbin - Search Information of binary shortest-paths \n');
    s6 = sprintf('SIwei_inv - Search Information of shortest-paths computed with an inv transform \n');
    s7 = sprintf('SIwei_log - Search Information of shortest-paths computed with a log transform \n');
    s8 = sprintf('SIdist - Search Information of shortest-paths computed with no transform \n');
    s9 = sprintf('T - Path Transitivity \n');
    s10 = sprintf('deltaMFPT - Column-wise z-scored mean first passage time \n');
    s11 = sprintf('neighOverlap - Neighborhood Overlap \n');
    s12 = sprintf('MI - Matching Index \n');
    error('%s \n %s %s %s %s %s %s %s %s %s %s %s %s',err_str,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12);
end

if flag_ED
    [n1,n2] = size(ED);
    if n1 == n2 && n1 == N
        % square ED matrix
        pred_data = ED(indx);
    elseif n1 == length(indx) && n2 == 1
        % ED is already an upper-triangle vector
        pred_data = ED;
    else
        error('ED must be square matrix or a vector containing the upper triangle of the square ED matrix \n')
    end
else
    pred_data = [];
end


if flag_var_names
    fprintf('\n----------------------');
    fprintf('\n Selected predictors: \n');
    ind2start = size(pred_data,2);
    pred_data = [pred_data,zeros(length(indx),length(pred_var))];
    
    for v = 1:length(pred_var)
        var_ind = find(strcmp(pred_var{v},pred_names));
        switch var_ind
            
            case 1   %SPLbin
                fprintf('Shortest-path length (binary) \n\n');
                data = distance_wei_floyd(double(SC>0));
            case 2   %SPLwei_inv
                fprintf('Shortest-path length computed with an inv transform \n');
                data = distance_wei_floyd(SC,'inv');
            case 3   %SPLwei_log
                fprintf('Shortest-path length computed with a log transform \n');
                data = distance_wei_floyd(SC,'log');
            case 4   %SPLdist
                fprintf('Shortest-path length computed with no transform \n');
                data = distance_wei_floyd(SC);
            case 5   %SIbin
                fprintf('Search Information of binary shortest-paths \n');
                data = search_information(double(SC>0));
                data = data + data';
            case 6   %SIwei_inv
                fprintf('Search Information of shortest-paths computed with an inv transform \n');
                data = search_information(SC,'inv');
                data = data + data';
            case 7   %SIwei_log
                fprintf('Search Information of shortest-paths computed with a log transform \n');
                data = search_information(SC,'log');
                data = data + data';
            case 8   %SIdist
                fprintf('Search Information of shortest-paths computed with no transform \n');
                data = search_information(SC);
                data = data + data';
            case 9   %T
                fprintf('Path Transitivity \n');
                data = path_transitivity(double(SC>0));
            case 10  %deltaMFPT
                fprintf('Column-wise z-scored mean first passage time \n');
                mfpt = mean_first_passage_time(SC);
                deltamfpt = zscore(mfpt,[],1);
                data = deltamfpt+deltamfpt';
            case 11  %neighOverlap
                fprintf('Neighborhood Overlap \n');
                data = double(SC>0) * double(SC>0)';
            case 12  %MI
                fprintf('Matching Index \n');
                data = matching_ind(SC);
            otherwise
                error('This is not an accepted predictor. See list of available predictors \n')
        end
        pred_data(:,ind2start+v) = data(indx);
    end
else
    if size(pred_var,1) == length(indx)
        pred_data = [pred_data,pred_var];
    else
        error('Custom predictors must be provided as KxM array of M predictors \n');
    end
end

pred_data = x2fx(pred_data,model);

if size(pred_data,2) == size(beta,1)
    Y = pred_data*beta;
    FCpre = zeros(N);
    FCpre(indx) = Y;
    FCpre = FCpre+FCpre';
end

if nargin == 6 && ~isempty(FC)
    flag_nan_corr = false;
    [n1,n2] = size(FC);
    if n1 == n2 && n1 == N
        % square FC matrix
        FCemp = FC(indx);
    elseif n1 == length(indx) && n2 == 1
        % FC is already an upper-triangle vector
        FCemp = FC;
    else
        warning('FC must be square matrix or a vector containing the upper triangle (no diagonal elements) of the square FC matrix \n')
        flag_nan_corr = true;
    end
    if ~flag_nan_corr
        Fcorr = corr(Y,FCemp);
    else
        Fcorr = nan;
    end
end
