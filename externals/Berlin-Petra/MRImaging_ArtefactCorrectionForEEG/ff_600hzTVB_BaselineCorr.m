function [out,base]=ff_600hzTVB_baselinecorr(in,basetime)
 
    base=mean(in(:,basetime),2);
    base_matrix=repmat(base,1,size(in,2));
    out=in-base_matrix;