function y = np2m(x)
% Convert NumPy array to MATLAB array

% kudos to http://fr.mathworks.com/matlabcentral/answers/157347-convert-python-numpy-array-to-double#answer_157736

%% setup table of conversions from Python array type codes to MATLAB classes
% is persistent faster than setting up the table every time?
persistent t2m % type code to matlab class

if isempty(t2m)
    t2m.c = @char;
    t2m.b = @int8;
    t2m.B = @uint8;
    t2m.h = @int16;
    t2m.H = @int16;
    t2m.i = @int32;
    t2m.I = @uint32;
    t2m.l = @int64;
    t2m.L = @uint64;
    t2m.f = @single;
    t2m.d = @double;
end

%% handle type of array
type_code = char(x.dtype.char);
matlab_class = t2m.(type_code);

%% convert data
y = matlab_class(py.array.array(type_code, py.numpy.nditer(x)));

%% handle shape if required
ndim = x.ndim * 1;
if ndim > 1
    shape = int32(zeros([1 ndim]));
    for i=1:ndim
        shape(i) = x.shape{ndim+1-i}*1;
    end
    y = reshape(y, shape);
end

%% benchmarks
% Running for various sizes, we can see it's not zero-copy, so to be used
% with this in mind.

%{
[np2m] 0.08 MB, 0.04 ms / iter / KB
[np2m] 0.76 MB, 0.03 ms / iter / KB
[np2m] 7.63 MB, 0.03 ms / iter / KB
%}

if 0
    %%
    n = 100000;
    m = 100;
    x = py.numpy.random.randn(n).reshape([10, -1]);
    tic
    for i=1:m
        y = np2m(x);
    end
    fprintf('[np2m] %0.2f MB, %.2f ms / iter / KB\n', ...
        n*8/1024/1024, toc*1000/m/(n*8/1024));
end