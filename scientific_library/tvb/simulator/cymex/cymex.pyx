"""

This Cython module forms the basis of a MATLAB interface to TVB.

It is necessary to redirect MATLAB's libgfortran link as found in 
$MATLAB/sys/os/glnxa64/libgfortran.so.3 to the one to which your NumPy's
extension module was linked, if it's a newer version than MATLAB's.
You might also try starting MATLAB like so

    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgfortran.so.3.0.0 matlab

Before compiling, you may need to mex -setup, and add to the end of the 
resulting mexopts.sh

    LDFLAGS="$LDFLAGS -Xlinker -export-dynamic"

Finally, to actually compile, 

    cython cymex.pyx
    mex cymex.c -I/usr/include/python2.7 -lpython2.7 -ldl

If you are using Octave, the second step is 

    mkoctfile --mex cymex.c -I/usr/include/python2.7 -lpython2.7 -ldl

Linear algebra (via numpy.linalg) may be tricky: routines like SVD call 
linked libraries like lapack; because NumPy might be built against a 
different underlying lapack implementation that MATLAB's, calls might
fail with MKL errors. This is currently under investigation.


This file is MIT licensed. 

.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

cdef extern from *:

    # some const defintions are required
    ctypedef char*     const_char_ptr        "const char*"
    ctypedef mxArray*  const_mxArray_ptr     "const mxArray*"
    ctypedef mxArray** const_mxArray_ptr_ptr "const mxArray**"

    # module initialization function
    void initcymex()

cdef extern from "Python.h":

    # http://docs.python.org/2/c-api/init.html
    void Py_Initialize()
    int Py_IsInitialized()

cdef extern from "dlfcn.h":
    int RTLD_LAZY, RTLD_GLOBAL
    void *dlopen(const_char_ptr fname, int flag)

cdef extern from "mex.h":

    # MATLAB's N-dimensional array type (opaque)
    ctypedef struct mxArray:
        pass

    # functions from MATLAB's documented extern API
    int mexPrintf(char *msg, ...)
    int mxGetN(mxArray*pm)
    void* mxMalloc(int)
    int mxGetString(mxArray*pm, char*str, int strlen)
    int mexCallMATLAB(int nlhs, mxArray **plhs,
                      int rlhs, mxArray **prhs, const_char_ptr cmd)

    # may call longjmp; is compatible with p/cython ?
    void mexErrMsgTxt(const_char_ptr msg) 
    void mexErrMsgIdAndTxt(const_char_ptr id, const_char_ptr txt, ...)

cdef extern from *:
    #cdef extern from "matrix.h":

    mxArray *mxCreateDoubleMatrix(int m, int n, int flag)
    double *mxGetPr(const_mxArray_ptr pm)

    # http://www.mathworks.com/help/matlab/apiref/mxclassid.html
    ctypedef enum mxClassID:

        mxUNKNOWN_CLASS
        mxCELL_CLASS
        mxSTRUCT_CLASS
        mxLOGICAL_CLASS
        mxCHAR_CLASS
        mxVOID_CLASS
        mxFUNCTION_CLASS

        # mxClassID Value   MATLAB Type     MEX Type    C Type                  numpy.dtype

        mxDOUBLE_CLASS  #   double          double      double                  float64
        mxSINGLE_CLASS  #   single          float       float                   float32
        mxINT8_CLASS    #   int8            int8_T      char, byte              byte, int8
        mxUINT8_CLASS   #   uint8           uint8_T     unsigned char, byte     ubyte, uint8
        mxINT16_CLASS   #   int16           int16_T     short                   etc.
        mxUINT16_CLASS  #   uint16          uint16_T    unsigned short
        mxINT32_CLASS   #   int32           int32_T     int
        mxUINT32_CLASS  #   uint32          uint32_T    unsigned int
        mxINT64_CLASS   #   int64           int64_T     long long
        mxUINT64_CLASS  #   uint64          uint64_T    unsigned long long

    mxClassID mxGetClassID(const_mxArray_ptr pm)

# redirect Python output to MATLAB
class mexPrinter(object):
    def write(self, msg):
        mexPrintf(msg)

import sys
sys.stdout = mexPrinter()
sys.stderr = mexPrinter()

cdef mstr(mxArray* arr):
    cdef:
        int buflen, status
        char *buf
    buflen = mxGetN(arr)*sizeof(char)+1
    buf = <char*> mxMalloc(buflen)
    status = mxGetString(arr, buf, buflen)
    return buf[:buflen-1] if status == 0 else ''

cdef mlsin(int n):
    cdef mxArray *l, *r
    l = mxCreateDoubleMatrix(1, 1, 0)
    r = mxCreateDoubleMatrix(1, 1, 0)
    cdef double *dp = <double*> mxGetPr(r)
    for i in range(n):
        dp[0] = i*1e-10
        mexCallMATLAB(1, &l, 1, &r, "sin")
    return (<double*>mxGetPr(l))[0]

class linalg_bypass(object):
    """
    A monkey patch to numpy.linalg
    """
    def __init__(self):
        """
        Install the monkey patch.

        """


        sys.modules['numpy.linalg']

cdef public void mexFunction(int nlhs, mxArray* lhs[],
                             int nrhs, const_mxArray_ptr_ptr rhs):

    # 1e6 calls from MATLAB
    #   empty       -> 2.34 s
    #   w/ niceties -> 3.39 s

    if not Py_IsInitialized():
        dlopen("libgfortran.so", RTLD_LAZY | RTLD_GLOBAL) # linux only
        dlopen("libpython2.7.so", RTLD_LAZY | RTLD_GLOBAL) # linux only
        Py_Initialize()
        initcymex()
        linalg_bypass()

    if nrhs == 0 or not mxGetClassID(rhs[0]) == mxCHAR_CLASS:
        mexErrMsgTxt(
            "usage: [output] = cymex('cmd', ...)"
            )
        return

    try:

        # consider first argument to mex function as the command
        cmd = mstr(rhs[0])

        if cmd == 'np':                     # ok 
            import numpy
            print numpy.random.randn(3)

        elif cmd == 'blas':                 # ok
            import numpy
            A = numpy.ones((3, 3))
            b = numpy.ones((3, ))
            print A.dot(b)

        elif cmd == 'lapack':               # MKL ERROR: Parameter 5 was incorrect on entry to DGESDD
            from numpy import linalg, r_
            A = r_[:20].reshape((4,5))
            print linalg.svd(A)[1]

        elif cmd == 'mlsin1':
            print mlsin(1)

        elif cmd == 'mlsinn': # 1e3 calls cymex mlsinn -> 6.3 s
            mlsin(int(1e3))

        else:
            print repr(cmd)

    except:
        import traceback, cStringIO
        cs = cStringIO.StringIO()
        traceback.print_exc(file=cs)
        txt = cs.getvalue()
        mexErrMsgTxt(txt)

    return # mexFunction

