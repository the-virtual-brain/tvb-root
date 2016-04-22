
import os
import numba
import numba.cuda

# http://numba.pydata.org/numba-doc/dev/cuda/simulator.html
try:
    CUDA_SIM = int(os.environ['NUMBA_ENABLE_CUDASIM']) == 1
except:
    CUDA_SIM = False


_cu_expr_type_map = {
    int: numba.int32,
    float: numba.float32
}


def cu_expr(expr, parameters, constants, return_fn=False):
    "Generate CUDA device function for given expression, with parameters and constants."
    ns = {}
    template = "from math import *\ndef fn(%s):\n    return %s"
    for name, value in constants.items():
        value_type = type(value)
        if value_type not in _cu_expr_type_map:
            msg = "unhandled constant type: %r" % (value_type, )
            raise TypeError(msg)
        ns[name] = _cu_expr_type_map[value_type](value)
    template %= ', '.join(parameters), expr
    exec template in ns
    fn = ns['fn']
    cu_fn = numba.cuda.jit(device=True)(fn)
    if return_fn:
        return cu_fn, fn
    return cu_fn