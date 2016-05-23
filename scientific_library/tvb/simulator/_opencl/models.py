"""
Experimental OpenCL models implementations.

.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""

import pyopencl
import pyopencl.array
from ..models import ReducedWongWang


class CLRWW(ReducedWongWang):

    _opencl_configured = False
    _opencl_ordered_params = 'a b d gamma tau_s w J_N I_o'.split()
    _opencl_program_source = """
    __kernel void dfun(__global float *state, __global float *coupling,
                       __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);
        float S=state[i], a=param[i], b=param[n+i], d=param[2*n+i], g=param[3*n+i],
              ts=param[4*n+i], w=param[5*n+i], j=param[6*n+i], io=param[7*n+i];

        float x = w*j*S + io + j*coupling[i];
        float h = (a*x - b) / (1.0f - exp(-d*(a*x - b)));
        float dx = - (S / ts) + (1.0f - S) * h * g;

        bool neg = S < 0.0f;
        bool gt1 = S > 1.0f;
        bool i01 = (!neg) && (!gt1);

        deriv[i] = neg*(0.0f - S) + gt1*(1.0f - S) + i01*dx;
    }
    """

    def configure_opencl(self, context, queue):
        self._context = context
        self._queue = queue
        self._program = pyopencl.Program(context, self._opencl_program_source).build()
        self._kernel = self._program.dfun

    def _alloc_opencl(self, n_nodes):
        if not hasattr(self, '_queue'):
            msg = "OpenCL components must be configured via the `configure_opencl` method prior to use."
            raise RuntimeError(msg)
        # arrays in component workspace
        arrays = {'state': (1, n_nodes), 'coupling': (1, n_nodes), 'deriv': (1, n_nodes),
                  'param': (len(self._opencl_ordered_params), n_nodes)}
        # alloc opencl device arrays
        self._arrays = {}
        for name, shape in arrays.items():
            self._arrays[name] = pyopencl.array.Array(self._queue, shape, 'f')
        # fill parameter values
        for i, name in enumerate(self._opencl_ordered_params):
            val = getattr(self, name)
            if val.size == 1:
                val = val.flat[0]
            self._arrays['param'][i] = val
        # setup kernel arguments
        self._kernel.set_args(*[self._arrays[key].data for key in 'state coupling param deriv'.split()])

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        n_nodes = state_variables.shape[1]
        if not hasattr(self, '_arrays'):
            self._alloc_opencl(n_nodes)
        pyopencl.enqueue_nd_range_kernel(self._queue, self._kernel, (n_nodes, ), None).wait()
        return self._arrays['deriv']


class CLRWWHostDevCopy(CLRWW):

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        n_nodes = state_variables.shape[1]
        if not hasattr(self, '_arrays'):
            self._alloc_opencl(n_nodes)
        # state_variables, coupling will be (1, n, 1)
        self._arrays['state'][:] = state_variables.reshape((1, n_nodes)).astype('f') # this will work ok except for modal models
        self._arrays['coupling'][:] = coupling.reshape((1, n_nodes)).astype('f')
        pyopencl.enqueue_nd_range_kernel(self._queue, self._kernel, (n_nodes, ), None).wait()
        return self._arrays['deriv'].get().reshape((1, n_nodes, 1)).astype('d')