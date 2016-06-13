"""
Experimental OpenCL models implementations.

.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""

import pyopencl
import pyopencl.array
import numpy
from ..models import ReducedWongWang

class CLComponent(object):

    def configure_opencl(self, context, queue):
        self._context = context
        self._queue = queue
        if hasattr(self, '_opencl_program_source'):
            self._program = pyopencl.Program(context, self._opencl_program_source).build()

class CLModel(CLComponent):

    def _alloc_opencl(self, n_nodes ,n_states=1):

        if not hasattr(self, '_queue'):
            msg = "OpenCL components must be configured via the `configure_opencl` method prior to use."
            raise RuntimeError(msg)

        # arrays in component workspace
        arrays = {'state': (n_states, n_nodes), 'coupling': (1, n_nodes), 'deriv': (n_states, n_nodes)}

        if hasattr(self, '_opencl_ordered_params'):
            arrays['param'] = (len(self._opencl_ordered_params), n_nodes)

        # alloc opencl device arrays
        self._arrays = {}
        for name, shape in arrays.items():
            self._arrays[name] = pyopencl.array.Array(self._queue, shape, 'f')

        # fill parameter values
        if hasattr(self, '_opencl_ordered_params'):
            for i, name in enumerate(self._opencl_ordered_params):
                val = getattr(self, name)
                if val.size == 1:
                    val = val.flat[0]
                self._arrays['param'][i] = val

        # setup kernel arguments
        self._kernel.set_args(*[self._arrays[key].data for key in 'state coupling param deriv'.split()])

    def configure_opencl(self, context, queue):
        super(CLModel, self).configure_opencl(context, queue)
        self._kernel = self._program.dfun


    def dfun(self, state_variables, coupling, local_coupling=0.0):

        n_nodes = state_variables.shape[1]

        # allocate data if not yet done so
        if not hasattr(self, '_arrays'):
            self._alloc_opencl(n_nodes)

        # copy if passed host arrays
        if isinstance(state_variables, numpy.ndarray):
            # state_variables, coupling will be (1, n, 1)
            self._arrays['state'][:] = state_variables.reshape((1, n_nodes)).astype('f')
            self._arrays['coupling'][:] = coupling.reshape((1, n_nodes)).astype('f')

        # set kernel arg if passed device arrays
        elif isinstance(state_variables, pyopencl.array.Array):
            self._kernel.set_args(state_variables.data,
                                  coupling.data,
                                  self._arrays['param'].data,
                                  self._arrays['deriv'].data)

        # otherwise, complain
        else:
            raise TypeError('unsupported data type %r', type(state_variables))

        # run the kernel and wait
        pyopencl.enqueue_nd_range_kernel(self._queue, self._kernel, (n_nodes,), None).wait()

        # return derivatives following input type
        deriv = self._arrays['deriv']
        if isinstance(state_variables, numpy.ndarray):
            deriv = deriv.get().reshape((1, n_nodes, 1)).astype('d')

        return deriv

class CLRWW(ReducedWongWang, CLModel):

    _opencl_ordered_params = 'a b d gamma tau_s w J_N I_o'.split()

    _opencl_program_source = """
    __kernel void dfun(__global float *state, __global float *coupling,
                       __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        // this is boilerplate and could be generated
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