# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Experimental OpenCL models implementations.

.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""

import pyopencl
import pyopencl.array
import numpy
from ..models import ReducedWongWang
DEBUG = True
class CLComponent(object):

    def configure_opencl(self, context, queue):
        self._context = context
        self._queue = queue
        if hasattr(self, '_opencl_program_source'):
            self._program = pyopencl.Program(context, self._opencl_program_source).build()
        elif hasattr(self, '_opencl_program_source_file'):
            self._program = pyopencl.Program(context, open(getattr(self,'_opencl_program_source_file'),'r').read()).build()
class CLModel(CLComponent):

    def _alloc_opencl(self, n_nodes ,n_states=1,n_mode=1):

        if not hasattr(self, '_queue'):
            msg = "OpenCL components must be configured via the `configure_opencl` method prior to use."
            raise RuntimeError(msg)

        # arrays in component workspace
        arrays = {'state': (n_states, n_nodes,n_mode), 'coupling': (1, n_nodes), 'deriv': (n_states, n_nodes, n_mode)}
        self.arrays = arrays
        if hasattr(self, '_opencl_ordered_params'):
            arrays['param'] = (len(self._opencl_ordered_params), n_nodes)
            #arrays['param'] = (len(self._opencl_ordered_params), n_nodes, n_mode)
        # alloc opencl device arrays
        self._arrays = {}
        for name, shape in arrays.items():
            self._arrays[name] = pyopencl.array.Array(self._queue, shape, 'f')

        # fill parameter values
        if hasattr(self, '_opencl_ordered_params'):
            if(DEBUG): print(self._opencl_ordered_params)
            for i, name in enumerate(self._opencl_ordered_params):
                if(DEBUG): print((i, name))
                val = getattr(self, name)

                if val.size == 1:
                    val = val.flat[0]

                self._arrays['param'][i] = val

        # setup kernel arguments
        self._kernel.set_args(*[self._arrays[key].data for key in 'state coupling param deriv'.split()])

    def configure_opencl(self, context, queue):
        super(CLModel, self).configure_opencl(context, queue)


        self._kernel = self._program.dfun

    def dfun(self , queue, nt, args ):
        self._kernel(queue, (int(nt),), None, *[arg.data for arg in args])


    def dfunKernel(self, state_variables, coupling, local_coupling=0.0 ):
        n_states = state_variables.shape[0]
        n_nodes = state_variables.shape[1]
        n_mode = state_variables.shape[2]
        # allocate data if not yet done so
        if not hasattr(self, '_arrays'):
            self._alloc_opencl(n_nodes,n_states = n_states,n_mode = n_mode)

        # copy if passed host arrays
        if isinstance(state_variables, numpy.ndarray):
            # state_variables, coupling will be (1, n, 1)
            if(DEBUG):
                print("state_variables are ndarray", "states:", state_variables.shape, "coupling:", coupling.shape)

            #self._arrays['state'][:] = state_variables.reshape((1, n_states*n_nodes*n_mode)).astype('f')
            #self._arrays['coupling'][:] = coupling.reshape((1, n_nodes)).astype('f')

            # self._arrays['state'] = state_variables.flatten()
            #self._arrays['coupling'] = coupling.reshape((1, n_nodes)).astype('f')
            if (DEBUG):
                print("state_variable shape:",
                      state_variables.reshape((n_states, n_nodes * n_mode, 1)).astype('f').shape)
                print("array state shape", self._arrays['state'][:].shape)
            self._arrays['state'][:] = state_variables.reshape((n_states,  n_nodes , n_mode)).astype('f')
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
        print("Run kernel...")

        pyopencl.enqueue_nd_range_kernel(self._queue, self._kernel, (n_nodes,), None).wait()

        # return derivatives following input type
        deriv = self._arrays['deriv']
        if(DEBUG):
            print("derive shape:", deriv.shape)
        if isinstance(state_variables, numpy.ndarray):
            deriv = deriv.get().reshape((n_states, n_nodes, n_mode)).astype('d')

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
