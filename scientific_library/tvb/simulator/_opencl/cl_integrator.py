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

import pyopencl
import pyopencl.array
import numpy
from .models import CLComponent, CLModel
from ..integrators import HeunDeterministic, HeunStochastic, EulerDeterministic, EulerStochastic, RungeKutta4thOrderDeterministic , Identity
from .util import *

grw = lambda arr, ctx: pyopencl.Buffer(ctx, pyopencl.mem_flags.COPY_HOST_PTR, hostbuf=arr)

class CLIntegrator(CLComponent):
    _opencl_program_source_file = "baseIntegrator.cl"
    # vectors = lenOfVec * size, numOfThreads = size = numOfCoefficients
    _opencl_program_source = """
    __kernel void integrate( int    size   , int lenOfVec,
                            __global float *vectors, __global float *coefficients,__global float *ans
        ){
    int id = get_global_id(0), n = get_global_size(0);
    //printf("vectors: %d ans: %d ",vectors[id], ans[id]);

    for(int i = 0; i < size; i++){
        ans[id]+=vectors[i*lenOfVec+id]*coefficients[i];
    }
    }"""

    def cl_integrate(self, numOfCoefficient, lenOfVec, vectors, coefficients, ):
        n_states = vectors.shape[0]
        n_nodes = lenOfVec
        n_mode = vectors.shape[2]

        size = numpy.int32(numOfCoefficient)
        lenOfVec = numpy.int32(lenOfVec)
        vec_cl = grw(vectors, self._context)
        co_cl = grw(coefficients, self._context)
        ans = numpy.zeros((n_states,n_nodes,n_mode))
        ans_cl = grw(ans, self._context)
        args = size,lenOfVec,vec_cl,co_cl,ans_cl
        self._kernel(self._queue, (int(lenOfVec),), None, *args)


        pyopencl.enqueue_copy( self._queue, ans , ans_cl)

        return ans

    def _alloc_opencl(self, n_nodes, n_states=1, n_mode=1 , stochasticVariant = False):
        if not hasattr(self, '_queue'):
            msg = "OpenCL components must be configured via the `configure_opencl` method prior to use."
            raise RuntimeError(msg)

        arrays = {'state': (n_states, n_nodes, n_mode), 'coupling': (1, n_nodes), 'deriv': (n_states, n_nodes, n_mode)}
        self._arrays = {}
        if stochasticVariant:
            arrays['noise'] = (n_states, n_nodes, n_mode)

        for name, shape in list(arrays.items()):
            self._arrays[name] = pyopencl.array.Array(self._queue, shape, 'f')
            #print "Name: type:",isinstance(self._arrays[name],pyopencl.Buffer)
            #print "name: %s, shape %s"%(name,shape)

    def scheme(self,  X , dfun, coupling, local_coupling = 0.0, stimulus = 0.0):
        n_states = X.shape[0]
        n_nodes = X.shape[1]
        n_mode = X.shape[2]
        print("X shape:",X.shape)
        # check for openclArray buffer
        if not hasattr(self, '_arrays'):
            self._alloc_opencl( n_nodes,n_states = n_states, n_mode = n_mode , stochasticVariant = True )

        print("self: %s %s X: %s %s"%(self._arrays['state'].dtype, self._arrays['state'].size, X.dtype, X.size))
        self._arrays['state'].set( X )

        self._arrays['coupling'].set( coupling )

        #X_cl = pyopencl.array.Array(self._queue, X.shape ,'f')
        #coupling_cl = pyopencl.array.Array(self._queue, coupling.shape ,'f')
        X_next = dfun(self._arrays['state'], self._arrays['coupling'])

        return X_next



    def setup_method(self):
        self._context, self._queue = context_and_queue(create_cpu_context())

    def configure_opencl(self, context, queue):
        super(CLIntegrator, self).configure_opencl(context, queue)

        self._kernel = self._program.integrate


class CL_EulerDeterministic( CLIntegrator, EulerDeterministic ):
    #_opencl_program_source_file = "EulerDeterministic.cl"
    # opencl_program_source = ""

    def scheme(self, X, dfun, coupling, local_coupling=0.0, stimulus=0.0):
        n_states = X.shape[0]
        n_nodes = X.shape[1]
        n_mode = X.shape[2]
        print("X shape:", X.shape)
        # check for openclArray buffer
        if not hasattr(self, '_arrays'):
            self._alloc_opencl(n_nodes, n_states=n_states, n_mode=n_mode, stochasticVariant=True)

        #print "self: %s %s X: %s %s" % (self._arrays['state'].dtype, self._arrays['state'].size, X.dtype, X.size)
        self._arrays['state'].set(X)
        self._arrays['coupling'].set(coupling)
        self.dX = dfun(self._arrays['state'], self._arrays['coupling'])
        #X_next = self.dX + stimulus
        X_next = self._arrays['state'] + self.dt * (self.dX + stimulus)
        return X_next


class CL_Identity(CLIntegrator, Identity):

    def scheme(self, X, dfun, coupling=None, local_coupling=0.0, stimulus=0.0):
        n_states = X.shape[0]
        n_nodes = X.shape[1]
        n_mode = X.shape[2]
        print("X shape:", X.shape)
        # check for openclArray buffer
        if not hasattr(self, '_arrays'):
            self._alloc_opencl(n_nodes, n_states=n_states, n_mode=n_mode, stochasticVariant=True)

        self._arrays['state'].set(X)

        self._arrays['coupling'].set(coupling)
        self.dX = dfun(self._arrays['state'], self._arrays['coupling'])

        return self.dX + stimulus



class CL_HeunDeterministic(CLIntegrator, HeunDeterministic):
    def _alloc_opencl(self, n_nodes, n_states=1, n_mode=1):
        super(CL_HeunDeterministic, self)._alloc_opencl(n_nodes, n_states, n_mode, stochasticVariant=True)

        self._arrays['inter'] = pyopencl.array.Array(self._queue, (n_states, n_nodes, n_mode), 'f')

    def scheme(self, X, dfun, coupling= None, local_coupling=0.0, stimulus=0.0):
        n_states = X.shape[0]
        n_nodes = X.shape[1]
        n_mode = X.shape[2]
        print("X shape:", X.shape)
        # check for openclArray buffer
        if not hasattr(self, '_arrays'):
            self._alloc_opencl(n_nodes, n_states=n_states, n_mode=n_mode)

        self._arrays['state'].set(X)

        self._arrays['coupling'].set(coupling)
        self._arrays['inter'] = dfun(self._arrays['state'], self._arrays['coupling'])


        inter = self._arrays['state'] + self.dt * (self._arrays['inter'] + stimulus)
        #print "shceme: m_dx_tn",m_dx_tn.shape ,"inter",inter.shape
        self.clamp_state(inter)

        dX = (self._arrays['inter'] + dfun(inter, self._arrays['coupling'])) * self.dt / 2.0
        #print "scheme dX.shape: X_cl shape self.dt stimulus ",dX.shape, X_cl.shape, self.dt,stimulus
        X_next = self._arrays['state'] + dX + self.dt * stimulus

        self.clamp_state(X_next)

        return X_next


class CL_EulerStochastic(CLIntegrator, EulerStochastic):
    def _alloc_opencl(self, n_nodes, n_states=1, n_mode=1 ):
        super( CL_EulerStochastic, self)._alloc_opencl(n_nodes, n_states, n_mode, stochasticVariant= True )

        self._arrays['inter'] = pyopencl.array.Array(self._queue, (n_states, n_nodes, n_mode), 'f')

    def scheme(self, X, dfun, coupling, local_coupling = 0.0, stimulus = 0.0):
        n_states = X.shape[0]
        n_nodes = X.shape[1]
        n_mode = X.shape[2]
        print("X shape:", X.shape)
        # check for openclArray buffer
        if not hasattr(self, '_arrays'):
            self._alloc_opencl( n_nodes, n_states=n_states, n_mode=n_mode )

        self._arrays['state'].set(X)
        self._arrays['coupling'].set(coupling)
        noise = self.noise.generate(X.shape).astype(numpy.float32)
        self._arrays['noise'].set(noise)

        dX = dfun(self._arrays['state'], self._arrays['coupling']) * self.dt
        noise_gfun = self.noise.gfun(X)
        X_next =  self._arrays['state'] + dX + noise_gfun * self._arrays['noise'] + self.dt * stimulus
        self.clamp_state(X_next)
        return X_next



class CL_HeunStochastic( CLIntegrator, HeunStochastic ):
    def _alloc_opencl(self, n_nodes, n_states=1, n_mode=1):
        super(CL_HeunStochastic, self)._alloc_opencl(n_nodes, n_states, n_mode, stochasticVariant=True)

        self._arrays['inter'] = pyopencl.array.Array(self._queue, (n_states, n_nodes, n_mode), 'f')

    def scheme(self,  X , dfun, coupling, local_coupling = 0.0, stimulus = 0.0):
        n_states = X.shape[0]
        n_nodes = X.shape[1]
        n_mode = X.shape[2]
        print("X shape:", X.shape)
        # check for openclArray buffer
        if not hasattr(self, '_arrays'):
            self._alloc_opencl(n_nodes, n_states=n_states, n_mode=n_mode )

        self._arrays['state'].set(X)
        self._arrays['coupling'].set(coupling)

        noise = self.noise.generate(X.shape).astype(numpy.float32)
        self._arrays['noise'].set(noise)

        noise_gfun = self.noise.gfun(X)

        if (noise_gfun.shape != (1,) and noise.shape[0] != noise_gfun.shape[0]):
            msg = str("Got shape %s for noise but require %s."
                      " You need to reconfigure noise after you have changed your model." % (
                          noise_gfun.shape, (noise.shape[0], noise.shape[1])))
            raise Exception(msg)

        m_dx_tn = dfun(self._arrays['state'], self._arrays['coupling'])

        noise *= noise_gfun

        self._arrays['inter'] = self._arrays['state'] + self.dt * m_dx_tn + self._arrays['noise'] + self.dt * stimulus
        self.clamp_state(self._arrays['inter'])

        dX = (m_dx_tn + dfun(self._arrays['inter'], self._arrays['coupling'] )) * self.dt / 2.0

        X_next = self._arrays['state'] + dX + self._arrays['noise'] + self.dt * stimulus
        self.clamp_state(X_next)

        return X_next




class CL_RungeKutta4thOrderDeterministic( CLIntegrator, RungeKutta4thOrderDeterministic):
    def scheme(self,  X , dfun, coupling, local_coupling = 0.0, stimulus = 0.0):

        n_states = X.shape[0]
        n_nodes = X.shape[1]
        n_mode = X.shape[2]
        print("X shape:", X.shape)
        # check for openclArray buffer
        if not hasattr(self, '_arrays'):
            self._alloc_opencl(n_nodes, n_states=n_states, n_mode=n_mode)

        self._arrays['state'].set(X)
        self._arrays['coupling'].set(coupling)

        dt = self.dt
        dt2 = dt / 2.0
        dt6 = dt / 6.0

        k1 = dfun(self._arrays['state'], self._arrays['coupling'])
        self._arrays['inter'] = self._arrays['state'] + dt2 * k1

        self.clamp_state(self._arrays['inter'])
        k2 = dfun(self._arrays['inter'], self._arrays['coupling'])
        self._arrays['inter'] = self._arrays['state'] + dt2 * k2

        self.clamp_state( self._arrays['inter'])
        k3 = dfun( self._arrays['inter'], self._arrays['coupling'])
        self._arrays['inter'] = self._arrays['state'] + dt * k3
        self.clamp_state(self._arrays['inter'])
        k4 = dfun(self._arrays['inter'], self._arrays['coupling'])

        dX = dt6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        X_next = self._arrays['state'] + dX + self.dt * stimulus
        self.clamp_state(X_next)
        return X_next


