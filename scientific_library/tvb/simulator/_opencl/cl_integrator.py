import pyopencl
import pyopencl.array
import numpy
from models import CLComponent, CLModel
from ..integrators import HeunDeterministic, HeunStochastic, EulerDeterministic, EulerStochastic, Identity
from util import *
class CLIntegrator(CLComponent):
    _opencl_program_source_file = "baseIntegrator.cl"
    _opencl_program_source = """
        __kernel void integrate(
            __global float *vectors, __global float *coefficients,
            __global float *ans,
            int    size   , int lenOfVec
            ){

        int id = get_global_id(0), n = get_global_size(0);

        for(int i = 0; i < size; i++){
            ans[id] += vectors[i*lenOfVec+id];
        } }"""

    def _alloc_opencl(self, n_nodes, n_states=1, n_mode=1):
        if not hasattr(self, '_queue'):
            msg = "OpenCL components must be configured via the `configure_opencl` method prior to use."
            raise RuntimeError(msg)

        arrays = {'state': (n_states, n_nodes, n_mode), 'coupling': (1, n_nodes), 'deriv': (n_states, n_nodes, n_mode)}

        for name, shape in arrays.items():
            self._arrays[name] = pyopencl.array.Array(self._queue, shape, 'f')

        pass
    def scheme(self,  X , dfun, coupling, local_coupling = 0.0, stimulus = 0.0):

        X_cl = pyopencl.array.Array(self._queue, X.shape ,'f')
        coupling_cl = pyopencl.array.Array(self._queue, coupling.shape ,'f')
        X_next = dfun(X_cl, coupling_cl)

        return X_next
    #TODO pass ans, st, to model; call kernel function of integrator

    def scheme_cl(self, X, clModel, coupling, local_coupling=0.0, stimulus=0.0):
        if not isinstance(clModel, CLModel ):
            msg = "the datatype of clModel should be openclModel"
            raise RuntimeError(msg)
        clModel.configure_opencl(self._context,self._queue)

        X_cl = pyopencl.array.Array(self._queue, X.shape, 'f')
        coupling_cl = pyopencl.array.Array(self._queue, coupling.shape, 'f')
        X_next = clModel.dfunKernel(X_cl, coupling_cl)

        return X_next

    def setUp(self):
        self._context, self._queue = context_and_queue(create_cpu_context())

    def configure_opencl(self, context, queue):
        super(CLIntegrator, self).configure_opencl(context, queue)

        self._kernel = self._program.integrate

class CL_EulerDeterministic(CLIntegrator, EulerDeterministic):
    #_opencl_program_source_file = "EulerDeterministic.cl"
    #_opencl_program_source = """ """

    def scheme(self, X, dfun, coupling, local_coupling=0.0, stimulus=0.0):
        X_cl = pyopencl.array.Array(self._queue, X, 'f')
        coupling_cl = pyopencl.array.Array(self._queue, coupling, 'f')
        self.dX = dfun(X_cl, coupling_cl)
        X_next = X + self.dt * (self.dX + stimulus)
        return X_next

class CL_EulerDeterministic( CLIntegrator, EulerDeterministic ):
    #_opencl_program_source_file = "EulerDeterministic.cl"
    # opencl_program_source = ""

    def scheme(self, X, dfun, coupling, local_coupling=0.0, stimulus=0.0):
        X_cl, coupling_cl = pyopencl.array.Array(self._queue, X.shape, 'f'), \
                            pyopencl.array.Array(self._queue, coupling.shape, 'f')
        self.dX = dfun(X_cl, coupling_cl)

        X_next = X_cl + self.dt * (self.dX + stimulus)

        return X_next

    def scheme_cl(self, X, clModel, coupling, local_coupling=0.0, stimulus=0.0):

        if not isinstance(clModel, CLModel):
            msg = "the datatype of clModel should be openclModel"
            raise RuntimeError(msg)
        clModel.configure_opencl(self._context, self._queue)
        X_cl, coupling_cl = pyopencl.array.Array(self._queue, X.shape, 'f'), \
                            pyopencl.array.Array(self._queue, coupling.shape, 'f')

        X_next = clModel.dfunKernel(X_cl, coupling_cl, local_coupling) + stimulus
        return X_next

class CL_Identity(CLIntegrator, Identity):

    def scheme(self, X, dfun, coupling=None, local_coupling=0.0, stimulus=0.0):
        X_cl, coupling_cl = pyopencl.array.Array(self._queue, X.shape, 'f'),\
                            pyopencl.array.Array(self._queue, coupling.shape, 'f')
        return dfun(X, coupling, local_coupling) + stimulus


    def scheme_cl(self, X, clModel, coupling, local_coupling=0.0, stimulus=0.0):
        if not isinstance(clModel, CLModel):
            msg = "the datatype of clModel should be openclModel"
            raise RuntimeError(msg)
        clModel.configure_opencl(self._context, self._queue)
        X_cl, coupling_cl = pyopencl.array.Array(self._queue, X.shape, 'f'), \
                            pyopencl.array.Array(self._queue, coupling.shape, 'f')

        X_next = clModel.dfunKernel( X_cl, coupling_cl,local_coupling ) + stimulus
        return X_next