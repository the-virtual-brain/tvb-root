import pyopencl
import pyopencl.array
import numpy
from models import CLComponent, CLModel
from ..integrators import HeunDeterministic, HeunStochastic, EulerDeterministic, EulerStochastic, Identity
from util import *

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

    def cl_integrate(self, numOfCoefficient, lenOfVec, vectors, coefficients, ans):
        size = numpy.int32(numOfCoefficient)
        lenOfVec = numpy.int32(lenOfVec)
        vec_cl = grw(vectors, self._context)
        co_cl = grw(coefficients, self._context)
        ans_cl = grw(ans, self._context)
        args = size,lenOfVec,vec_cl,co_cl,ans_cl
        self._kernel(self._queuequeue, (int(lenOfVec),), None, *args)

        tmp = numpy.zeros((lenOfVec, 1), 'f')
        pyopencl.enqueue_copy( self._queue, tmp , ans_cl)
        return tmp

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


class CL_EulerDeterministic( CLIntegrator, EulerDeterministic ):
    #_opencl_program_source_file = "EulerDeterministic.cl"
    # opencl_program_source = ""

    def scheme(self, X, dfun, coupling, local_coupling=0.0, stimulus=0.0):
        X_cl, coupling_cl = pyopencl.array.Array(self._queue, X.shape, 'f'), \
                            pyopencl.array.Array(self._queue, coupling.shape, 'f')
        self.dX = dfun(X_cl, coupling_cl)
        vectors = numpy.vstack((X_cl.data,self.dX))

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

        self.dx = clModel.dfunKernel( X_cl, coupling_cl,local_coupling )
        vectors = self.dx + stimulus
        coefficients = [1,1]
        ans = pyopencl.array.Array(self._queue, (len(X)), 'f')
        
        return


class CL_HeunDeterministic(CLIntegrator, HeunDeterministic):

    def scheme(self, X, dfun, coupling= None, local_coupling=0.0, stimulus=0.0):
        X_cl, coupling_cl = pyopencl.array.Array(self._queue, X.shape, 'f'),\
                            pyopencl.array.Array(self._queue, coupling.shape, 'f')
        m_dx_tn = dfun(X_cl, coupling_cl, local_coupling)
        inter = X_cl + self.dt * (m_dx_tn + stimulus)
        self.clamp_state(inter)

        dX = (m_dx_tn + dfun(inter, coupling_cl, local_coupling)) * self.dt / 2.0

        X_next = X_cl + dX + self.dt * stimulus
        self.clamp_state(X_next)

        return X_next

    def scheme_cl(self, X, clModel, coupling, local_coupling=0.0, stimulus=0.0):
        if not isinstance(clModel, CLModel):
            msg = "the datatype of clModel should be openclModel"
            raise RuntimeError(msg)
        clModel.configure_opencl(self._context, self._queue)
        X_cl, coupling_cl = pyopencl.array.Array(self._queue, X.shape, 'f'), \
                            pyopencl.array.Array(self._queue, coupling.shape, 'f')

        X_next = clModel.dfunKernel(X_cl, coupling_cl, local_coupling) + stimulus
        vector = X_cl + coupling_cl
        return X_next