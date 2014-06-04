"Base backend driver classes"

import string
import logging

from numpy import int32, float32

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

from . import templates

class Code(object):
    "Interface managing code to run on a device"

    def build_kernel(self, src=None, templ='tvb.cu', debug=False, **kwds):
        self.source = src or templates.sources[templ]
        self.kernel = string.Template(self.source).substitute(**kwds)
        if debug:
            temppath = os.path.abspath('./temp.cu')
            LOG.debug('completed template written to %r', temppath)
            with open(temppath, 'w') as fd:
                fd.write(source)


    @classmethod
    def build(cls, fns=[], T=string.Template, templ='tvb.cu', debug=False, **kwds):

        source = T(templates.sources[templ]).substitute(**args) 

        if debug:
            temppath = os.path.abspath('./temp.cu')
            LOG.debug('completed template written to %r', temppath)
            with open(temppath, 'w') as fd:
                fd.write(source)

        if using_gpu:
            cls.mod = CUDASourceModule("#define TVBGPU\n" + source, 
                                       options=["--ptxas-options=-v"])
        else:
            cls.mod = cee.srcmod("#include <math.h>\n" + source, fns)


class Global(object):
    "Interface to global (scalar) variables stored on device. "

    def __init__(self, name, dtype):
        self.code  = None # Code.build()
        self.name  = name
        self.dtype = dtype
        self.__post_init = True

    def post_init(self):
        self.__post_init = False

    # __get__
    # __set__


class Array(object):
    "Interface to N-dim. array stored on device"

    @property
    def cpu(self):
        if not hasattr(self, '_cpu'):
            self._cpu = zeros(self.shape).astype(self.type)
        return self._cpu

    @property
    def shape(self):
        return tuple(getattr(self.parent, k) for k in self.dimensions)

    @property
    def nbytes(self):
        bytes_per_elem = empty((1,), dtype=self.type).nbytes
        return prod(self.shape)*bytes_per_elem

    # @property device
    # @property value
    # def set(arg)

    def __init__(self, name, type, dimensions):
        self.parent = None
        self.name = name
        self.type = type
        self.dimensions = dimensions

class Driver(object):
    "Base driver class"

class RegionParallel(Driver):
    "Driver for parallel region simulations"


    #########################################################
    # simulation workspace dimensions
    _dimensions = set(['horizon', 'n_node', 'n_thr', 'n_rthr', 'n_svar', 
                       'n_cvar', 'n_cfpr', 'n_mmpr', 'n_nspr', 'n_inpr', 
                       'n_tavg', 'n_msik', 'n_mode'])

    # generate accessors for global constants
    horizon = Global('horizon', int32)
    n_node  = Global('n_node', int32)
    n_thr   = Global('n_thr', int32)
    n_rthr  = Global('n_rthr', int32)
    n_svar  = Global('n_svar', int32)
    n_cvar  = Global('n_cvar', int32)
    n_cfpr  = Global('n_cfpr', int32)
    n_mmpr  = Global('n_mmpr', int32)
    n_nspr  = Global('n_nspr', int32)
    n_inpr  = Global('n_inpr', int32)
    n_tavg  = Global('n_tavg', int32)
    n_msik  = Global('n_msik', int32)
    n_mode  = Global('n_mode', int32)


    ##########################################################
    # simulation workspace arrays, also an ORDERED list of the arguments
    # to the update function
    device_state = ['idel', 'cvars', 'inpr', 'conn', 'cfpr', 'nspr', 'mmpr',
        'input', 'x', 'hist', 'dx1', 'dx2', 'gx', 'ns', 'stim', 'tavg']

    # thread invariant, call invariant
    idel  = Array('idel',    int32, ('n_node', 'n_node'))
    cvars = Array('cvars',   int32, ('n_cvar', ))
    inpr  = Array('inpr',  float32, ('n_inpr', ))

    # possibly but not currently thread varying, call invariant
    conn  = Array('conn',  float32, ('n_node', 'n_node'))

    # thread varying, call invariant
    nspr  = Array('nspr',  float32, ('n_node', 'n_nspr', 'n_thr'))
    mmpr  = Array('mmpr',  float32, ('n_node', 'n_mmpr', 'n_thr'))
    cfpr  = Array('cfpr',  float32, (          'n_cfpr', 'n_thr'))

    # thread varying, call varying
    input = Array('input', float32, (                     'n_cvar', 'n_thr'))
    x     = Array('x',     float32, (           'n_node', 'n_svar', 'n_thr'))
    hist  = Array('hist',  float32, ('horizon', 'n_node', 'n_cvar', 'n_thr'))
    dx1   = Array('dx1',   float32, (                     'n_svar', 'n_thr'))
    dx2   = Array('dx2',   float32, (                     'n_svar', 'n_thr'))
    gx    = Array('gx',    float32, (                     'n_svar', 'n_thr'))
    ns    = Array('ns',    float32, (           'n_node', 'n_svar', 'n_thr'))
    stim  = Array('stim',  float32, (           'n_node', 'n_svar', 'n_thr'))
    tavg  = Array('tavg',  float32, (           'n_node', 'n_svar', 'n_thr'))

    

    _example_code = {
        'model_dfun': """
        float a   = P(0)
            , b   = P(1)
            , tau = P(2)
            , x   = X(0)
            , y   = X(1) ;

        DX(0) = (x - x*x*x/3.0 + y)*tau;
        DX(1) = (a + b*y - x + I(0))/tau;
        """,

        'noise_gfun': """
        float nsig;
        for (int i_svar=0; i_svar<n_svar; i_svar++)
        {
            nsig = P(i_svar);
            GX(i_svar) = sqrt(2.0*nsig);
        }
        """, 

        'integrate': """
        float dt = P(0);
        model_dfun(dx1, x, mmpr, input);
        noise_gfun(gx, x, nspr);
        for (int i_svar=0; i_svar<n_svar; i_svar++)
            X(i_svar) += dt*(DX1(i_svar) + STIM(i_svar)) + GX(i_svar)*NS(i_svar);
        """,

        'coupling': """
        // parameters
        float a = P(0);

        I = 0.0;
        for (int j_node=0; j_node<n_node; j_node++, idel++, conn++)
            I += a*GIJ*XJ;


        """
        }

    def build_kernel(self):
        args = dict()
        for k in 'model_dfun noise_gfun integrate coupling'.split():
            if k not in kwds:
                LOG.debug('using example code for %r', k)
                src = cls._example_code[k]
            else:
                src = kwds[k]
            args[k] = src


    def fill_with(self, idx, sim):
        """
        fill_with extracts the necessary workspace arrays from the 
        sequence of simulators, and packs the data into the device's 
        workspace arrays.

        conn and idelays are transposed because the simulator uses the 
        convention that weights[i, j] is the weight from i to j, where
        the device code sez that the weight is from j to i. Go figure.

        """

        if idx == 0:
            self.idel  .cpu[:] = sim.connectivity.idelays.T
            self.cvars .cpu[:] = sim.model.device_info.cvar
            self.inpr  .cpu[:] = array([sim.integrator.dt], dtype=float32)

            # could eventually vary with idx
            self.conn  .cpu[:] = sim.connectivity.weights.T

        self.cfpr  .cpu[..., idx] = sim.coupling.device_info.cfpr
        self.mmpr  .cpu[..., idx] = sim.model.device_info.mmpr

        # each device_info should know the required shape, so none of these
        # assignments should fail with a shape error
        if hasattr(sim.integrator, 'noise'):
            self.nspr  .cpu[..., idx] = sim.integrator.noise.device_info.nspr

        # sim's history shape is (horizon, n_svars, n_nodes, n_modes)

        # our state's shape is  ('n_node', 'n_svar', 'n_thr'))
        # and we fold modes into svars
        state = sim.history[sim.current_step].transpose((1, 0, 2))
        self.x     .cpu[..., idx] = state.reshape((-1, prod(state.shape[-2:])))

        # and our history's shape is ('horizon', 'n_node', 'n_cvar', 'n_thr'))
        history = sim.history[:, sim.model.cvar].transpose((0, 2, 1, 3))
        with_modes_folded = history.shape[:2] + (prod(history.shape[-2:]),)
        self.hist  .cpu[..., idx] = history.reshape(with_modes_folded)
       
        # several arrays are quite simply temporary storage, no need
        # to set them here
        """
        self.input .cpu[..., idx] = #(                     'n_cvar', 'n_thr'))
        self.dx1   .cpu[..., idx] = #(                     'n_svar', 'n_thr'))
        self.dx2   .cpu[..., idx] = #(                     'n_svar', 'n_thr'))
        self.gx    .cpu[..., idx] = #(                     'n_svar', 'n_thr'))
        self.tavg  .cpu[..., idx] = #(           'n_node', 'n_svar', 'n_thr'))

        """

        # some arrays will be set at every step
        """
        self.ns    .cpu[..., idx] = #(           'n_node', 'n_svar', 'n_thr'))
        self.stim  .cpu[..., idx] = #(           'n_node', 'n_svar', 'n_thr'))

        """

    block_dim = property(lambda s: 256)

    def __init__(self, code_args={}, **kwds):

        fns = ['update'] + [f+d for d in self._dimensions 
                                for f in ['set_', 'get_']]

        device_code.build(fns, **code_args)

        for k in self._dimensions:
            if k in kwds:
                setattr(self, k, kwds.get(k))
            else:
                missing = self._dimensions - set(kwds.keys())
                if missing:
                    msg = 'device_handler requires the keyword value %r' % missing
                    raise TypeError(msg)

        for k in self.device_state:
            getattr(self, k).parent = self
        
        self._device_update = self.get_update_fun()

        self.i_step = 0

    @classmethod
    def init_like(cls, sim, n_msik=1):
        """
        The init_from classmethod builds a device_handler based on a 
        prototype template

        """

        # make sure we're supported 
        components = ['model', 'integrator', 'coupling']
        for component in components:
            obj = eval('sim.%s' % (component,))
            if not (hasattr(obj, 'device_info') and 
                    (hasattr(sim.integrator.noise, 'device_info')
                        if isinstance(obj, integrators.IntegratorStochastic)
                        else True)):
                msg = "%r does not support device execution" % obj
                raise TypeError(msg)

        dt = sim.integrator.dt
        tavg = [m for m in sim.monitors if isinstance(m, monitors.TemporalAverage)]
        n_tavg = int(tavg[0].period / dt) if tavg else 1
        if tavg and not n_tavg == tavg[0].period / dt:
            msg = """given temporal average period coerced to integer multiple
            of integration time step, %f -> %f"""
            msg %= (tavg[0].period, n_tavg*dt)
            LOG.warning(msg)

        if n_msik > 1:
            LOG.warning("%r: %s", self, """
            implementation of noise & stimulus in the kernel does not allow for
            multiple steps in kernel, but is acceptable for simulations that
            are deterministic, w/o stimulus. 
            """)

        # is noisy?
        stoch = isinstance(sim.integrator, integrators.IntegratorStochastic)

        # extract dimensions of simulation
        dims = {'horizon': sim.horizon,
                'n_node' : sim.number_of_nodes,
                'n_thr'  : 0, # caller can use nbytes1
                'n_rthr' : 1, # caller resets 
                'n_svar' : sim.model            .device_info.n_svar,
                'n_cvar' : sim.model            .device_info.n_cvar,
                'n_cfpr' : sim.coupling         .device_info.n_cfpr,
                'n_mmpr' : sim.model            .device_info.n_mmpr,
                'n_inpr' : sim.integrator       .device_info.n_inpr,
                'n_nspr' : sim.integrator.noise .device_info.n_nspr if stoch else 1,
                'n_mode' : sim.model            .device_info.n_mode,

                'n_tavg' : n_tavg, 'n_msik' : n_msik }

        # extract code from simulator components
        code = {'model_dfun': sim.model.device_info.kernel,
                'integrate':  sim.integrator.device_info.kernel,
                'noise_gfun': sim.integrator.noise.device_info.kernel if stoch else "",
                'coupling':   sim.coupling.device_info.kernel}

        dh = cls(code_args=code, **dims)
        return dh


    @property
    def nbytes(self):

        memuse = sum([getattr(self, k).nbytes for k in self.device_state])

        free, total = self.mem_info

        if free and total: # are not None
            if memuse > free:
                print '%r: nbytes=%d exceeds free device memory' % (self, memuse)
            if memuse > total:
                raise MemoryError('%r: nbytes=%d exceeds total device memory' % (self, memuse))

        return memuse

    @property
    def nbytes1(self):
        old_n_thr = self.n_thr
        self.n_thr = 1
        nbs = self.nbytes
        self.n_thr = old_n_thr 
        return nbs

    def __call__(self):
        raise NotImplementedError
