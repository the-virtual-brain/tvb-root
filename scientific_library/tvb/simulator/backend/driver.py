# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (in press)
#
#

import sys
import os.path
import string
import glob
import subprocess
import logging

from numpy import *

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

import driver_conf as dc

using_gpu = getattr(dc, 'using_gpu', 0)
LOG.debug('using_gpu = %r', using_gpu)

if using_gpu:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule as CUDASourceModule
    from pycuda import gpuarray
    import pycuda.tools

    class OccupancyRecord(pycuda.tools.OccupancyRecord):
        def __repr__(self):
            ret = "Occupancy(tb_per_mp=%d, limited_by=%r, warps_per_mp=%d, occupancy=%0.3f)"
            return ret % (self.tb_per_mp, self.limited_by, self.warps_per_mp, self.occupancy)

    _, total_mem = cuda.mem_get_info()

    from pycuda.curandom import XORWOWRandomNumberGenerator as XWRNG
    rng = XWRNG()

    # FIXME should be on the noise objects, but has different interface
    # FIXME this is additive white noise
    def gen_noise_into(devary, dt):
        gary = devary.device
        rng.fill_normal(gary)
        gary.set(gary.get()*sqrt(dt))

else: # using Cpu
    import psutil
    import ctypes

    total_mem = psutil.phymem_usage().total

    # FIXME this is additive white noise
    def gen_noise_into(devary, dt):
        devary.cpu[:] = random.normal(size=devary.shape)
        devary.cpu[:] *= sqrt(dt)

from tvb.simulator.lab import *

from . import cee, templates


class device_code(object):

    @classmethod
    def build(cls, fns=[], T=string.Template, templ='tvb.cu', debug=False,
              **kwds):
        """
        Build a device code object based on code template and arguments. You
        may provide the following keyword arguments to customize the template:

            ``model_dfun``: mass model dynamics' equations
            ``noise_gfun``: noise coefficient calculation
            ``integrate``: integration scheme
            ``coupling``: coupling function

        In each case, a default is available (FitzHugh-Nagumo dynamics, linear
        additive noise, stochastic Euler & linear coupling; respectively), on
        the device_code.defaults attribute.

        Please refer to tvb.cu template file for the context in which each
        template argument is used, the definition of the macros, etc. The following
        illustrative examples show what kind of code can be inserted


        """

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


        args = dict()
        for k in 'model_dfun noise_gfun integrate coupling'.split():
            if k not in kwds:
                LOG.debug('using example code for %r', k)
                src = cls._example_code[k]
            else:
                src = kwds[k]
            args[k] = src

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

class device_global(object):
    """
    Encapsulates a source module device global in a Python data descriptor
    for easy handling

    """

    def __init__(self, name, dtype):
        self.code  = device_code
        self.name  = name
        self.dtype = dtype
        self.__post_init = True

    def post_init(self):
        if self.__post_init:
            if using_gpu:
                self.ptr   = self.code.mod.get_global(self.name)[0]
            else:
                self._cget = getattr(self.code.mod, 'get_'+self.name)
                self._cset = getattr(self.code.mod, 'set_'+self.name)
            self.__post_init = False

    def __get__(self, inst, ownr):
        self.post_init()
        if using_gpu:
            buff = array([0]).astype(self.dtype)
            cuda.memcpy_dtoh(buff, self.ptr)
            return buff[0]
        else:
            ret = self._cget()
            return ret


    def __set__(self, inst, val):
        self.post_init()
        if using_gpu:
            cuda.memcpy_htod(self.ptr, self.dtype(val))
            buff = empty((1,)).astype(self.dtype)
            cuda.memcpy_dtoh(buff, self.ptr)
        else:
            ctype = ctypes.c_int32 if self.dtype==int32 else ctypes.c_float
            self._cset(ctype(val))
        

class device_array(object):
    """
    Encapsulates an array that is on the device

    """

    @property
    def cpu(self):
        if not hasattr(self, '_cpu'):
            self._cpu = zeros(self.shape).astype(self.type)
        return self._cpu

    @property
    def device(self):
        if not hasattr(self, '_device'):
            if using_gpu:
                if self.pagelocked:
                    raise NotImplementedError
                self._device = gpuarray.to_gpu(self.cpu)
            else:
                ctype = ctypes.c_float if self.type==float32 else ctypes.c_int32
                ptrtype = ctypes.POINTER(ctype)
                self._device = ascontiguousarray(self.cpu).ctypes.data_as(ptrtype)
        return self._device

    def set(self, ary):
        """
        In place update the device array.
        """
        _ = self.device
        if using_gpu:
            self._device.set(ary)
        else:
            delattr(self, '_device')
            self._cpu[:] = ary
            _ = self.cpu

    @property
    def value(self):
        return self.device.get() if using_gpu else self.cpu.copy()

    @property
    def shape(self):
        return tuple(getattr(self.parent, k) for k in self.dimensions)

    @property
    def nbytes(self):
        bytes_per_elem = empty((1,), dtype=self.type).nbytes
        return prod(self.shape)*bytes_per_elem

    def __init__(self, name, type, dimensions, pagelocked=False):
        self.parent = None
        self.name = name
        self.type = type
        self.dimensions = dimensions
        if using_gpu:
            self.pagelocked = pagelocked
        else:
            if pagelocked:
                print 'ignoring pagelocked on CPU'


class device_handler(object):
    """
    The device_handler class is a convenience class designed around the
    kernel functions implemented in the tvb.cu file.

    """

    #########################################################
    # simulation workspace dimensions
    _dimensions = set(['horizon', 'n_node', 'n_thr', 'n_rthr', 'n_svar', 
                       'n_cvar', 'n_cfpr', 'n_mmpr', 'n_nspr', 'n_inpr', 
                       'n_tavg', 'n_msik', 'n_mode'])

    # generate accessors for global constants
    horizon = device_global('horizon', int32)
    n_node  = device_global('n_node', int32)
    n_thr   = device_global('n_thr', int32)
    n_rthr  = device_global('n_rthr', int32)
    n_svar  = device_global('n_svar', int32)
    n_cvar  = device_global('n_cvar', int32)
    n_cfpr  = device_global('n_cfpr', int32)
    n_mmpr  = device_global('n_mmpr', int32)
    n_nspr  = device_global('n_nspr', int32)
    n_inpr  = device_global('n_inpr', int32)
    n_tavg  = device_global('n_tavg', int32)
    n_msik  = device_global('n_msik', int32)
    n_mode  = device_global('n_mode', int32)


    ##########################################################
    # simulation workspace arrays, also an ORDERED list of the arguments
    # to the update function
    device_state = ['idel', 'cvars', 'inpr', 'conn', 'cfpr', 'nspr', 'mmpr',
        'input', 'x', 'hist', 'dx1', 'dx2', 'gx', 'ns', 'stim', 'tavg']

    # thread invariant, call invariant
    idel  = device_array('idel',    int32, ('n_node', 'n_node'))
    cvars = device_array('cvars',   int32, ('n_cvar', ))
    inpr  = device_array('inpr',  float32, ('n_inpr', ))

    # possibly but not currently thread varying, call invariant
    conn  = device_array('conn',  float32, ('n_node', 'n_node'))

    # thread varying, call invariant
    nspr  = device_array('nspr',  float32, ('n_node', 'n_nspr', 'n_thr'))
    mmpr  = device_array('mmpr',  float32, ('n_node', 'n_mmpr', 'n_thr'))
    cfpr  = device_array('cfpr',  float32, (          'n_cfpr', 'n_thr'))

    # thread varying, call varying
    input = device_array('input', float32, (                     'n_cvar', 'n_thr'))
    x     = device_array('x',     float32, (           'n_node', 'n_svar', 'n_thr'))
    hist  = device_array('hist',  float32, ('horizon', 'n_node', 'n_cvar', 'n_thr'))
    dx1   = device_array('dx1',   float32, (                     'n_svar', 'n_thr'))
    dx2   = device_array('dx2',   float32, (                     'n_svar', 'n_thr'))
    gx    = device_array('gx',    float32, (                     'n_svar', 'n_thr'))
    ns    = device_array('ns',    float32, (           'n_node', 'n_svar', 'n_thr'))
    stim  = device_array('stim',  float32, (           'n_node', 'n_svar', 'n_thr'))
    tavg  = device_array('tavg',  float32, (           'n_node', 'n_svar', 'n_thr'))

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
        
        if using_gpu:
            updatefn = device_code.mod.get_function('update')
        else:
            updatefn = device_code.mod.update
        self._device_update = updatefn

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

        if using_gpu:
            free, total = cuda.mem_get_info()
        elif psutil:
            phymem = psutil.phymem_usage()
            free, total = phymem.free, phymem.total
        else:
            free, total = None, None

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

    @property
    def occupancy(self):
        if using_gpu:
            try:
                return OccupancyRecord(pycuda.tools.DeviceData(), self.n_thr)
            except Exception as exc:
                return exc

    @property
    def extra_args(self):
        if using_gpu:
            bs = int(self.n_thr)%1024
            gs = int(self.n_thr)/1024
            if bs == 0:
                bs = 1024
            if gs == 0:
                gs = 1
            return {'block': (bs, 1, 1),
                    'grid' : (gs, 1)}
        else:
            return {}

    def __call__(self, extra=None, step_type=int32 if using_gpu else ctypes.c_int32):
        args  = [step_type(self.i_step)]
        for k in self.device_state:
            args.append(getattr(self, k).device)
        try:
            kwds = extra or self.extra_args
            self._device_update(*args, **kwds)
        except cuda.LogicError as e:
            print 0, 'i_step', type(args[0])
            for i, k in enumerate(self.device_state):
                attr = getattr(self, k).device
                print i+1, k, type(attr), attr.dtype
            print kwds
            raise e
        self.i_step += self.n_msik
        if using_gpu:
            cuda.Context.synchronize()

