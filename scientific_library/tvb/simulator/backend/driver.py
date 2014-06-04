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
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
tvb.simulator.backend.driver
============================

This modules defines several classes that are informally base
or abstract classes for various actual backends. Backend specific
code is provided by subclassing the classes in this module, and in
most cases, the classes here don't function at all, so the user 
should e.g. import backend.cee directly.

class Code -       Setup and load template code on device
class Global -     Setup & access global variables in code
class Array -      Alloc, free, set, get main data storage
class Handler -    Coordinate code & data on device

"""

import sys
import os.path
import string
import glob
import subprocess
from numpy import *
from tvb.simulator.lab import *

def cpp(filename):
    proc = subprocess.Popen(['cpp', filename], stdout=subprocess.PIPE)
    return proc.stdout.read()

class Code(object):
    here = os.path.dirname(os.path.abspath(__file__)) + os.path.sep 
    def sources(self):
        srcs = {}
        for name in glob.glob(self.here + '*.cu'):
            key = os.path.basename(name)
            with open(name, 'r') as fd:
                srcs[key] = fd.read()
        return srcs

    def __init__(self, fns=[], T=string.Template, **kwds):
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
        template argument is used, the definition of the macros, etc.

        """

        args = dict()
        for k in self.defaults.keys():
            args[k] = kwds.get(k, self.defaults[k])

        self.source = T(self.sources()['tvb.cu']).substitute(**args) 
        with open('temp.cu', 'w') as fd:
            fd.write(self.source)

        self.fns = fns

    # default template filler
    defaults = {
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

class Global(object):
    """
    Encapsulates a source module device global in a Python data descriptor
    for easy handling

    """

    def __init__(self, name, dtype, code):
        self.code  = code
        self.name  = name
        self.dtype = dtype
        self.__post_init = True
       

class Array(object):
    """
    Encapsulates an array that is on the device

    """

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

    def __init__(self, name, type, dimensions):
        self.parent = None
        self.name = name
        self.type = type
        self.dimensions = dimensions


class Handler(object):
    """
    The device_handler class is a convenience class designed around the
    kernel functions implemented in the tvb.cu file.

    """

    #########################################################
    # simulation workspace dimensions
    _dimensions = set(['horizon', 'n_node', 'n_thr', 'n_rthr', 'n_svar', 
                       'n_cvar', 'n_cfpr', 'n_mmpr', 'n_nspr', 'n_inpr', 
                       'n_tavg', 'n_msik', 'n_mode'])

    def init_globals(self, Global=Global):
        "generate accessors for global constants"
        for const in ['horizon', 'n_node', 'n_thr' , 'n_rthr', 'n_svar',
                'n_cvar', 'n_cfpr', 'n_mmpr', 'n_nspr', 'n_inpr', 'n_tavg',
                'n_msik', 'n_mode']:
            setattr(self, const, Global(const, int32, self.code))

    ##########################################################
    # simulation workspace arrays, also an ORDERED list of the arguments
    # to the update function
    device_state = ['idel', 'cvars', 'inpr', 'conn', 'cfpr', 'nspr', 'mmpr',
        'input', 'x', 'hist', 'dx1', 'dx2', 'gx', 'ns', 'stim', 'tavg']

    def init_arrays(self, Array=Array):
        "initialize arrays"
        # thread invariant, call invariant
        self.idel  = Array('idel',    int32, ('n_node', 'n_node'))
        self.cvars = Array('cvars',   int32, ('n_cvar', ))
        self.inpr  = Array('inpr',  float32, ('n_inpr', ))

        # possibly but not currently thread varying, call invariant
        self.conn  = Array('conn',  float32, ('n_node', 'n_node'))
        self.cfpr  = Array('cfpr',  float32, ('n_cfpr', ))

        # thread varying, call invariant
        self.nspr  = Array('nspr',  float32, ('n_node', 'n_nspr', 'n_thr'))
        self.mmpr  = Array('mmpr',  float32, ('n_node', 'n_mmpr', 'n_thr'))

        # thread varying, call varying
        self.input = Array('input', float32, (                     'n_cvar', 'n_thr'))
        self.x     = Array('x',     float32, (           'n_node', 'n_svar', 'n_thr'))
        self.hist  = Array('hist',  float32, ('horizon', 'n_node', 'n_cvar', 'n_thr'))
        self.dx1   = Array('dx1',   float32, (                     'n_svar', 'n_thr'))
        self.dx2   = Array('dx2',   float32, (                     'n_svar', 'n_thr'))
        self.gx    = Array('gx',    float32, (                     'n_svar', 'n_thr'))
        self.ns    = Array('ns',    float32, (           'n_node', 'n_svar', 'n_thr'))
        self.stim  = Array('stim',  float32, (           'n_node', 'n_svar', 'n_thr'))
        self.tavg  = Array('tavg',  float32, (           'n_node', 'n_svar', 'n_thr'))

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
            self.cfpr  .cpu[:] = sim.coupling.device_info.cfpr

        # each device_info should know the required shape, so none of these
        # assignments should fail with a shape error
        if hasattr(sim.integrator, 'noise'):
            self.nspr  .cpu[..., idx] = sim.integrator.noise.device_info.nspr

        # mass model parameters
        self.mmpr  .cpu[..., idx] = sim.model.device_info.mmpr

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

    def __init__(self, code_args={}, Code=Code, Global=Global, Array=Array, **kwds):

        fns = ['update'] + [f+d for d in self._dimensions 
                                for f in ['set_', 'get_']]

        self.code = Code(fns, **code_args)
        self.init_globals(Global)
        self.init_arrays(Array)

        for k in self._dimensions:
            if k in kwds:
                setattr(self, k, kwds.get(k))
            else:
                missing = self._dimensions - set(kwds.keys())
                if missing:
                    msg = 'Handler requires the keyword value %r' % missing
                    raise TypeError(msg)

        for k in self.device_state:
            getattr(self, k).parent = self

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
        code = {'model_dfun': sim.model           .device_info.kernel,
                'integrate':  sim.integrator      .device_info.kernel,
                'noise_gfun': sim.integrator.noise.device_info.kernel if stoch else "",
                'coupling':   sim.coupling        .device_info.kernel}

        dh = cls(code_args=code, **dims)
        return dh


    @property
    def nbytes(self):
        memuse = sum([getattr(self, k).nbytes for k in self.device_state])
        free, total = self.mem_info
        if free and total: # are not None
            if memuse > free:
                print '%r: nbytes=%d exceeds free memory' % (self, memuse)
            if memuse > total:
                raise MemoryError('%r: nbytes=%d exceeds total memory' % (self, memuse))
        return memuse

    @property
    def nbytes1(self):
        old_n_thr = self.n_thr
        self.n_thr = 1
        nbs = self.nbytes
        self.n_thr = old_n_thr 
        return nbs


    def __call__(self, extra=None):
        args  = [self.i_step_type(self.i_step)]
        for k in self.device_state:
            args.append(getattr(self, k).device)
        kwds = extra or self.extra_args
        self._device_update(*args, **kwds)
        self.i_step += self.n_msik


