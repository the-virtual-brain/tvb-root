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
This module defines utilities and a main class for running parameter sweeps.

Parameter sweeps on stream computing hardware
---------------------------------------------

Here we are *mainly* concerned with mapping parameter space explorations
onto the multithread architecture of recent hardware like graphics cards.
A significant constraint is that the groups of threads must act identically,
which means that while numerical parameters may change, the memory access
patterns and code across threads cannot vary. 

In TVB, this mainly means that certain parameters are streamable i.e. suitable
for variation in a thread group, e.g.

- mass model parameters
- noise & noise parameters
- connectivity matrix (not currently supported, but possible)
- stimulation pattern

while others are not, e.g.

- integrator parameters
- conduction velocity
- model dynamics
- etc.

.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

import time
import itertools
import functools
import collections

from tvb.simulator import lab as l


LOG = l.get_logger(__name__)

def logged(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwds):
        #LOG.info('%r called with %r, %r', fn, args, kwds)
        ret = fn(*args, **kwds)
        #LOG.info('%r returned %r', fn, ret)
        return ret
    return wrapper

# path rel to self. where self is the simulation. eg. integrator.dt
Par = collections.namedtuple('Par', 'path, streamable, values')

class par_sweep(object):
    """
    The par_sweep class manages parameter variations and their mapping
    to simulations and if necessary, device handler.


    Rationale
    ---------

    __init__
    1. we need a list of parameters to be varied, pars
    2. reorder pars so that streamable parameters at end
    3. generate all pars configurations (itertools.product(*pars))

    next()
    4. while simulations remain to be run
        1. pop first parameter configuration
        2. generate simulator based on config
        3. build device handler based on simulation
        4. decide num threads to run on device based on memory available -> n_thr
        5. pop next n_thr-1 par configurations that do not vary a non streamable parameter
        6. build simulations on rest of configurations
        7. pack all their data into device handler
        8. build multithreaded monitors as necessary
        9. while not end of simulation
            1. update sim
            2. record time averaged data
            3. push data to monitors as necessary
        10. yield results of this simulation batch

    """

    @logged
    def __init__(self, *pars, **kwds):
        """
        par_sweep.__init__ takes a sequence of positional arguments to be the
        set of parameters to vary, and keywords to be the default, baseline
        simulator components that do *not* vary. The baseline configured
        simulator object will be updated for a given parameter configuration
        so it's necessary to provide baseline configuration that can 
        successfully initialize the simulator.

        """


        self.kwds = kwds
        if 'stimulus' in kwds:
            raise NotImplementedError('it will take 5 minutes, plz send patch')
        self.sim = simulator.Simulator(**kwds)

        # does it stream?
        yes, no = [], []
        for p in pars:
            p = Par(*p) if len(p)==3 else Par(p[0], False, p[1])
            (yes if p.streamable else no).append(p)
        self.pars = no + yes

        for i_par, par in enumerate(self.pars):
            LOG.info('%r par[%d] -> %r, streamable=%r', 
                     self, i_par, par.path, par.streamable)

        # axis starting with which pars can be streamed
        self.part = len(no)

        # make ndarray of configurations
        ndconf_shape = [len(p.values) for p in self.pars]
        self.ndconf = empty(ndconf_shape, dtype=object)
        for i, conf in enumerate(itertools.product(*(p.values for p in self.pars))):
            self.ndconf.flat[i] = conf

        self.split_shape = (int(prod(ndconf_shape[:self.part])),
                            int(prod(ndconf_shape[self.part:])))
        self.ndconf_split = self.ndconf.reshape(self.split_shape)
        self.split_idx = (0, 0)

        LOG.info('%r, split_shape %r', self, self.split_shape)

        self.n_msik = 1

        if 'drv' in kwds:
            self.drv = kwds['drv']
        else:
            LOG.info('initializing backend driver, may take a moment...')
            from tvb.simulator.backend import driver as drv
            self.drv = drv

    @logged
    def __iter__(self):
        self.results = []
        return self

    @logged
    def next_sim(self):
        """
        The next_sim method increments the indices for non-streamable and
        streamable parameters as necessary, checks if we've finished and 
        otherwise creates another simulator based on the next configuration
        of parameters in the sweep, and returns this simulation.

        """

        r, c = self.split_idx

        if r == self.split_shape[0]:
            raise StopIteration

        sim = self.build_sim(self.ndconf_split[r, c])

        if c == self.split_shape[1]-1:
            self.split_idx = (r+1, 0)
        else:
            self.split_idx = (r, c+1)

        return sim

    @logged
    def next(self):
        """
        The next method implements part of the Python iterator protocol. When
        the caller loops over this object, the next method first checks if 
        there are any simulation results to return and does so. Then, the
        next simulation or batch of simulations is built and run, generating
        more results. This continues until the parameter sweep has been
        completed, resulting in a StopIteration exception to be raised (by the
        next_sim method), indicating that everything is done

        """

        if self.results:
            return self.results.pop()

        # TODO add a 'numpy mode' so that we just use the simulator itself
        # and return the "correct" results

        sim = self.next_sim()
        sim.current_step = 0
        dh = self.drv.device_handler.init_like(sim, self.n_msik)
        maxthr = self.drv.total_mem/dh.nbytes1
        
        # how many threads to run?
        if using_gpu:
            block = dh.block_dim
            todo = self.split_shape[1] - self.split_idx[1]
            todo += 1 # becuase we want to run the one we've taken already
            grid = min(todo/block, maxthr/block)
            if grid < maxthr/block and todo - grid*block > 0:
                grid += 1 # padded
            n_thr = grid*block
            n_rthr = min(todo, n_thr)
            msg = "%r using_gpu: grid %d block %d n_thr %d"
            LOG.info(msg, self, grid, block, n_thr)
        else:
            n_thr = n_rthr = 1 # unless using_omp?
            grid, block = None, None

        dh.n_thr = n_thr
        dh.n_rthr = n_rthr
        dh.fill_with(0, sim)
        for i in range(n_rthr-1):
            dh.fill_with(i+1, self.next_sim())

        self.setup_monitors(sim, dh, n_rthr)

        launch_config = {'block': (block, 1, 1), 
                         'grid': (grid, 1, 1)} if using_gpu else {}

        tic = time()
        self.results = []
        t = lambda : sim.current_step*sim.integrator.dt
        print 'sim.simulation_length', sim.simulation_length
        while t() < sim.simulation_length:
            self.drv.gen_noise_into(dh.ns)
            #FIXME dh.stim[:] = sim.stimulus[sim.current_step] or something
            dh(launch_config)
            for i_out, out in enumerate(self.pump_monitors(sim.current_step, dh)):
                if out is not None:
                    mon_step = self.monitor_steps[i_out]
                    self.monitor_output[i_out][mon_step, ...] = out.copy()
                    self.monitor_steps[i_out] += 1
            sim.current_step += dh.n_msik
        print 'that took %0.2f s', time() - tic



        # Now, we reorganize results list which initially looks like
        #
        #     [mon][time, svar, node, mode, thr]
        #
        # so that it looks like
        #
        #     [thr][monitor][time, svar, node, mode]


        LOG.info('reorganizing results...')
        per_thread = [[out[..., i_thr] for out in self.monitor_output]
                                       for i_thr in xrange(n_rthr)]
        self.results = per_thread
        LOG.info('reorganizing done')

    @logged
    def setup_monitors(self, sim, dh, n_rthr):
        """
        setup monitors will first check that the monitor is supported and
        will intialize the monitor as necessary

        In fact, for the moment we support

        - temporal average

            This is performed in device (though technically parsweep
            class shouldn't care..)

        - parallel bold

            This is a subclass of bold with the same HRF but can handle
            many parellel simulations at once.        

        Eventually, all monitors will be updated to handle parallel
        simulations, in which case parsweep simply needs to handle one of 
        the parallel monitors each, unless someone decides to vary a monitor
        parameter then they will have some lovely work to do.

        Ah but maybe since there's a base line monitor, and per thread 
        batch, we don't vary monitor, and assume monitors handle multiple
        threads, we cna just consider monitor variations as NON streamable,
        and then we can use monitors as the simulator does.

        """

        self.monitors       = []
        self.monitor_output = []
        self.monitor_steps  = []
        for mon in self.sim.monitors:
            if type(mon) not in (monitors.TemporalAverage, monitors.BoldMultithreaded):
                msg = "%r: ignoring unsupported monitor %r"
                LOG.warning(msg, self, mon)
                continue

            # allocate space for output
            n_out = int(sim.simulation_length / mon.period)
            out = empty((n_out, len(mon.voi), dh.n_node, dh.n_mode, n_rthr), dtype=numpy.float32)

            if isinstance(mon, l.monitors.TemporalAverage):
                self.tavg_istep = int(mon.period / self.sim.integrator.dt)
            else: # is bold
                mon.config_for_sim(sim, n_thr=n_rthr)

            msg = '%r requires output storage with shape %r and size %d MB'
            LOG.info(msg, mon, out.shape, out.nbytes >> 20)

            self.monitors       .append(mon)
            self.monitor_output .append(out)
            self.monitor_steps  .append(0)

        
    @logged
    def pump_monitors(self, step, dh):
        """
        The monitors methods takes a device_handler instance as an argument
        and pulls the relevant state from the device arrays as necessary. 
        It then updates the current monitors with this state information
        and returns whatever output is given back by the monitors. 

        We have a few array tricks to play here when passing data to 
        other monitors

        - unfold modes from state variables
        - put svar axis before node axis

        """

        out = []
        for mon in self.monitors:

            is_tavg = isinstance(mon, monitors.TemporalAverage)

            ary = 'tavg' if is_tavg else 'x'
            dev = 'device.get()' if using_gpu else 'cpu.copy()'

            unfolded_modes  = (dh.n_node, dh.n_svar, dh.n_mode, dh.n_thr)

            ys = eval('dh.%s.%s' % (ary, dev))
            ys = rollaxis(ys.reshape(unfolded_modes), 1)

            if using_gpu:
                ys = ys[..., :dh.n_rthr] # strip padding threads

            if is_tavg:
                mys = ys[mon.voi] if step % self.tavg_istep == 0 else None
            else:
                maybe_mys = mon.record(step, ys)
                mys = maybe_mys[1] if maybe_mys else None

            out.append(mys)

        return out


    @logged
    def build_sim(self, config):
        """
        The build_sim method is responsible for taking a set of parameters
        defined as part of the parameter sweep, combining those parameters
        with the default parameters given as keywords to initialize the
        sweep and to initialize a Simulator instance with this information.

        """

        # update with parameter configuration
        for (path, _, _), val in zip(self.pars, config):
            code = "sim.%s = val" % (path, )
            exec code in {'sim': self.sim, 'val': val}

        self.sim.configure()
        return self.sim



if __name__ == '__main__':

    from guppy import hpy
    __h__ = hpy()

    from tvb.simulator.backend import driver_conf
    driver_conf.using_gpu = using_gpu = 1

    from tvb.simulator.lab import *

    # this is test driven development speaking how i can help you

    model = models.Generic2dOscillator()
    conn = connectivity.Connectivity(load_default=True, speed=array([4.0]))
    coupling = coupling.Linear(a=0.0152)

    hiss = noise.Additive(nsig=ones((2,)) * 2 ** -10)
    heun = integrators.EulerStochastic(dt=2 ** -4, noise=hiss)

    mon = (monitors.BoldMultithreaded(period=500.0),
           monitors.TemporalAverage(period=5.0))

    sweep = par_sweep(('coupling.a', True, r_[0:1e-5:64j]),
                      model=model,
                      connectivity=conn,
                      coupling=coupling,
                      integrator=heun,
                      monitors=mon,
                      simulation_length=1000)

    sweep.n_msik=1 
    # FIXME else we skip monitors..
    # FIXME and noise/stim array is reused

    for i_config, data in enumerate(sweep):
        print i_config


    print '__h__ is the guppy memory profiler'
