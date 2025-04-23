# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2024, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
This modules defines a set of classes for hybrid-model simulation.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import collections
import numpy as np
from typing import List
import tvb.basic.neotraits.api as t
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models import Model
from tvb.simulator.integrators import Integrator
from tvb.simulator.monitors import Monitor


# XXX this should probably be built into our normal
# monitor implementation: store stuff into list or directly to disk.
# Teh framework supports this but not sure how easy to use from notebook.
class Recorder(t.HasTraits):
    monitor: Monitor = t.Attr(Monitor)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.times = []
        self.samples = []

    def record(self, step, state):
        ty = self.monitor.record(step, state)
        if ty is not None:
            t, y = ty
            self.times.append(t)
            self.samples.append(y)

    @property
    def shape(self):
        return (len(self.samples), ) + self.samples[0].shape

    def to_arrays(self):
        return np.array(self.times), np.array(self.samples)


class Subnetwork(t.HasTraits):
    "Represents a subnetwork that can be reused across different models."

    name: str = t.Attr(str)
    model: Model = t.Attr(Model)
    scheme: Integrator = t.Attr(Integrator)
    monitors: List[Recorder]
    nnodes: int = t.Int()

    def __init__(self, **kwargs):
        self.monitors = []
        super().__init__(**kwargs)

    def configure(self):
        self.model.configure()
        return self

    def add_monitor(self, monitor: Monitor):
        monitor._config_dt(self.scheme.dt)
        monitor._config_stock(len(self.model.variables_of_interest),
                            self.nnodes,
                            self.model.number_of_modes)
        self.monitors.append(Recorder(monitor=monitor))

    @property
    def var_shape(self) -> tuple[int]:
        return self.nnodes, self.model.number_of_modes

    def zero_states(self) -> np.ndarray:
        return np.zeros((self.model.nvar, ) + self.var_shape)

    def zero_cvars(self) -> np.ndarray:
        return np.zeros((self.model.cvar.size, ) + self.var_shape)

    def step(self, step, x, c):
        nx = self.scheme.scheme(x, self.model.dfun, c, 0, 0)
        for monitor in self.monitors:
            monitor.record(step, self.model.observe(nx))
        return nx


class Stim(Subnetwork):
    "Stimulator adapted for hybrid cases"
    # classic use is non-modal:
    # stimulus[self.model.stvar, :, :] = \
    #   self.stimulus(stim_step).reshape((1, -1, 1))
    pass


class Projection(t.HasTraits):
    """Represents a projection from a source subnetwork to a target
    subnetwork, specifying indices of coupling variables."""

    source: Subnetwork = t.Attr(Subnetwork)
    target: Subnetwork = t.Attr(Subnetwork)
    source_cvar: int = t.Int()
    target_cvar: int = t.Int()
    scale: float = t.Float(default=1.0)
    weights: np.ndarray = t.NArray(dtype=np.float_)
    mode_map: np.ndarray = t.NArray(required=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.mode_map is None:
            self.mode_map = np.ones(
                (self.source.model.number_of_modes,
                 self.target.model.number_of_modes, ))
            self.mode_map /= self.source.model.number_of_modes

    def apply(self, tgt, src):
        gx = self.scale * self.weights @ src[self.source_cvar] @ self.mode_map
        tgt[self.target_cvar] += gx


class NetworkSet(t.HasTraits):
    """Collects subnetworks & projections, with methods for evaluating
    full network coupling and time stepping.
    """
    subnets: [Subnetwork] = t.List(of=Subnetwork)
    projections: [Subnetwork] = t.List(of=Projection)

    # NOTE dynamically generated namedtuple based on subnetworks
    States: collections.namedtuple = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.States = collections.namedtuple(
            'States',
            ' '.join([_.name for _ in self.subnets]))
        self.States.shape = property(lambda self: [_.shape for _ in self])

    def zero_states(self) -> States:
        return self.States(*[_.zero_states() for _ in self.subnets])

    def zero_cvars(self) -> States:
        return self.States(*[_.zero_cvars() for _ in self.subnets])

    def observe(self, states: States, flat=False) -> np.ndarray:
        "Compute observations (variables of interest) across all subnets."
        obs = self.States(*[sn.model.observe(x).sum(axis=-1)[..., None]
                          for sn, x in zip(self.subnets, states)])
        if flat:
            obs = np.hstack(obs)
        return obs

    def cfun(self, eff: States) -> States:
        "applies all projections to compute total coupling."
        aff = self.zero_cvars()
        for p in self.projections:
            tgt = getattr(aff, p.target.name)
            src = getattr(eff, p.source.name)
            p.apply(tgt, src)
        return aff

    def step(self, step, xs: States) -> States:
        cs = self.cfun(xs)
        nxs = self.zero_states()
        for sn, nx, x, c in zip(self.subnets, nxs, xs, cs):
            nx[:] = sn.step(step, x, c)
        return nxs


class Simulator(t.HasTraits):
    nets: NetworkSet = t.Attr(NetworkSet)
    monitors: List[Monitor] = t.List(of=Monitor)
    simulation_length: float = t.Float()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validate_dts()
        self.validate_vois()

    def validate_vois(self):
        if len(self.monitors) == 0:
            return
        nv0 = self.nets.subnets[0].model.variables_of_interest
        for sn in self.nets.subnets[1:]:
            msg = 'Variables of interest must have same size on all models.'
            assert len(nv0) == len(sn.model.variables_of_interest), msg
        for monitor in self.monitors:
            num_nodes = sum([sn.nnodes for sn in self.nets.subnets])
            monitor._config_stock(len(nv0), num_nodes, 1)

    def validate_dts(self):
        self._dt0 = self.nets.subnets[0].scheme.dt
        for sn in self.nets.subnets[1:]:
            assert self._dt0 == sn.scheme.dt
        for monitor in self.monitors:
            monitor: Monitor
            monitor._config_dt(self._dt0)
            monitor.voi = slice(None)  # all vars

    def run(self):
        x = self.nets.zero_states()
        mts = [[] for _ in self.monitors]
        mxs = [[] for _ in self.monitors]
        for step in range(int(self.simulation_length / self._dt0)):
            x = self.nets.step(step, x)
            if self.monitors:
                ox = self.nets.observe(x, flat=True)
                for mt, mx, mon in zip(mts, mxs, self.monitors):
                    maybe_tx = mon.record(step, ox)
                    if maybe_tx is not None:
                        mt.append(maybe_tx[0])
                        mx.append(maybe_tx[1])
        return [(np.array(t), np.array(x)) for t, x in zip(mts, mxs)]
