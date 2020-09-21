# -*- coding: utf-8 -*-
import os
import numpy
from tvb.basic.neotraits.api import HasTraits, Attr, Float, NArray, List
from tvb.simulator.common import iround
from tvb.simulator.history import BaseHistory, SparseHistory


class CosimUpdate(HasTraits):

    proxy_inds = NArray(
        dtype=numpy.int,
        label="Indices of TVB proxy nodes",
        required=True,
    )

    number_of_proxy_nodes = Attr(field_type=int, required=True,
                                 default=0)

    voi = NArray(
        dtype=int,
        label="Cosimulation model state variables' indices",
        doc=("Indices of model's variables of interest (VOI) that"
             "should be updated (i.e., overwriten) during cosimulation."),
        required=False)

    def configure(self, simulator):
        self.number_of_proxy_nodes = len(self.proxy_inds)


class CosimStateUpdate(CosimUpdate):

    exclusive = Attr(
        field_type=bool,
        default=False, required=False,
        doc="1, when the proxy nodes substitute TVB nodes and their mutual connections should be removed.")

    def configure(self, simulator):
        super(CosimStateUpdate, self).configure()
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.variables_of_interest)]

    def _update(self, data):
        return data

    def update(self, state, update=None):
        state[self.voi, self.proxy_inds] = self._update(update)
        return state


class CosimStateUpdateFromFile(CosimStateUpdate):

    path = Attr(field_type=os.PathLike, required=False, default="")

    def _update(self, data=None):
        raise NotImplementedError


class CosimStateUpdateFromMPI(CosimStateUpdate):

    path = Attr(field_type=os.PathLike, required=False, default="")

    def _update(self, data=None):
        raise NotImplementedError


class CosimHistoryUpdate(CosimUpdate):

    history = Attr(
        field_type=BaseHistory,
        label="Simulator history",
        default=SparseHistory(),
        required=True,
        doc="""A tvb.simulator.history""")

    period = Float(
        label="Updating period (ms)",
        required=False,
        default=0.9765625,  # ms. 0.9765625 => 1024Hz #ms, 0.5 => 2000Hz
        doc="""Updating period in milliseconds, must be an integral multiple
                    of integration-step size. As a guide: 2048 Hz => 0.48828125 ms ;  
                    1024 Hz => 0.9765625 ms ; 512 Hz => 1.953125 ms.""")

    dt = Float(
        label="Integration step (ms)",  # order = 10
        default=0.9765625,  # ms. 0.9765625 => 1024Hz #ms, 0.5 => 2000Hz
        required=False,
        doc="""Sampling period in milliseconds, must be an integral multiple
                    of integration-step size. As a guide: 2048 Hz => 0.48828125 ms ;  
                    1024 Hz => 0.9765625 ms ; 512 Hz => 1.953125 ms.""")

    def configure(self, simulator):
        super(CosimHistoryUpdate, self).configure()
        self.history = simulator.history
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.cvar)]
        self.dt = simulator.integrator.dt
        self.istep = iround(self.period / self.dt)
        if self.istep > self.history.n_time:
            raise ValueError("Synchronization time %g for cosimulation update cannot "
                             "be longer than the history buffer time length %g!"
                             % (self.period, self.dt * self.n_time))

    def update(self, step, update=None):
        if step % self.istep == 0:
            self.history.buffer[(step - self.istep + 1) % self.n_time:step % self.n_time + 1,
                                self.voi,
                                self.proxy_inds] = self._update(update)


class CosimHistoryUpdateFromFile(CosimHistoryUpdate):

    path = Attr(field_type=os.PathLike, required=False, default="")

    def _update(self, data=None):
        raise NotImplementedError


class CosimHistoryUpdateFromMPI(CosimHistoryUpdate):

    path = Attr(field_type=os.PathLike, required=False, default="")

    def _update(self, data=None):
        raise NotImplementedError


class CosimToTVBInterfaces(HasTraits):

    state_interfaces = List(of=CosimStateUpdate)
    history_interfaces = List(of=CosimHistoryUpdate)
