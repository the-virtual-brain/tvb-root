# -*- coding: utf-8 -*-
import os
import numpy
from tvb.basic.neotraits.api import HasTraits, Attr, Float, NArray, List
from tvb.simulator.common import iround
from tvb.simulator.history import BaseHistory, SparseHistory


class CosimUpdate(HasTraits):

    """Base CosimUpdate class"""

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
        """Method to compute the number_of_proxy_nodes, from user defined proxy_inds"""
        self.number_of_proxy_nodes = len(self.proxy_inds)


class CosimStateUpdate(CosimUpdate):

    """Class for updating the current TVB state from co-simulator at each time step."""

    exclusive = Attr(
        field_type=bool,
        default=False, required=False,
        doc="1, when the proxy nodes substitute TVB nodes and their mutual connections should be removed.")

    def configure(self, simulator):
        """Method to compute the number_of_proxy_nodes, from user defined proxy_inds,
           and to set the default indices of the variables to be updated by the cosimulator,
           if user input is not provided."""
        super(CosimStateUpdate, self).configure()
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.variables_of_interest)]

    def update(self, state, update=None):
        """This method will update the input state for
           - state variables of indices voi,
           - and region nodes of indices proxy_inds,
           with update data, if the latter is not None"""
        if update:
            state[self.voi, self.proxy_inds] = update
        return state


class CosimStateUpdateFromFile(CosimStateUpdate):

    """Class for reading data from a file
       and updating the current TVB state from co-simulator at each time step."""

    path = Attr(field_type=os.PathLike, required=False, default="")

    def update_from_file(self):
        """Method to read update data from file."""
        raise NotImplementedError

    def update(self, state, update=None):
        """This method will update the input state for
           - state variables of indices voi,
           - and region nodes of indices proxy_inds,
           with data read from file by update_from_file() method"""
        state[self.voi, self.proxy_inds] = self.update_from_file()
        return state


class CosimStateUpdateFromMPI(CosimStateUpdate):

    """Class for receving data from a MPI port
       and updating the current TVB state from co-simulator at each time step."""

    path = Attr(field_type=os.PathLike, required=False, default="")

    def update_from_mpi(self):
        """Method to receive update data from a MPI port."""
        raise NotImplementedError

    def update(self, state, update=None):
        """This method will update the input state for
           - state variables of indices voi,
           - and region nodes of indices proxy_inds,
           with data received from a MPI port by update_from_mpi() method"""
        state[self.voi, self.proxy_inds] = self.update_from_mpi()
        return state


class CosimHistoryUpdate(CosimUpdate):

    """Class for updating the history of TVB state from co-simulator every period time."""

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
        default=None,  # ms. 0.9765625 => 1024Hz #ms, 0.5 => 2000Hz
        required=False,
        doc="""Sampling period in milliseconds, must be an integral multiple
                    of integration-step size. As a guide: 2048 Hz => 0.48828125 ms ;  
                    1024 Hz => 0.9765625 ms ; 512 Hz => 1.953125 ms.""")

    def configure(self, simulator):
        """Method to
             - compute the number_of_proxy_nodes, from user defined proxy_inds,
             - set the default indices of the variables to be updated by the cosimulator,
               if user input is not provided,
             - set history from simulator.history,
             - set integration time step dt from simulator.integrator.dt
             - set default period of update equal to dt if user has not defined it,
             - compute the istep time steps of update as iround(period / dt),
             - and confirm that istep is not greater than history.n_time."""
        super(CosimHistoryUpdate, self).configure()
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.cvar)]
        self.history = simulator.history
        self.dt = simulator.integrator.dt
        if not self.period:
            self.period = self.dt
        self.istep = iround(self.period / self.dt)
        if self.istep > self.history.n_time:
            raise ValueError("Synchronization time %g for cosimulation update cannot "
                             "be longer than the history buffer time length %g!"
                             % (self.period, self.dt * self.n_time))

    def update(self, step, update=None):
        """This method will update the last istep steps of history for
            - coupling variables of indices voi,
            - and region nodes of indices proxy_inds,
            with update data,
            if the latter is not None, and time step is modulo the period istep."""
        if update and step % self.istep == 0:
            self.history.buffer[(step - self.istep + 1) % self.n_time:step % self.n_time + 1,
                                self.voi,
                                self.proxy_inds] = update


class CosimHistoryUpdateFromFile(CosimHistoryUpdate):

    """Class for reading data from a file and
       updating the history of TVB state from co-simulator every period time."""

    path = Attr(field_type=os.PathLike, required=False, default="")

    def update_from_file(self):
        """Method to read update data from file."""
        raise NotImplementedError

    def update(self, step, update=None):
        """This method will update the last istep steps of history for
            - coupling variables of indices voi,
            - and region nodes of indices proxy_inds,
            with update data
           that are read from file by update_from_file method,
           if time step is modulo the period istep."""
        if step % self.istep == 0:
            self.history.buffer[(step - self.istep + 1) % self.n_time:step % self.n_time + 1,
                                self.voi,
                                self.proxy_inds] = self.update_from_file()


class CosimHistoryUpdateFromMPI(CosimHistoryUpdate):

    """Class for getting data from a MPI port and
       updating the history of TVB state from co-simulator every period time."""

    path = Attr(field_type=os.PathLike, required=False, default="")

    def update_from_mpi(self, data=None):
        """Method to receive update data from a MPI port."""
        raise NotImplementedError

    def update(self, step, update=None):
        """This method will update the last istep steps of history for
            - coupling variables of indices voi,
            - and region nodes of indices proxy_inds,
            with update data
           that are received from a MPI port by update_from_mpi method,
           if time step is modulo the period istep."""
        if step % self.istep == 0:
            self.history.buffer[(step - self.istep + 1) % self.n_time:step % self.n_time + 1,
                                self.voi,
                                self.proxy_inds] = self.update_from_mpi()


class CosimToTVBInterfaces(HasTraits):

    """This class holds lists of
           - CosimStateUpdate,
           - CosimHistoryUpdate,
           class instances"""

    state_interfaces = List(of=CosimStateUpdate)
    history_interfaces = List(of=CosimHistoryUpdate)
