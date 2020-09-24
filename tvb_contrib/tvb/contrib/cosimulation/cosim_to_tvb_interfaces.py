# -*- coding: utf-8 -*-
import os
import numpy
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List
from tvb.simulator.common import iround
from tvb.contrib.cosimulation.history import CosimHistory


class CosimUpdate(HasTraits):

    """Base class to update TVB state or history from data from co-simulator"""

    proxy_inds = NArray(
        dtype=numpy.int,
        label="Indices of TVB proxy nodes",
        required=True)

    number_of_proxy_nodes = Attr(field_type=int,
                                 required=True,
                                 default=0,
                                 label="Number of TVB proxy nodes")

    voi = NArray(
        dtype=int,
        label="Cosimulation model state variables' indices",
        doc=("Indices of model's variables of interest (VOI) that"
             "should be updated (i.e., overwriten) during cosimulation."),
        required=False)

    exclusive = Attr(
        field_type=bool,
        default=False, required=False,
        doc="1, when the proxy nodes substitute TVB nodes and their mutual connections should be removed.")

    def configure(self, simulator):
        """Method to compute the number_of_proxy_nodes, from user defined proxy_inds"""
        self.number_of_proxy_nodes = len(self.proxy_inds)
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.variables_of_interest)]
        super(CosimUpdate, self).configure()


class CosimStateUpdate(CosimUpdate):

    def update_from_cosimulator(self):
        """Method to get update data from co-simulator."""
        raise NotImplementedError

    def update(self, state):
        state[self.voi, self.proxy_inds] = self.update_from_cosimulator()
        return state


class CosimStateUpdateFromFile(CosimUpdate):

    """Class for reading data to update state from a file
       and updating the current TVB state from co-simulator at each time step."""

    path = Attr(field_type=os.PathLike, required=False, default="")

    def update_from_file(self):
        """Method to read update data from file."""
        raise NotImplementedError

    def update(self, state):
        """This method will update the input state for
           - state variables of indices voi,
           - and region nodes of indices proxy_inds,
           with data read from file by update_from_file() method"""
        super(CosimStateUpdateFromFile, self).update(state, *self.update_from_file())


class CosimStateUpdateFromMPI(CosimUpdate):

    """Class for receving data to update state from a MPI port
       and updating the current TVB state from co-simulator at each time step."""

    path = Attr(field_type=os.PathLike, required=False, default="")

    def update_from_mpi(self):
        """Method to receive update data from a MPI port."""
        raise NotImplementedError

    def update(self, state):
        """This method will update the input state for
           - state variables of indices voi,
           - and region nodes of indices proxy_inds,
           with data received from a MPI port by update_from_mpi() method"""
        super(CosimStateUpdateFromMPI, self).update(state, *self.update_from_mpi())


class CosimHistoryUpdate(HasTraits):

    """CosimHistoryUpdate class"""

    history = Attr(
        field_type=CosimHistory,
        label="Cosimulation history",
        default=None,
        required=True,
        doc="""A history of the whole TVB state for the cosimulation synchronization time.""")

    def configure(self, simulator):
        """Method to compute the number_of_proxy_nodes, from user defined proxy_inds"""
        self.history = simulator.history
        super(CosimHistoryUpdate, self).configure()

    def update_from_cosimulator(self):
        """Method to get update data from co-simulator."""
        pass

    def update(self, times=None, new_states=None):
        if times is None or new_states is None:
            times, new_states = self.update_from_cosimulator()
        # First convert times into simulation steps and then update history
        # for steps, vois and proxy_inds:
        self.history.update_from_cosim([iround(time / self.dt) for time in times],
                                       new_states,
                                       self.voi, self.proxy_inds)


class CosimHistoryUpdateFromFile(CosimUpdate):

    """Class for reading data to update history from a file
       and updating the current TVB state from co-simulator at each time step."""

    path = Attr(field_type=os.PathLike, required=False, default="")

    def update_from_file(self):
        """Method to read update data from file."""
        raise NotImplementedError

    def update(self, state, new_state=None):
        """This method will update the input state for
           - state variables of indices voi,
           - and region nodes of indices proxy_inds,
           with data read from file by update_from_file() method"""
        super(CosimHistoryUpdateFromFile, self).update(state, *self.update_from_file())


class CosimUpdateFromMPI(CosimUpdate):

    """Class for receving data to update history from a MPI port
       and updating the current TVB state from co-simulator at each time step."""

    path = Attr(field_type=os.PathLike, required=False, default="")

    def update_from_mpi(self):
        """Method to receive update data from a MPI port."""
        raise NotImplementedError

    def update(self, state, new_state=None):
        """This method will update the input state for
           - state variables of indices voi,
           - and region nodes of indices proxy_inds,
           with data received from a MPI port by update_from_mpi() method"""
        super(CosimUpdateFromMPI, self).update(state, *self.update_from_mpi())


class CosimToTVBInterfaces(HasTraits):

    """This class holds lists of CosimStateUpdate and CosimHistoryUpdate class instances"""

    state_interfaces = List(of=CosimStateUpdate)
    history_interfaces = List(of=CosimHistoryUpdate)

    @property
    def interfaces(self):
        return self.state_interfaces + self.history_interfaces

    @property
    def state_vois(self):
        return [state_interface.voi for state_interface in self.state_interfaces]

    @property
    def state_proxy_inds(self):
        return [state_interface.proxy_inds for state_interface in self.state_interfaces]

    @property
    def history_vois(self):
        return [history_interface.voi for history_interface in self.history_interfaces]

    @property
    def history_proxy_inds(self):
        return [history_interface.proxy_inds for history_interface in self.history_interfaces]

    @property
    def vois(self):
        return self.state_vois + self.history_vois

    @property
    def proxy_inds(self):
        return self.state_proxy_inds + self.history_proxy_inds

    def configure(self, simulator):
        for interfaces in self.interfacess:
            interfaces.configure(simulator)
        super(CosimToTVBInterfaces, self).configure()
