# -*- coding: utf-8 -*-
import os
import numpy
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List
from tvb.contrib.cosimulation.history import CosimHistory


class CosimUpdate(HasTraits):

    """Base class to update TVB state or cosimulation history from data from co-simulator"""

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
        """Method to ccnfigure CosimUpdate
           the variables of interest,
           and the number of proxy nodes"""
        self.number_of_proxy_nodes = len(self.proxy_inds)
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.variables_of_interest)]
        super(CosimUpdate, self).configure()

    def update_from_cosimulator(self):
        """Method to get update data from co-simulator."""
        pass


class CosimStateUpdate(CosimUpdate):

    """Class to update the current TVB state from data from co-simulator"""

    def update_from_cosimulator(self):
        """Method to get update data from co-simulator."""
        raise NotImplementedError

    def update(self, state):
        state[self.voi, self.proxy_inds] = self.update_from_cosimulator()
        return state


class CosimStateUpdateFromFile(CosimStateUpdate):

    """Class to update the current TVB state from a file,
       by implementing the update_from_cosimulator() method accordingly."""

    path = Attr(field_type=os.PathLike, required=False, default="")

    def update_from_cosimulator(self):
        """Method to get update data from co-simulator."""
        raise NotImplementedError


class CosimStateUpdateFromMPI(CosimStateUpdate):

    """"Class to update the current TVB state from a MPI port,
        by implementing the update_from_cosimulator() method accordingly."""

    def update_from_cosimulator(self):
        """Method to get update data from co-simulator."""
        raise NotImplementedError


class CosimHistoryUpdate(CosimUpdate):

    """Class to update the cosimulation history from data from co-simulator"""

    cosim_history = Attr(
        field_type=CosimHistory,
        label="Cosimulation history",
        default=None,
        required=True,
        doc="""A history of the whole TVB state and coupling for the cosimulation synchronization time.""")

    def configure(self, simulator):
        """Method to ccnfigure CosimHistoryUpdate
        with the TVB's cosimulation history,
        the variables of interest,
        and the number of proxy nodes"""
        self.cosim_history = simulator.cosim_history
        super(CosimHistoryUpdate, self).configure()

    def update(self, steps, new_states=None):
        """This method will update the input state  for the given time steps and for
           - state variables of indices voi,
           - and region nodes of indices proxy_inds,
           with data optionally read from the update_from_cosimulator() method"""
        if new_states is None:
            new_states = self.update_from_cosimulator()
        self.cosim_history.update_state_from_cosim(steps, new_states, self.voi, self.proxy_inds)


class CosimHistoryUpdateFromFile(CosimHistoryUpdate):

    """Class for reading data to update cosimulation history from a file,
       by implementing the update_from_cosimulator() method accordingly."""

    path = Attr(field_type=os.PathLike, required=False, default="")

    def update_from_cosimulator(self):
        """Method to get update data from co-simulator."""
        raise NotImplementedError


class CosimHistoryUpdateFromMPI(CosimHistoryUpdate):

    """Class for reading data to update cosimulation history from a MPI port,
       by implementing the update_from_cosimulator() method accordingly."""

    def update_from_cosimulator(self):
        """Method to get update data from co-simulator."""
        raise NotImplementedError


class CosimToTVBInterfaces(HasTraits):

    """This class holds lists of CosimStateUpdate and CosimHistoryUpdate class instances"""

    state_interfaces = List(of=CosimStateUpdate)
    history_interfaces = List(of=CosimHistoryUpdate)

    update_variables_fun = None

    number_of_state_interfaces = None
    number_of_history_interfaces = None

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
        self.number_of_state_interfaces = len(self.state_interfaces)
        self.number_of_history_interfaces = len(self.history_interfaces)
        super(CosimToTVBInterfaces, self).configure()
