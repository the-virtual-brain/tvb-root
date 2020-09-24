# -*- coding: utf-8 -*-
from tvb.basic.neotraits.api import HasTraits, List

# -*- coding: utf-8 -*-
import os
import numpy
from tvb.basic.neotraits.api import Attr, NArray
from tvb.contrib.cosimulation.monitors import CosimMonitor, CouplingMonitor


class TVBtoCosimInterface(HasTraits):

    """TVBtoCosimInterface base class holding a CosimMonitor class instance."""

    proxy_inds = NArray(
        dtype=numpy.int,
        label="Indices of TVB proxy nodes",
        required=True,
    )

    number_of_proxy_nodes = Attr(field_type=int,
                                 required=True,
                                 default=0,
                                 label="Number of TVB proxy nodes")

    monitor = Attr(field_type=CosimMonitor,
                   required=True,
                   default=0,
                   label="Number of TVB proxy nodes")

    def configure(self, simulator):
        """Method to configure the CosimMonitor of the interface
           and compute the number_of_proxy_nodes, from user defined proxy_inds"""
        self.monitor.config_for_sim(simulator)
        self.number_of_proxy_nodes = len(self.proxy_inds)
        super(TVBtoCosimInterfaces).configure()

    def record(self, step, state):
        """Record from the CosimMonitor of the interface and return the data only of proxy_inds"""
        output = self.monitor.record(self, step, state)
        if output is not None:
            output[1] = output[1][:, :, self.proxy_inds]
            return output


class TVBtoCosimStateInterface(TVBtoCosimInterface):
    """ TVBtoCosimStateInterface class:
        Monitors the value for the model's variable/s of interest over all
        the nodes at each sampling period. Time steps that are not modulo ``istep``
        are stored temporarily in the ``_stock`` attribute and then that temporary
        store is returned when time step is modulo ``istep``.

    """
    pass


class TVBtoCosimStateInterfaceToFile(TVBtoCosimStateInterface):
    """ CosimStateMonitorToFile class:
            Monitors the value for the model's variable/s of interest over all
            the nodes at each sampling period. Time steps that are not modulo ``istep``
            are stored temporarily in the ``_stock`` attribute and then that temporary
            store is returned when time step is modulo ``istep``.

            Records to a file by overloading the record method.

        """

    path = Attr(field_type=os.PathLike, required=False, default="")

    def record(self, step, state=None):
        """Method to record monitor output to a file."""
        output = super(TVBtoCosimStateInterfaceToFile, self).record(step, state)
        # TODO: Record output to file now!
        raise NotImplementedError


class TVBtoCosimStateInterfaceToMPI(TVBtoCosimStateInterface):
    """ CosimStateMonitorToMPI class:
            Monitors the value for the model's variable/s of interest over all
            the nodes at each sampling period. Time steps that are not modulo ``istep``
            are stored temporarily in the ``_stock`` attribute and then that temporary
            store is returned when time step is modulo ``istep``.

            Records to a MPI port by overloading the record method.
    """

    path = Attr(field_type=os.PathLike, required=False, default="")

    def record(self, step, state=None):
        """Method to record monitor output to a MPI port."""
        output = super(TVBtoCosimStateInterfaceToMPI, self).record(step, state)
        # TODO: Record output to MPI port now!
        raise NotImplementedError


class TVBtoCosimHistoryInterface(TVBtoCosimInterface):
    """ TVBtoCosimStateInterface class:
        Monitors the value for the model's variable/s of interest over all
        the nodes at each sampling period. Time steps that are not modulo ``istep``
        are stored temporarily in the ``_stock`` attribute and then that temporary
        store is returned when time step is modulo ``istep``.

    """
    pass


class TVBtoCosimHistoryInterfaceToFile(TVBtoCosimHistoryInterface):
    """ TVBtoCosimHistoryInterfaceToFile class:
            Monitors the value for the model's variable/s of interest over all
            the nodes at each sampling period. Time steps that are not modulo ``istep``
            are stored temporarily in the ``_stock`` attribute and then that temporary
            store is returned when time step is modulo ``istep``.

            Records to a file by overloading the record method.

        """

    path = Attr(field_type=os.PathLike, required=False, default="")

    def record(self, step, state=None):
        """Method to record monitor output to a file."""
        output = super(TVBtoCosimHistoryInterfaceToFile, self).record(step, state)
        # TODO: Record output to file now!
        raise NotImplementedError


class TVBtoCosimHistoryInterfaceToMPI(TVBtoCosimHistoryInterface):
    """ TVBtoCosimHistoryInterfaceToMPI class:
            Monitors the value for the model's variable/s of interest over all
            the nodes at each sampling period. Time steps that are not modulo ``istep``
            are stored temporarily in the ``_stock`` attribute and then that temporary
            store is returned when time step is modulo ``istep``.

            Records to a MPI port by overloading the record method.
    """

    path = Attr(field_type=os.PathLike, required=False, default="")

    def record(self, step, state=None):
        """Method to record monitor output to a MPI port."""
        output = super(TVBtoCosimHistoryInterfaceToMPI, self).record(step, state)
        # TODO: Record output to MPI port now!
        raise NotImplementedError


class TVBtoCosimInterfaces(HasTraits):

    """This class holds lists of
       - state_interfaces,
       - history_interfaces (including coupling interfaces),
       monitors"""

    state_interfaces = List(of=TVBtoCosimStateInterface)
    history_interfaces = List(of=TVBtoCosimHistoryInterface)

    @property
    def interfaces(self):
        return self.state_interfaces + self.history_interfaces

    @property
    def coupling_interfaces(self):
        return [interfaces for interfaces in self.interfaces if isinstance(interfaces.monitor, CouplingMonitor)]

    @property
    def sync_times(self):
        return [history_interface.sync_time for history_interface in self.history_interfaces]

    @property
    def voi(self):
        return [interfaces.voi for interfaces in self.interfaces if not isinstance(interfaces, CouplingMonitor)]

    @property
    def cvoi(self):
        return [interfaces.voi for interfaces in self.coupling_interfaces]

    @property
    def state_proxy_inds(self):
        return [interfaces.proxy_inds for interfaces in self.interfaces if not isinstance(interfaces, CouplingMonitor)]

    @property
    def coupling_proxy_inds(self):
        return [interfaces.proxy_inds for interfaces in self.coupling_interfaces]

    @property
    def proxy_inds(self):
        return self.state_proxy_inds + self.coupling_proxy_inds

    def configure(self, simulator):
        for interface in self.interfaces:
            interface.configure(simulator)
        super(TVBtoCosimInterfaces, self).configure()

