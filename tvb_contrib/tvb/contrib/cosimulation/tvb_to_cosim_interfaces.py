# -*- coding: utf-8 -*-
from tvb.basic.neotraits.api import HasTraits, Float, List

# -*- coding: utf-8 -*-
import os
import numpy
from tvb.basic.neotraits.api import Attr, NArray
from tvb.simulator import monitor


class TVBtoCosimInterface(HasTraits):

    """TVBtoCosimInterface base class holding a Monitor class instance."""

    proxy_inds = NArray(
        dtype=numpy.int,
        label="Indices of TVB proxy nodes",
        required=True,
    )

    voi = NArray(
        dtype=int,
        label="Cosimulation model state variables' indices",
        doc=("Indices of model's variables of interest (VOI) that"
             "should be updated (i.e., overwriten) during cosimulation."),
        required=True)

    _number_of_proxy_nodes = 0


    path = Attr(field_type=os.PathLike, required=False, default="")

    record_to = Attr(field_type=str, required=True, default="memory")

    _record = None  # Method to implement recording to memory, file, or MPI port

    def record_to_memory(self, data):
        """Record input data to memory"""
        raise NotImplementedError

    def record_to_file(self, data):
        """Record input data to a file.
        """
        raise NotImplementedError

    def record_to_mpi(self, data):
        """Record input data to a MPI port.
        """
        raise NotImplementedError

    def _config_recording_target(self):
        self.record_to = self.record_to.lower()
        if self.record_to == "file":
            self._record = self.record_to_file
        elif self.record_to == "mpi":
            self._record = self.record_to_mpi
        else:
            self._record = self.record_to_memory

    def configure(self):
        """Method to configure the CosimMonitor of the interface
           and compute the number_of_proxy_nodes, from user defined proxy_inds"""
        self._config_recording_target()
        self.number_of_proxy_nodes = len(self.proxy_inds)
        super(TVBtoCosimInterfaces).configure()

    def record(self, data):
        """Record a sample of the observed state at the given step to the target destination.
           Use the TVB Monitor record method to get the data, and pass them to one of the
           record_to_memory, record_to_file and record_to_mpi methods,
           depending on the user input of record_to.
        """
        return self._record(data)


class TVBtoCosimInterfaces(HasTraits):

    """This class holds a list of state_interfaces"""

    interfaces = List(of=TVBtoCosimInterface)

    number_of_interfaces = None

    @property
    def voi(self):
        return [interfaces.voi for interfaces in self.interfaces] # Todo need to be unique

    @property
    def proxy_inds(self):
        return [interfaces.proxy_inds for interfaces in self.interfaces] # Todo need to be unique

    def configure(self):
        for interface in self.interfaces:
            interface.configure()
        self.number_of_interfaces = len(self.interfaces)
        super(TVBtoCosimInterfaces, self).configure()
