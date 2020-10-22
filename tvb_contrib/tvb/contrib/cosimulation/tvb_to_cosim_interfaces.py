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

    number_of_proxy_nodes = Attr(field_type=int,
                                 required=True,
                                 default=0,
                                 label="Number of TVB proxy nodes")

    monitor = Attr(field_type=monitor.Raw,
                   required=True,
                   default=0,
                   label="TVB monitor")

    path = Attr(field_type=os.PathLike, required=False, default="")

    record_to = Attr(field_type=str, required=True, default="memory")

    _record = None  # Method to implement recording to memory, file, or MPI port

    def record_to_memory(self, data):
        """Record input data to memory"""
        return data

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

    @property
    def voi(self):
        return self.monitor.voi

    def _sample(self, step, state):
        """Record from the TVB Monitor of the interface and return the data only of proxy_inds"""
        # Get the output by using the original sample method of the TVB monitor:
        output = self.monitor.__class__.sample(self.monitor, step, state)
        if output is not None:
            # Get only proxy data by assuming that the last two dimensions are (region nodes, modes):
            output[1] = output[1][..., self.proxy_inds, :]
            return output

    def configure(self, simulator):
        """Method to configure the CosimMonitor of the interface
           and compute the number_of_proxy_nodes, from user defined proxy_inds"""
        self._config_recording_target()
        self.monitor.config_for_sim(simulator)
        # Overwrite the sample method of the TVB Monitor class:
        self.monitor.sample = self._sample
        self.number_of_proxy_nodes = len(self.proxy_inds)
        super(TVBtoCosimInterfaces).configure()

    def record(self, step, state):
        """Record a sample of the observed state at the given step to the target destination.
           Use the TVB Monitor record method to get the data, and pass them to one of the
           record_to_memory, record_to_file and record_to_mpi methods,
           depending on the user input of record_to.
        """
        return self._record(self.monitor.record(step, state))


class TVBtoCosimInterfaces(HasTraits):

    """This class holds a list of state_interfaces"""

    interfaces = List(of=TVBtoCosimInterface)

    number_of_interfaces = None

    @property
    def voi(self):
        return [interfaces.voi for interfaces in self.interfaces]

    @property
    def proxy_inds(self):
        return [interfaces.proxy_inds for interfaces in self.interfaces]

    def configure(self, simulator):
        for interface in self.interfaces:
            interface.configure(simulator)
        self.number_of_interfaces = len(self.interfaces)
        super(TVBtoCosimInterfaces, self).configure()
