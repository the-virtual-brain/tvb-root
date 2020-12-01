# -*- coding: utf-8 -*-
import os
import numpy
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List

class CosimUpdate(HasTraits):

    """Base class to get update TVB states from data from co-simulator"""

    proxy_inds = NArray(
        dtype=numpy.int,
        label="Indices of TVB proxy nodes",
        required=True)

    _number_of_proxy_nodes=0

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
        """Method to configure CosimUpdate
           the variables of interest,
           and the number of proxy nodes"""
        self.number_of_proxy_nodes = len(self.proxy_inds)
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.variables_of_interest)]
        super(CosimUpdate, self).configure()

    def get_value(self):
        """Method to get data from co-simulator."""
        raise NotImplementedError


class CosimFromFile(CosimUpdate):

    """Class to update the current TVB state from a file,
       by implementing the get_value() method accordingly."""

    path = Attr(field_type=os.PathLike, required=False, default="")

    def get_value(self):
        """Method to get update data from co-simulator."""
        raise NotImplementedError


class CosimFromMPI(CosimUpdate):

    """"Class to update the current TVB state from a MPI port,
        by implementing the get_value() method accordingly."""

    def get_value(self):
        """Method to get update data from co-simulator."""
        raise NotImplementedError

class CosimToTVBInterfaces(HasTraits):

    """This class holds lists of CosimStateUpdate and CosimHistoryUpdate class instances"""

    _interfaces = List(of=CosimUpdate) # list of interface

    @property
    def interfaces(self):
        return self._interfaces

    @property
    def vois(self):
        return [interface.voi for interface in self._interfaces]  # Todo need to be unique

    @property
    def proxy_inds(self):
        return [interface.proxy_inds for interface in self._interfaces]  # Todo need to be unique

    def configure(self, simulator):
        for interfaces in self.interfacess:
            interfaces.configure(simulator)
        super(CosimToTVBInterfaces, self).configure()
