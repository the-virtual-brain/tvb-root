# -*- coding: utf-8 -*-
import numpy
from tvb.basic.neotraits.api import NArray, List
from tvb.basic.neotraits.ex import TraitAttributeError
from tvb.simulator.models.base import Model, ModelNumbaDfun
from tvb.contrib.scripts.utils.data_structure_utils import flatten_list


class CosimModel(Model):

    """CosimModel base class"""

    cosim_vars = NArray(
        dtype=int,
        label="Cosimulation model state variables",
        doc=("Indices of model's state variables of interest (VOI) that"
             "should be updated (i.e., overwritten) during cosimulation."),
        required=False)

    cosim_vars_proxy_inds = List(of=NArray,
                                 doc=("Indices of proxy nodes' per cosimulation state variable, "
                                      "the state of which is updated (i.e., overwritten) during cosimulation."),)


    _cosim_nvar = None
    _tot_cosim_vars_proxy_inds = None
    _n_cosim_vars_proxy_inds = None

    def from_model(self, model):
        """
        Copy the value of an instance of a TVB Model class
        :param model: TVB model
        """
        for key, value in vars(model).items():
            try:
                setattr(self, key, value)
            except TraitAttributeError:
                # variable final don't need to copy
                pass

    @property
    def cosim_nvar(self):
        """ The number of state variables in this model that are updated from cosimulation. """
        if not self._cosim_nvar:
            self._cosim_nvar = len(numpy.unique(flatten_list(self.cosim_vars)))
        return self._cosim_nvar

    @property
    def tot_cosim_vars_proxy_inds(self):
        """ All the unique proxy region nodes indices for state variables updated from cosimulation. """
        if not self._tot_cosim_vars_proxy_inds:
            self._tot_cosim_vars_proxy_inds = numpy.unique(flatten_list(self.cosim_vars_proxy_inds))
        return self._tot_cosim_vars_proxy_inds

    @property
    def n_cosim_vars_proxy_inds(self):
        """ The total number of proxy region nodes for state variables updated from cosimulation. """
        if not self._n_cosim_vars_proxy_inds:
            self._n_cosim_vars_proxy_inds = len(self.tot_cosim_vars_proxy_inds)
        return self._n_cosim_vars_proxy_inds

    def configure(self):
        "Configure base CosimModel and compute the cosimulation related attributes."
        super(CosimModel).configure()
        self._cosim_nvar = len(self.cosim_vars)
        self._tot_cosim_cvars_proxy_inds = numpy.unique(flatten_list(self.cosim_cvars_proxy_inds))
        self._n_cosim_vars_proxy_inds = len(self.tot_cosim_vars_proxy_inds)


class CosimModelNumbaDfun(CosimModel, ModelNumbaDfun):

    @property
    def spatial_param_reshape(self):
        return -1,
