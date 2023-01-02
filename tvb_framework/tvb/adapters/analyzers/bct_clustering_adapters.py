# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

import bct
import numpy
from tvb.core.entities.model.model_operation import AlgorithmTransientGroup
from tvb.adapters.analyzers.bct_adapters import BaseBCT, BaseUndirected, LABEL_CONN_WEIGHTED_UNDIRECTED, \
    LABEL_CONN_WEIGHTED_DIRECTED

BCT_GROUP_CLUSTERING = AlgorithmTransientGroup("Clustering Algorithms", "Brain Connectivity Toolbox", "bctclustering")


class ClusteringCoefficient(BaseBCT):
    """
    """
    _ui_group = BCT_GROUP_CLUSTERING
    _ui_name = "Clustering Coefficient BD: Binary directed connection matrix"
    _ui_description = bct.clustering_coef_bd.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.clustering_coef_bd(connectivity.weights)
        measure_index = self.build_connectivity_measure(result, connectivity, "Clustering Coefficient BD")
        return [measure_index]


class ClusteringCoefficientBU(BaseUndirected):
    """
    """
    _ui_group = BCT_GROUP_CLUSTERING
    _ui_name = "Clustering Coefficient BU"
    _ui_description = bct.clustering_coef_bu.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.clustering_coef_bu(connectivity.weights)
        measure_index = self.build_connectivity_measure(result, connectivity, "Clustering Coefficient BU")
        return [measure_index]


class ClusteringCoefficientWU(BaseUndirected):
    """
    """
    _ui_group = BCT_GROUP_CLUSTERING
    _ui_name = "Clustering Coefficient WU: " + LABEL_CONN_WEIGHTED_UNDIRECTED
    _ui_description = bct.clustering_coef_wu.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        try:
            result = bct.clustering_coef_wu(connectivity.scaled_weights())
        except FloatingPointError as ex:
            self.log.exception(ex)
            self.add_operation_additional_info("FloatingPointError got during computation, defaulted to 0s!")
            result = numpy.zeros(connectivity.number_of_regions)

        measure_index = self.build_connectivity_measure(result, connectivity, "Clustering Coefficient WU")
        return [measure_index]


class ClusteringCoefficientWD(ClusteringCoefficient):
    """
    """
    _ui_name = "Clustering Coefficient WD: " + LABEL_CONN_WEIGHTED_DIRECTED
    _ui_description = bct.clustering_coef_wd.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.clustering_coef_wd(connectivity.scaled_weights())
        measure_index = self.build_connectivity_measure(result, connectivity, "Clustering Coefficient WD")
        return [measure_index]


class TransitivityBinaryDirected(BaseBCT):
    """
    """
    _ui_group = BCT_GROUP_CLUSTERING
    _ui_name = "Transitivity Binary Directed: Binary directed connection matrix"
    _ui_description = bct.transitivity_bd.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.transitivity_bd(connectivity.weights)
        value = self.build_float_value_wrapper(result, "Transitivity Binary Directed")
        return [value]


class TransitivityWeightedDirected(TransitivityBinaryDirected):
    """
    """
    _ui_name = "Transitivity Weighted Directed: " + LABEL_CONN_WEIGHTED_DIRECTED
    _ui_description = bct.transitivity_wd.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.transitivity_wd(connectivity.scaled_weights())
        value = self.build_float_value_wrapper(result, "Transitivity Weighted Directed")
        return [value]


class TransitivityBinaryUnDirected(BaseUndirected):
    """
    """
    _ui_group = BCT_GROUP_CLUSTERING
    _ui_name = "Transitivity Binary Undirected"
    _ui_description = bct.transitivity_bu.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.transitivity_bu(connectivity.weights)
        value = self.build_float_value_wrapper(result, "Transitivity Binary Undirected")
        return [value]


class TransitivityWeightedUnDirected(TransitivityBinaryUnDirected):
    """
    """
    _ui_name = "Transitivity Weighted undirected: " + LABEL_CONN_WEIGHTED_UNDIRECTED
    _ui_description = bct.transitivity_wu.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.transitivity_wu(connectivity.scaled_weights())
        value = self.build_float_value_wrapper(result, "Transitivity Weighted Undirected")
        return [value]
