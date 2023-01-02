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
from tvb.adapters.analyzers.bct_adapters import BaseBCT, BaseUndirected, LABEL_CONNECTIVITY_BINARY
from tvb.core.entities.model.model_operation import AlgorithmTransientGroup

BCT_GROUP_CENTRALITY = AlgorithmTransientGroup("Centrality Algorithms", "Brain Connectivity Toolbox", "bctcentrality")


class CentralityNodeBinary(BaseBCT):
    """
    """
    _ui_group = BCT_GROUP_CENTRALITY
    _ui_name = "Node Betweenness Centrality Binary: " + LABEL_CONNECTIVITY_BINARY
    _ui_description = bct.betweenness_bin.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.betweenness_bin(connectivity.binarized_weights)
        measure_index = self.build_connectivity_measure(result, connectivity,
                                                        "Node Betweenness Centrality Binary", "Nodes")
        return [measure_index]


class CentralityNodeWeighted(BaseBCT):
    """
    """
    _ui_group = BCT_GROUP_CENTRALITY
    _ui_name = "Node Betweenness Centrality Weighted: Weighted (directed/undirected) connection matrix"
    _ui_description = bct.betweenness_wei.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.betweenness_wei(connectivity.weights)
        measure_index = self.build_connectivity_measure(result, connectivity,
                                                        "Node Betweenness Centrality Weighted", "Nodes")
        return [measure_index]


# class CentralityEdgeBinary(CentralityNodeBinary):
#     """
#     COMMENT OUT BECAUSE in v0.5.2:
#     could not broadcast input array from shape (16,) into shape (15,)
#     """
#     _ui_name = "Edge Betweenness Centrality Binary"
#     _ui_description = bct.edge_betweenness_bin.__doc__
#
#     def launch(self, view_model):
#         connectivity = self.get_connectivity(view_model)
#         result = bct.edge_betweenness_bin(connectivity.binarized_weights)
#         measure_index1 = self.build_connectivity_measure(result[0], connectivity, "Edge Betweenness Centrality Matrix")
#         measure_index2 = self.build_connectivity_measure(result[1], connectivity, "Node Betweenness Centrality Vector")
#         return [measure_index1, measure_index2]


# class CentralityEdgeWeighted(CentralityNodeWeighted):
#     """
#     COMMENT OUT BECAUSE in v0.5.2:
#     could not broadcast input array from shape (16,) into shape (15,)
#     """
#     _ui_name = "Edge Betweenness Centrality Weighted"
#     _ui_description = bct.edge_betweenness_wei.__doc__
#
#     def launch(self, view_model):
#         connectivity = self.get_connectivity(view_model)
#         result = bct.edge_betweenness_wei(connectivity.weights)
#         measure_index1 = self.build_connectivity_measure(result[0], connectivity, "Edge Betweeness Centrality Matrix")
#         measure_index2 = self.build_connectivity_measure(result[1], connectivity, "Node Betweenness Centrality Vector")
#         return [measure_index1, measure_index2]


class CentralityEigenVector(BaseUndirected):
    """
    """
    _ui_group = BCT_GROUP_CENTRALITY
    _ui_name = "EigenVector Centrality"
    _ui_description = bct.eigenvector_centrality_und.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.eigenvector_centrality_und(connectivity.weights)
        measure_index = self.build_connectivity_measure(result, connectivity, "Eigen vector centrality")
        return [measure_index]


class CentralityKCoreness(BaseUndirected):
    """
    """
    _ui_group = BCT_GROUP_CENTRALITY
    _ui_name = "K-coreness centrality BU: " + LABEL_CONNECTIVITY_BINARY
    _ui_description = bct.kcoreness_centrality_bu.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.kcoreness_centrality_bu(connectivity.binarized_weights)
        measure_index1 = self.build_connectivity_measure(result[0], connectivity, "Node coreness BU")
        measure_index2 = self.build_connectivity_measure(result[1], connectivity, "Size of k-core")
        return [measure_index1, measure_index2]


class CentralityKCorenessBD(CentralityNodeBinary):
    """
    """
    _ui_name = "K-coreness centrality BD"
    _ui_description = bct.kcoreness_centrality_bd.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.kcoreness_centrality_bd(connectivity.binarized_weights)
        measure_index1 = self.build_connectivity_measure(result[0], connectivity, "Node coreness BD")
        measure_index2 = self.build_connectivity_measure(result[1], connectivity, "Size of k-core")
        return [measure_index1, measure_index2]


class CentralityShortcuts(CentralityNodeBinary):
    """
    """

    _ui_name = "Centrality Shortcuts: Binary directed connection matrix"
    _ui_description = bct.erange.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.erange(connectivity.binarized_weights)

        measure_index1 = self.build_connectivity_measure(result[0], connectivity, "Range for each edge")
        value1 = self.build_int_value_wrapper(result[1], "Average range for entire graph")
        measure_index2 = self.build_connectivity_measure(result[2], connectivity, "Shortcut edges")
        value2 = self.build_float_value_wrapper(result[3], "Fraction of shortcuts in the graph")

        return [measure_index1, value1, measure_index2, value2]


class FlowCoefficients(CentralityNodeBinary):
    """
    """
    _ui_name = "Node-wise flow coefficients"
    _ui_description = bct.flow_coef_bd.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.flow_coef_bd(connectivity.binarized_weights)

        measure_index1 = self.build_connectivity_measure(result[0], connectivity, "Flow coefficient for each node")
        value1 = self.build_float_value_wrapper(result[1], "Average flow coefficient over the network")
        measure_index2 = self.build_connectivity_measure(result[2], connectivity,
                                                         "Number of paths that flow across the central node")
        return [measure_index1, value1, measure_index2]


class ParticipationCoefficient(BaseBCT):
    """
    """
    _ui_group = BCT_GROUP_CENTRALITY
    _ui_name = "Participation Coefficient"
    _ui_description = bct.participation_coef.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        ci, _ = bct.modularity_dir(connectivity.weights)
        try:
            result = bct.participation_coef(connectivity.weights, ci)
        except FloatingPointError as ex:
            self.log.exception(ex)
            self.add_operation_additional_info("FloatingPointError got during computation, defaulted to 0s!")
            result = numpy.zeros(connectivity.number_of_regions)

        measure_index = self.build_connectivity_measure(result, connectivity, "Participation Coefficient")
        return [measure_index]


class ParticipationCoefficientSign(ParticipationCoefficient):
    """
    """
    _ui_name = "Participation Coefficient Sign"
    _ui_description = bct.participation_coef_sign.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        ci, _ = bct.modularity_dir(connectivity.weights)
        ppos, pneg = bct.participation_coef_sign(connectivity.weights, ci)

        measure_index1 = self.build_connectivity_measure(ppos, connectivity,
                                                         "Participation Coefficient from positive weights")
        measure_index2 = self.build_connectivity_measure(pneg, connectivity,
                                                         "Participation Coefficient from negative weights")
        return [measure_index1, measure_index2]


class SubgraphCentrality(CentralityNodeBinary):
    """
    """

    _ui_name = "Subgraph centrality of a network: Adjacency matrix (binary)"
    _ui_description = bct.subgraph_centrality.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.subgraph_centrality(connectivity.binarized_weights)

        measure_index = self.build_connectivity_measure(result, connectivity, "Subgraph Centrality")
        return [measure_index]
