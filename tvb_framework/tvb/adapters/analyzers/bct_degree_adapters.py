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
from tvb.core.entities.model.model_operation import AlgorithmTransientGroup
from tvb.adapters.analyzers.bct_adapters import BaseBCT

BCT_GROUP_DEGREE = AlgorithmTransientGroup("Degree and Similarity Algorithms",
                                           "Brain Connectivity Toolbox", "bctdegree")
BCT_GROUP_DENSITY = AlgorithmTransientGroup("Density Algorithms", "Brain Connectivity Toolbox", "bctdensity")


class Degree(BaseBCT):
    """
    """
    _ui_group = BCT_GROUP_DEGREE
    _ui_name = "Degree: Undirected (binary/weighted) connection matrix"
    _ui_description = bct.degrees_und.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.degrees_und(connectivity.weights)
        measure_index = self.build_connectivity_measure(result, connectivity, "Node degree")
        return [measure_index]


class DegreeIOD(Degree):
    """
    """

    _ui_name = "Indegree and outdegree: Directed (binary/weighted) connection matrix"
    _ui_description = bct.degrees_dir.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.degrees_dir(connectivity.weights)

        measure_index1 = self.build_connectivity_measure(result[0], connectivity, "Node indegree")
        measure_index2 = self.build_connectivity_measure(result[1], connectivity, "Node outdegree")
        measure_index3 = self.build_connectivity_measure(result[2], connectivity, "Node degree (indegree + outdegree)")

        return [measure_index1, measure_index2, measure_index3]


# class JointDegree(Degree):
#     """
#     Commented because in bctpy 0.5.2 numpy can not index with floats (only ints)
#     """
#     _ui_name = "Joint Degree"
#     _ui_description = bct.jdegree.__doc__
#
#     def launch(self, view_model):
#         connectivity = self.get_connectivity(view_model)
#         result = bct.jdegree(connectivity.weights)
#         result_j_od = result[1]
#         result_j_id = result[2]
#         result_j_bl = result[3]
#         measure_index = self.build_connectivity_measure(result[0], connectivity,
#                                                         "Joint Degree JOD=" + str(result_j_od) +
#                                                         ", JID=" + str(result_j_id) +
#                                                         ", JBL=" + str(result_j_bl),
#                                                         "Connectivity Nodes", "Connectivity Nodes")
#         value1 = self.build_int_value_wrapper(result_j_od, "Number of vertices with od > id")
#         value2 = self.build_int_value_wrapper(result_j_id, "Number of vertices with id > od")
#         value3 = self.build_int_value_wrapper(result_j_bl, "Number of vertices with id = od")
#
#         return [measure_index, value1, value2, value3]


class MatchingIndex(Degree):
    """
    """
    _ui_name = "Matching Index: Connection/adjacency matrix"
    _ui_description = bct.matching_ind.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.matching_ind(connectivity.weights)

        measure_index1 = self.build_connectivity_measure(result[0], connectivity,
                                                         "Matching index for incoming connections")
        measure_index2 = self.build_connectivity_measure(result[1], connectivity,
                                                         "Matching index for outgoing connections")
        measure_index3 = self.build_connectivity_measure(result[2], connectivity,
                                                         "Matching index for all connections")
        return [measure_index1, measure_index2, measure_index3]


class Strength(Degree):
    """
    """
    _ui_name = "Strength: Directed weighted connection matrix"
    _ui_description = bct.strengths_und.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.strengths_und(connectivity.weights)
        measure_index = self.build_connectivity_measure(result, connectivity, "Node strength")
        return [measure_index]


class StrengthISOS(Strength):
    """
    """
    _ui_name = "Instrength and Outstrength"
    _ui_description = bct.strengths_dir.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.strengths_dir(connectivity.weights)
        measure_index = self.build_connectivity_measure(result, connectivity, "Node instrength + outstrength")
        return [measure_index]


class StrengthWeights(Strength):
    """
    """
    _ui_name = "Strength and Weight"
    _ui_description = bct.strengths_und_sign.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.strengths_und_sign(connectivity.weights)

        measure_index1 = self.build_connectivity_measure(result[0], connectivity,
                                                         "Nodal strength of positive weights")
        measure_index2 = self.build_connectivity_measure(result[1], connectivity,
                                                         "Nodal strength of negative weights")
        value1 = self.build_float_value_wrapper(result[2], "Total positive weight")
        value2 = self.build_float_value_wrapper(result[3], "Total negative weight")

        return [measure_index1, measure_index2, value1, value2]


class DensityDirected(BaseBCT):
    """
    """
    _ui_group = BCT_GROUP_DENSITY
    _ui_name = "Density Directed: Directed (weighted/binary) connection matrix"
    _ui_description = bct.density_dir.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.density_dir(connectivity.weights)

        value1 = self.build_float_value_wrapper(result[0], title="Density")
        value2 = self.build_int_value_wrapper(result[1], title="Number of vertices")
        value3 = self.build_int_value_wrapper(result[2], title="Number of edges")

        return [value1, value2, value3]


class DensityUndirected(DensityDirected):
    """
    """
    _ui_name = "Density Undirected: Undirected (weighted/binary) connection matrix"
    _ui_description = bct.density_und.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.density_und(connectivity.weights)

        value1 = self.build_float_value_wrapper(result[0], title="Density")
        value2 = self.build_int_value_wrapper(result[1], title="Number of vertices")
        value3 = self.build_int_value_wrapper(result[2], title="Number of edges")

        return [value1, value2, value3]
