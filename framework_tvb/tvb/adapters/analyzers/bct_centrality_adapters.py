# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
# Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
# Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
# The Virtual Brain: a simulator of primate brain network dynamics.
# Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#
from tvb.basic.filters.chain import FilterChain

from tvb.core.entities.model import AlgorithmTransientGroup
from tvb.adapters.analyzers.bct_adapters import _BaseBCT, bct_description
from tvb.datatypes.connectivity import Connectivity


BCT_GROUP_CENTRALITY = AlgorithmTransientGroup("Centrality Algorithms", "Brain Connectivity Toolbox")



class CentralityNodeBinary(_BaseBCT):
    """
    """
    _ui_group = BCT_GROUP_CENTRALITY
    _ui_connectivity_label = "Binary (directed/undirected) connection matrix:"

    _ui_name = "Node Betweenness Centrality Binary"
    _ui_description = bct_description("betweenness_bin.m")
    _matlab_code = "C = betweenness_bin(A);"


    def launch(self, connectivity, **kwargs):
        kwargs['A'] = connectivity.binarized_weights
        result = self.execute_matlab(self._matlab_code, **kwargs)
        measure = self.build_connectivity_measure(result, 'C', connectivity,
                                                  "Node Betweenness Centrality Binary", "Nodes")
        return [measure]



class CentralityNodeWeighted(CentralityNodeBinary):
    """
    """
    _ui_connectivity_label = "Weighted (directed/undirected)  connection matrix:"

    _ui_name = "Node Betweenness Centrality Weighted"
    _ui_description = bct_description("betweenness_wei.m")
    _matlab_code = "C = betweenness_wei(A);"


    def launch(self, connectivity, **kwargs):
        kwargs['A'] = connectivity.weights
        result = self.execute_matlab(self._matlab_code, **kwargs)
        measure = self.build_connectivity_measure(result, 'C', connectivity,
                                                  "Node Betweenness Centrality Weighted", "Nodes")
        return [measure]



class CentralityEdgeBinary(CentralityNodeBinary):
    """
    """
    _ui_name = "Edge Betweenness Centrality Weighted"
    _ui_description = bct_description("edge_betweenness_bin.m")
    _matlab_code = "[EBC,BC] = edge_betweenness_bin(A);"


    def launch(self, connectivity, **kwargs):
        kwargs['A'] = connectivity.binarized_weights
        result = self.execute_matlab(self._matlab_code, **kwargs)
        measure1 = self.build_connectivity_measure(result, 'EBC', connectivity, "Edge Betweenness Centrality Matrix")
        measure2 = self.build_connectivity_measure(result, 'BC', connectivity, "Node Betweenness Centrality Vector")
        return [measure1, measure2]



class CentralityEdgeWeighted(CentralityNodeWeighted):
    """
    """
    _ui_name = "Edge Betweenness Centrality Weighted"
    _ui_description = bct_description("edge_betweenness_wei.m")
    _matlab_code = "[EBC,BC] = edge_betweenness_wei(A);"


    def launch(self, connectivity, **kwargs):
        kwargs['A'] = connectivity.weights
        result = self.execute_matlab(self._matlab_code, **kwargs)
        measure1 = self.build_connectivity_measure(result, 'EBC', connectivity, "Edge Betweenness Centrality Matrix")
        measure2 = self.build_connectivity_measure(result, 'BC', connectivity, "Node Betweenness Centrality Vector")
        return [measure1, measure2]



class CentralityEigenVector(CentralityNodeBinary):
    """
    """
    _ui_connectivity_label = "Binary/weighted undirected adjacency matrix"

    _ui_name = "EigenVector Centrality"
    _ui_description = bct_description("eigenvector_centrality_und.m")
    _matlab_code = "v = eigenvector_centrality_und(CIJ)"


    def get_input_tree(self):
        return [dict(name="connectivity", label=self._ui_connectivity_label, type=Connectivity, required=True,
                     conditions=FilterChain(fields=[FilterChain.datatype + '._undirected'],
                                            operations=["=="], values=['1']))]


    def launch(self, connectivity, **kwargs):
        kwargs['CIJ'] = connectivity.weights
        result = self.execute_matlab(self._matlab_code, **kwargs)
        measure = self.build_connectivity_measure(result, 'v', connectivity, "Eigen vector centrality")
        return [measure]



class CentralityKCoreness(CentralityEigenVector):
    """
    """
    _ui_connectivity_label = "Connection/adjacency matrix (binary, unidirected)"

    _ui_name = "K-coreness centrality BU"
    _ui_description = bct_description("kcoreness_centrality_bu.m")
    _matlab_code = "[coreness, kn] = kcoreness_centrality_bu(CIJ);"


    def launch(self, connectivity, **kwargs):
        kwargs['CIJ'] = connectivity.binarized_weights
        result = self.execute_matlab(self._matlab_code, **kwargs)
        measure1 = self.build_connectivity_measure(result, 'coreness', connectivity, "Node coreness BU")
        measure2 = self.build_connectivity_measure(result, 'kn', connectivity, "Size of k-core")
        return [measure1, measure2]



class CentralityShortcuts(CentralityNodeBinary):
    """
    """
    _ui_connectivity_label = "Binary directed connection matrix:"

    _ui_name = "Centrality Shortcuts"
    _ui_description = bct_description("erange.m")
    _matlab_code = "[Erange,eta,Eshort,fs]  = erange(A);"


    def launch(self, connectivity, **kwargs):
        kwargs['A'] = connectivity.binarized_weights
        result = self.execute_matlab(self._matlab_code, **kwargs)

        measure1 = self.build_connectivity_measure(result, 'Erange', connectivity, "Range for each edge")
        value1 = self.build_int_value_wrapper(result, 'eta', "Average range for entire graph")
        measure2 = self.build_connectivity_measure(result, 'Eshort', connectivity, "Shortcut edges")
        value2 = self.build_float_value_wrapper(result, 'fs', "Fraction of shortcuts in the graph")
        return [measure1, value1, measure2, value2]

