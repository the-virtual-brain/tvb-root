# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
# Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
# Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
# The Virtual Brain: a simulator of primate brain network dynamics.
# Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#
from tvb.core.entities.load import load_entity_by_gid
from tvb.core.entities.model.model_operation import AlgorithmTransientGroup
from tvb.adapters.analyzers.bct_adapters import BaseBCT, bct_description, BaseBCTForm
from tvb.core.neocom import h5

BCT_GROUP_DEGREE = AlgorithmTransientGroup("Degree and Similarity Algorithms",
                                           "Brain Connectivity Toolbox", "bctdegree")
BCT_GROUP_DENSITY = AlgorithmTransientGroup("Density Algorithms", "Brain Connectivity Toolbox", "bctdensity")


class DegreeForm(BaseBCTForm):
    @staticmethod
    def get_connectivity_label():
        return "Undirected (binary/weighted) connection matrix"

class Degree(BaseBCT):
    """
    """
    _ui_group = BCT_GROUP_DEGREE

    _ui_name = "Degree"
    _ui_description = bct_description("degrees_und.m")
    _matlab_code = "deg = degrees_und(CIJ);"

    def get_form_class(self):
        return DegreeForm

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        data = dict([('CIJ', connectivity.weights)])

        result = self.execute_matlab(self._matlab_code, data=data)
        measure = self.build_connectivity_measure(result, 'deg', connectivity, "Node degree")
        measure_index = self.load_entity_by_gid(measure.gid.hex)
        return [measure_index]

class DegreeIODForm(BaseBCTForm):
    @staticmethod
    def get_connectivity_label():
        return "Directed (binary/weighted) connection matrix"

class DegreeIOD(Degree):
    """
    """

    _ui_name = "Indegree and outdegree"
    _ui_description = bct_description("degrees_dir.m")
    _matlab_code = "[id,od,deg] = degrees_dir(CIJ);"

    def get_form_class(self):
        return DegreeIODForm

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        data = dict([('CIJ', connectivity.weights)])

        result = self.execute_matlab(self._matlab_code, data=data)
        measure1 = self.build_connectivity_measure(result, 'id', connectivity, "Node indegree")
        measure_index1 = self.load_entity_by_gid(measure1.gid.hex)
        measure2 = self.build_connectivity_measure(result, 'od', connectivity, "Node outdegree")
        measure_index2 = self.load_entity_by_gid(measure2.gid.hex)
        measure3 = self.build_connectivity_measure(result, 'deg', connectivity, "Node degree (indegree + outdegree)")
        measure_index3 = self.load_entity_by_gid(measure3.gid.hex)
        return [measure_index1, measure_index2, measure_index3]

class JointDegreeForm(BaseBCTForm):
    @staticmethod
    def get_connectivity_label():
        return "Connection Matrix"

class JointDegree(Degree):
    """
    """

    _ui_name = "Joint Degree"
    _ui_description = bct_description("jdegree.m")
    _matlab_code = "[J,J_od,J_id,J_bl] = jdegree(CIJ);"

    def get_form_class(self):
        return JointDegreeForm

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        data = dict([('CIJ', connectivity.weights)])

        result = self.execute_matlab(self._matlab_code, data=data)
        measure = self.build_connectivity_measure(result, 'J', connectivity,
                                                  "'Joint Degree JOD= ' +str(result['J_od'])+ ', JID= ' +str(result['J_id'])+ ', JBL= ' +str(result['J_bl'])",
                                                  "Connectivity Nodes", "Connectivity Nodes")
        measure_index = self.load_entity_by_gid(measure.gid.hex)
        value1 = self.build_int_value_wrapper(result, 'J_od', "Number of vertices with od &gt; id")
        value2 = self.build_int_value_wrapper(result, 'J_id', "Number of vertices with id &gt; od")
        value3 = self.build_int_value_wrapper(result, 'J_bl', "Number of vertices with id = od")
        return [measure_index, value1, value2, value3]

class MatchingIndexForm(BaseBCTForm):
    @staticmethod
    def get_connectivity_label():
        return "Connection/adjacency matrix"

class MatchingIndex(Degree):
    """
    """

    _ui_name = "Matching Index"
    _ui_description = bct_description("matching_ind.m")
    _matlab_code = "[Min,Mout,Mall] = matching_ind(CIJ);"

    def get_form_class(self):
        return MatchingIndexForm

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        data = dict([('CIJ', connectivity.weights)])

        result = self.execute_matlab(self._matlab_code, data=data)
        measure1 = self.build_connectivity_measure(result, 'Min', connectivity,
                                                   "Matching index for incoming connections")
        measure_index1 = self.load_entity_by_gid(measure1.gid.hex)
        measure2 = self.build_connectivity_measure(result, 'Mout', connectivity,
                                                   "Matching index for outgoing connections")
        measure_index2 = self.load_entity_by_gid(measure2.gid.hex)
        measure3 = self.build_connectivity_measure(result, 'Mall', connectivity, "Matching index for all connections")
        measure_index3 = self.load_entity_by_gid(measure3.gid.hex)
        return [measure_index1, measure_index2, measure_index3]


class Strength(Degree):
    """
    """
    _ui_connectivity_label = "Directed weighted connection matrix"

    _ui_name = "Strength"
    _ui_description = bct_description("strengths_und.m")
    _matlab_code = "strength = strengths_und(CIJ);"


    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        data = dict([('CIJ', connectivity.weights)])

        result = self.execute_matlab(self._matlab_code, data=data)
        measure = self.build_connectivity_measure(result, 'strength', connectivity, "Node strength")
        measure_index = self.load_entity_by_gid(measure.gid.hex)
        return [measure_index]


class StrengthISOS(Strength):
    """
    """
    _ui_name = "Instrength and Outstrength"
    _ui_description = bct_description("strengths_dir.m")
    _matlab_code = "[is,os,strength] = strengths_dir(CIJ);"


    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        data = dict([('CIJ', connectivity.weights)])

        result = self.execute_matlab(self._matlab_code, data=data)
        measure1 = self.build_connectivity_measure(result, 'is', connectivity, "Node instrength")
        measure_index1 = self.load_entity_by_gid(measure1.gid.hex)
        measure2 = self.build_connectivity_measure(result, 'os', connectivity, "Node outstrength")
        measure_index2 = self.load_entity_by_gid(measure2.gid.hex)
        measure3 = self.build_connectivity_measure(result, 'strength', connectivity,
                                                   "Node strength (instrength + outstrength)")
        measure_index3 = self.load_entity_by_gid(measure3.gid.hex)
        return [measure_index1, measure_index2, measure_index3]


class StrengthWeights(Strength):
    """
    """
    _ui_name = "Strength and Weight"
    _ui_description = bct_description("strengths_und_sign.m")
    _matlab_code = "[Spos,Sneg,vpos,vneg] = strengths_und_sign(CIJ);"


    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        data = dict([('CIJ', connectivity.weights)])

        result = self.execute_matlab(self._matlab_code, data=data)
        measure1 = self.build_connectivity_measure(result, 'Spos', connectivity, "Nodal strength of positive weights")
        measure_index1 = self.load_entity_by_gid(measure1.gid.hex)
        measure2 = self.build_connectivity_measure(result, 'Sneg', connectivity, "Nodal strength of negative weights")
        measure_index2 = self.load_entity_by_gid(measure2.gid.hex)
        value1 = self.build_float_value_wrapper(result, 'vpos', "Total positive weight")
        value2 = self.build_float_value_wrapper(result, 'vneg', "Total negative weight")
        return [measure_index1, measure_index2, value1, value2]

class DensityDirectedForm(BaseBCTForm):
    @staticmethod
    def get_connectivity_label():
        return "(weighted/binary) connection matrix"

class DensityDirected(BaseBCT):
    """
    """
    _ui_group = BCT_GROUP_DENSITY

    _ui_name = "Density Directed"
    _ui_description = bct_description("density_dir.m")
    _matlab_code = "[kden,N,K]  = density_dir(A);"

    def get_form_class(self):
        return DensityDirectedForm

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        data = dict([('A', connectivity.weights)])

        result = self.execute_matlab(self._matlab_code, data=data)
        value1 = self.build_float_value_wrapper(result, 'kden', title="Density")
        value2 = self.build_int_value_wrapper(result, 'N', title="Number of vertices")
        value3 = self.build_int_value_wrapper(result, 'K', title="Number of edges")
        return [value1, value2, value3]


class DensityUndirected(DensityDirected):
    """
    """
    _ui_connectivity_label = "Undirected (weighted/binary) connection matrix:"

    _ui_name = "Density Unirected"
    _ui_description = bct_description("density_und.m")
    _matlab_code = "[kden,N,K]  = density_und(A);"
