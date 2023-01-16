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
from abc import abstractmethod
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.adapters.datatypes.db.mapped_value import ValueWrapperIndex
from tvb.adapters.datatypes.h5.mapped_value_h5 import ValueWrapper
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.model.model_operation import AlgorithmTransientGroup
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.graph import ConnectivityMeasure

BCT_GROUP_MODULARITY = AlgorithmTransientGroup("Modularity Algorithms", "Brain Connectivity Toolbox", "bct")
BCT_GROUP_DISTANCE = AlgorithmTransientGroup("Distance Algorithms", "Brain Connectivity Toolbox", "bctdistance")

LABEL_CONNECTIVITY_BINARY = "Binary (directed/undirected) connection matrix"
LABEL_CONN_WEIGHTED_DIRECTED = "Weighted directed connection matrix"
LABEL_CONN_WEIGHTED_UNDIRECTED = "Weighted undirected connection matrix"


class BaseBCTModel(ViewModel):
    connectivity = DataTypeGidAttr(
        linked_datatype=Connectivity,
        label='Connectivity',
    )


class BaseBCTForm(ABCAdapterForm):
    def __init__(self):
        super(BaseBCTForm, self).__init__()
        self.connectivity = TraitDataTypeSelectField(BaseBCTModel.connectivity, name="connectivity",
                                                     conditions=self.get_filters(), has_all_option=True)

    @staticmethod
    def get_required_datatype():
        return ConnectivityIndex

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_input_name():
        return "connectivity"

    @staticmethod
    def get_view_model():
        return BaseBCTModel


class BaseUnidirectedBCTForm(BaseBCTForm):

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.undirected'], operations=["=="], values=['1'])


class BaseBCT(ABCAdapter):
    """
    Interface between Brain Connectivity Toolbox of Olaf Sporns and TVB Framework.
    """

    def get_form_class(self):
        return BaseBCTForm

    def get_output(self):
        return [ConnectivityMeasureIndex, ValueWrapperIndex]

    def get_required_memory_size(self, view_model):
        # We do not know how much memory is needed.
        return -1

    def get_required_disk_size(self, view_model):
        return 0

    def get_connectivity(self, view_model):
        return self.load_traited_by_gid(view_model.connectivity)

    def build_connectivity_measure(self, array_data, connectivity, title="", label_x="", label_y=""):
        measure = ConnectivityMeasure()
        measure.array_data = array_data
        measure.connectivity = connectivity
        measure.title = title
        measure.label_x = label_x
        measure.label_y = label_y
        return self.store_complete(measure)

    def build_float_value_wrapper(self, result_value, title=""):
        value = ValueWrapper()
        value.data_value = str(float(result_value))
        value.data_type = 'float'
        value.data_name = title
        return self.store_complete(value)

    def build_int_value_wrapper(self, result_int, title=""):
        value = ValueWrapper()
        value.data_value = str(int(result_int))
        value.data_type = 'int'
        value.data_name = title
        return self.store_complete(value)

    @abstractmethod
    def launch(self, view_model):
        pass


class BaseUndirected(BaseBCT):
    """
    """

    def get_form_class(self):
        return BaseUnidirectedBCTForm

    @abstractmethod
    def launch(self, view_model):
        pass


class ModularityOCSM(BaseBCT):
    """
    """
    _ui_group = BCT_GROUP_MODULARITY

    _ui_name = "Compute optimal Community Structure and Modularity from a Directed (weighted or binary) connection " \
               "matrix: "
    _ui_description = bct.modularity_dir.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.modularity_dir(connectivity.weights)
        measure = self.build_connectivity_measure(result[0], connectivity, "Optimal Community Structure")
        value = self.build_float_value_wrapper(result[1], title="Maximized Modularity")
        return [measure, value]


class ModularityOpCSMU(ModularityOCSM):
    """
    """
    _ui_name = "Optimal Community Structure and Modularity (Undirected):"
    _ui_description = bct.modularity_und.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.modularity_und(connectivity.weights)
        measure = self.build_connectivity_measure(result[0], connectivity, "Optimal Community Structure")
        value = self.build_float_value_wrapper(result[1], title="Maximized Modularity")
        return [measure, value]


DISTANCE_MATRIX_TITLE = "Distance matrix"


class DistanceDBIN(BaseBCT):
    """
    """
    _ui_group = BCT_GROUP_DISTANCE

    _ui_name = "Distance binary matrix"
    _ui_description = bct.distance_bin.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result_arr = bct.distance_bin(connectivity.weights)
        measure = self.build_connectivity_measure(result_arr, connectivity, DISTANCE_MATRIX_TITLE)
        return [measure]


class DistanceDWEI(DistanceDBIN):
    """
    """
    _ui_name = "Distance weighted matrix over a Weighted (directed/undirected) connection matrix"
    _ui_description = bct.distance_wei.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.distance_wei(connectivity.weights)[0]
        measure = self.build_connectivity_measure(result, connectivity, DISTANCE_MATRIX_TITLE)
        return [measure]


class DistanceRDM(DistanceDBIN):
    """
    """
    _ui_name = "Reachability and distance matrices (Breadth-first search)"
    _ui_description = bct.breadthdist.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.breadthdist(connectivity.weights)
        measure1 = self.build_connectivity_measure(result[0], connectivity, "Reachability matrix")
        measure2 = self.build_connectivity_measure(result[1], connectivity, DISTANCE_MATRIX_TITLE)
        return [measure1, measure2]


class DistanceRDA(DistanceRDM):
    """
    """
    _ui_name = "Reachability and distance matrices (Algebraic path count)"
    _ui_description = bct.reachdist.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.reachdist(connectivity.weights)
        measure1 = self.build_connectivity_measure(result[0], connectivity, "Reachability matrix")
        measure2 = self.build_connectivity_measure(result[1], connectivity, DISTANCE_MATRIX_TITLE)
        return [measure1, measure2]


class DistanceNETW(DistanceDBIN):
    """
    """
    _ui_name = "Network walks"
    _ui_description = bct.findwalks.__doc__

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        result = bct.findwalks(connectivity.weights)

        measure1 = self.build_connectivity_measure(result[0], connectivity, "3D matrix")
        measure2 = self.build_connectivity_measure(result[2], connectivity, "Walk length distribution")
        value = self.build_float_value_wrapper(result[1], title="Total number of walks found")

        return [measure1, value, measure2]
