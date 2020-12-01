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
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

import os
from abc import abstractmethod
from tvb.adapters.analyzers.matlab_worker import MatlabWorker
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.adapters.datatypes.db.mapped_value import ValueWrapperIndex
from tvb.adapters.datatypes.h5.mapped_value_h5 import ValueWrapper
from tvb.basic.profile import TvbProfile
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.model.model_operation import AlgorithmTransientGroup
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.core.utils import extract_matlab_doc_string
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.graph import ConnectivityMeasure

BCT_GROUP_MODULARITY = AlgorithmTransientGroup("Modularity Algorithms", "Brain Connectivity Toolbox", "bct")
BCT_GROUP_DISTANCE = AlgorithmTransientGroup("Distance Algorithms", "Brain Connectivity Toolbox", "bctdistance")

BCT_PATH = os.path.join(TvbProfile.current.EXTERNALS_FOLDER_PARENT, "externals/BCT")
BCT_PATH_ENV = 'BCT_PATH'
if BCT_PATH_ENV in os.environ and os.path.exists(os.environ[BCT_PATH_ENV]) and os.path.isdir(os.environ[BCT_PATH_ENV]):
    BCT_PATH = os.environ[BCT_PATH_ENV]

LABEL_CONNECTIVITY_BINARY = "Binary (directed/undirected) connection matrix"
LABEL_CONN_WEIGHTED_DIRECTED = "Weighted directed connection matrix"
LABEL_CONN_WEIGHTED_UNDIRECTED = "Weighted undirected connection matrix"


def bct_description(mat_file_name):
    return extract_matlab_doc_string(os.path.join(BCT_PATH, mat_file_name))


class BaseBCTModel(ViewModel):
    connectivity = DataTypeGidAttr(
        linked_datatype=Connectivity,
        label='Connectivity',
    )


class BaseBCTForm(ABCAdapterForm):
    def __init__(self, project_id=None):
        super(BaseBCTForm, self).__init__(project_id)
        self.connectivity = TraitDataTypeSelectField(BaseBCTModel.connectivity, self.project_id, name="connectivity",
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
    This adapter requires BCT deployed locally, and Matlab or Octave installed separately of TVB.
    """

    def __init__(self):
        ABCAdapter.__init__(self)
        self.matlab_worker = MatlabWorker()

    @staticmethod
    def can_be_active():
        return not not TvbProfile.current.MATLAB_EXECUTABLE

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
        conn_index = self.load_entity_by_gid(view_model.connectivity)
        return h5.load_from_index(conn_index)

    def execute_matlab(self, matlab_code, data):
        self.matlab_worker.add_to_path(BCT_PATH)
        self.log.info("Starting execution of MATLAB code:" + matlab_code)
        runcode, matlablog, result = self.matlab_worker.matlab(matlab_code, data=data, work_dir=self.storage_path)
        self.log.debug("Code run in MATLAB: " + str(runcode))
        self.log.debug("MATLAB log: " + str(matlablog))
        self.log.debug("Finished MATLAB execution:" + str(result))
        return result

    def build_connectivity_measure(self, result, key, connectivity, title="", label_x="", label_y=""):
        measure = ConnectivityMeasure()
        measure.array_data = result[key]
        measure.connectivity = connectivity
        measure.title = title
        measure.label_x = label_x
        measure.label_y = label_y
        return h5.store_complete(measure, self.storage_path)

    def build_float_value_wrapper(self, result, key, title=""):
        value = ValueWrapper()
        value.data_value = str(float(result[key]))
        value.data_type = 'float'
        value.data_name = title
        return h5.store_complete(value, self.storage_path)

    def build_int_value_wrapper(self, result, key, title=""):
        value = ValueWrapper()
        value.data_value = str(int(result[key]))
        value.data_type = 'int'
        value.data_name = title
        return h5.store_complete(value, self.storage_path)

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

    _ui_name = "Compute optimal Community Structure and Modularity from a Directed (weighted or binary) connection matrix:"
    _ui_description = bct_description("modularity_dir.m")
    _matlab_code = "[Ci,Q] = modularity_dir(CW);"

    def launch(self, view_model):
        # Prepare parameters
        connectivity = self.get_connectivity(view_model)
        data = {'CW': connectivity.weights}

        # Execute the matlab code
        result = self.execute_matlab(self._matlab_code, data=data)
        # Gather results
        measure = self.build_connectivity_measure(result, 'Ci', connectivity, "Optimal Community Structure")
        value = self.build_float_value_wrapper(result, 'Q', title="Maximized Modularity")
        return [measure, value]


class ModularityOpCSMU(ModularityOCSM):
    """
    """
    _ui_name = "Optimal Community Structure and Modularity (Undirected):"
    _ui_description = bct_description("modularity_und.m")
    _matlab_code = "[Ci,Q] = modularity_und(CW);"


class DistanceDBIN(BaseBCT):
    """
    """
    _ui_group = BCT_GROUP_DISTANCE

    _ui_name = "Distance binary matrix"
    _ui_description = bct_description("distance_bin.m")
    _matlab_code = "D = distance_bin(A);"

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        data = {'A': connectivity.weights}

        result = self.execute_matlab(self._matlab_code, data=data)
        measure = self.build_connectivity_measure(result, 'D', connectivity, "Distance matrix")
        return [measure]


class DistanceDWEI(DistanceDBIN):
    """
    """
    _ui_name = "Distance weighted matrix over a Weighted (directed/undirected) connection matrix"
    _ui_description = bct_description("distance_wei.m")
    _matlab_code = "D = distance_wei(A);"


class DistanceRDM(DistanceDBIN):
    """
    """
    _ui_name = "Reachability and distance matrices (Breadth-first search)"
    _ui_description = bct_description("breadthdist.m")
    _matlab_code = "[R,D] = breadthdist(A);"

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        data = {'A': connectivity.weights}

        result = self.execute_matlab(self._matlab_code, data=data)

        measure1 = self.build_connectivity_measure(result, 'R', connectivity, "Reachability matrix")
        measure2 = self.build_connectivity_measure(result, 'D', connectivity, "Distance matrix")
        return [measure1, measure2]


class DistanceRDA(DistanceRDM):
    """
    """
    _ui_name = "Reachability and distance matrices (Algebraic path count)"
    _ui_description = bct_description("reachdist.m")
    _matlab_code = "[R,D] = reachdist(A);"


class DistanceNETW(DistanceDBIN):
    """
    """
    _ui_name = "Network walks"
    _ui_description = bct_description("findwalks.m")
    _matlab_code = "[Wq,twalk,wlq]  = findwalks(A);"

    def launch(self, view_model):
        connectivity = self.get_connectivity(view_model)
        data = {'A': connectivity.weights}

        result = self.execute_matlab(self._matlab_code, data=data)

        measure1 = self.build_connectivity_measure(result, 'Wq', connectivity, "3D matrix")
        measure2 = self.build_connectivity_measure(result, 'wlq', connectivity, "Walk length distribution")
        value = self.build_float_value_wrapper(result, 'twalk', title="Total number of walks found")
        return [measure1, value, measure2]
