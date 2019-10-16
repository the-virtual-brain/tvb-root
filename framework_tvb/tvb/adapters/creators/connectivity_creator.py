# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import os
import uuid
import numpy
from tvb.core.adapters.abcadapter import ABCAsynchronous, ABCAdapterForm, ABCSynchronous
from tvb.core.entities.file.datatypes.connectivity_h5 import ConnectivityH5
from tvb.core.entities.file.datatypes.region_mapping_h5 import RegionMappingH5
from tvb.core.entities.model.datatypes.connectivity import ConnectivityIndex
from tvb.core.entities.model.datatypes.region_mapping import RegionMappingIndex
from tvb.core.entities.storage import dao
from tvb.datatypes.connectivity import Connectivity

from tvb.core.neotraits._forms import DataTypeSelectField, SimpleBoolField, SimpleArrayField
from tvb.interfaces.neocom._h5loader import DirLoader


class ConnectivityCreatorForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(ConnectivityCreatorForm, self).__init__(prefix, project_id)
        self.original_connectivity = DataTypeSelectField(self.get_required_datatype(), self, required=True,
                                                         name='original_connectivity', label='Parent connectivity',
                                                         conditions=self.get_filters())
        self.new_weights = SimpleArrayField(self, name='new_weights', dtype=numpy.float, required=True,
                                            label='Weights json array')
        self.new_tracts = SimpleArrayField(self, name='new_tracts', dtype=numpy.float, required=True,
                                           label='Tracts json array')
        self.interest_area_indexes = SimpleArrayField(self, name='interest_area_indexes', dtype=numpy.int,
                                                      required=True, label='Indices of selected nodes as json array')
        self.is_branch = SimpleBoolField(self, required=True, name='is_branch', label='Is it a branch')

    @staticmethod
    def get_required_datatype():
        return ConnectivityIndex

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_input_name():
        return '_original_connectivity'


#TODO: make this work asynchronously (form fill issue)
class ConnectivityCreator(ABCSynchronous):
    """
    This adapter creates a Connectivity.
    """
    form = None

    def get_input_tree(self): return None

    def get_form(self):
        if self.form is None:
            return ConnectivityCreatorForm
        return self.form

    def set_form(self, form):
        self.form = form

    def get_output(self):
        return [ConnectivityIndex, RegionMappingIndex]

    def get_required_disk_size(self, original_connectivity, new_weights, new_tracts, interest_area_indexes, **kwargs):
        n = len(new_weights) if kwargs.get('is_branch') else len(interest_area_indexes)
        matrices_nr_elems = 2 * n * n
        arrays_nr_elems = (1 + 3 + 1 + 3) * n  # areas, centres, hemispheres, orientations
        matrices_estimate = (matrices_nr_elems + arrays_nr_elems) * numpy.array(0).itemsize
        labels_guesstimate = n * numpy.array(['some label']).nbytes
        return (matrices_estimate + labels_guesstimate) / 1024

    def get_required_memory_size(self, **_):
        return -1  # does not consume significant additional memory beyond the parameters

    def _store_connectivity_datatype(self, new_connectivity, parent_connectivity=None):
        new_connectivity_idx = ConnectivityIndex()
        new_connectivity_idx.fill_from_has_traits(new_connectivity)

        loader = DirLoader(self.storage_path)
        new_conn_path = loader.path_for(ConnectivityH5, new_connectivity_idx.gid)
        with ConnectivityH5(new_conn_path) as new_conn_h5:
            new_conn_h5.store(new_connectivity)
            new_conn_h5.gid.store(uuid.UUID(new_connectivity_idx.gid))
            new_conn_h5.parent_connectivity.store(parent_connectivity)

        return new_connectivity_idx

    def _store_related_region_mappings(self, original_connectivity_id, new_connectivity_index):
        result = []

        linked_region_mappings = dao.get_generic_entity(RegionMappingIndex, original_connectivity_id, 'connectivity_id')
        for mapping in linked_region_mappings:
            dir_loader = DirLoader(os.path.join(os.path.dirname(self.storage_path), str(mapping.fk_from_operation)))
            rm_h5_path = dir_loader.path_for(RegionMappingH5, mapping.gid)

            result_rm_index = RegionMappingIndex()
            dir_loader = DirLoader(os.path.join(self.storage_path))
            result_rm_h5_path = dir_loader.path_for(RegionMappingH5, result_rm_index.gid)

            with RegionMappingH5(rm_h5_path) as rm_h5, RegionMappingH5(result_rm_h5_path) as result_rm_h5:
                result_rm_h5.connectivity.store(uuid.UUID(new_connectivity_index.gid))
                result_rm_h5.gid.store(uuid.UUID(result_rm_index.gid))
                result_rm_h5.surface.store(rm_h5.surface.load())
                result_rm_h5.array_data.store(rm_h5.array_data.load())

            result_rm_index.array_data_min = mapping.array_data_min
            result_rm_index.array_data_max = mapping.array_data_max
            result_rm_index.array_data_mean = mapping.array_data_mean
            result_rm_index.surface_id = mapping.surface_id
            #TODO: add connectivity_id fk for new RMs
            result_rm_index.connectivity = new_connectivity_index

            result.append(result_rm_index)

        return result

    def launch(self, original_connectivity, new_weights, new_tracts, interest_area_indexes, **kwargs):
        """
        Method to be called when user submits changes on the
        Connectivity matrix in the Visualizer.
        """
        # note: is_branch is missing instead of false because browsers only send checked boxes in forms.

        loader = DirLoader(
            os.path.join(os.path.dirname(self.storage_path), str(original_connectivity.fk_from_operation)))
        original_conn_path = loader.path_for(ConnectivityH5, original_connectivity.gid)
        original_conn_dt = Connectivity()
        with ConnectivityH5(original_conn_path) as original_conn_h5:
            original_conn_h5.load_into(original_conn_dt)

        if not kwargs.get('is_branch'):
            result_connectivity_dt = original_conn_dt.cut_new_connectivity_from_ordered_arrays(
                numpy.array(new_weights), numpy.array(interest_area_indexes), numpy.array(new_tracts))

            return [self._store_connectivity_datatype(result_connectivity_dt)]

        else:
            result = []
            result_connectivity_dt = original_conn_dt.branch_connectivity_from_ordered_arrays(numpy.array(new_weights),
                                                                                              numpy.array(interest_area_indexes),
                                                                                              numpy.array(new_tracts))
            new_conn_index = self._store_connectivity_datatype(result_connectivity_dt,
                                                               original_connectivity.gid)
            result.append(new_conn_index)
            result.extend(self._store_related_region_mappings(original_connectivity.id, new_conn_index))

            return result
