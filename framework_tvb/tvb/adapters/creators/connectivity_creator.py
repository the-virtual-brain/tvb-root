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

"""
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import numpy
from tvb.basic.neotraits.api import Attr, NArray
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAsynchronous
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.core.entities.load import load_entity_by_gid
from tvb.core.entities.storage import dao
from tvb.core.neotraits.forms import ArrayField, BoolField, TraitDataTypeSelectField
from tvb.core.neocom import h5
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.region_mapping import RegionMapping


class ConnectivityCreatorModel(ViewModel):

    original_connectivity = DataTypeGidAttr(
        linked_datatype=Connectivity,
        default=None,
        label="Parent connectivity",
        required=True
    )

    new_weights = NArray(
        default=None,
        label="Weights json array",
        required=True,
        doc="""""")

    new_tracts = NArray(
        default=None,
        label="Tracts json array",
        required=True,
        doc="""""")

    interest_area_indexes = NArray(
        dtype=numpy.int,
        default=None,
        label="Indices of selected nodes as json array",
        required=True,
        doc="""""")

    is_branch = Attr(
        field_type=bool,
        label="Is it a branch",
        required=False,
        doc="""""")

class ConnectivityCreatorForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(ConnectivityCreatorForm, self).__init__(prefix, project_id)
        self.original_connectivity = TraitDataTypeSelectField(ConnectivityCreatorModel.original_connectivity, self,
                                                         name='original_connectivity', conditions=self.get_filters())
        self.new_weights = ArrayField(ConnectivityCreatorModel.new_weights, self)
        self.new_tracts = ArrayField(ConnectivityCreatorModel.new_tracts, self)
        self.interest_area_indexes = ArrayField(ConnectivityCreatorModel.interest_area_indexes, self)
        self.is_branch = BoolField(ConnectivityCreatorModel.is_branch, self)

    @staticmethod
    def get_view_model():
        return ConnectivityCreatorModel

    @staticmethod
    def get_required_datatype():
        return ConnectivityIndex

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_input_name():
        return 'original_connectivity'


class ConnectivityCreator(ABCAsynchronous):
    """
    This adapter creates a Connectivity.
    """

    def get_form_class(self):
        return ConnectivityCreatorForm

    def get_output(self):
        return [ConnectivityIndex, RegionMappingIndex]

    def get_required_disk_size(self, view_model):
        n = len(view_model.new_weights) if view_model.is_branch else len(view_model.interest_area_indexes)
        matrices_nr_elems = 2 * n * n
        arrays_nr_elems = (1 + 3 + 1 + 3) * n  # areas, centres, hemispheres, orientations
        matrices_estimate = (matrices_nr_elems + arrays_nr_elems) * numpy.array(0).itemsize
        labels_guesstimate = n * numpy.array(['some label']).nbytes
        return (matrices_estimate + labels_guesstimate) / 1024

    def get_required_memory_size(self, view_model):
        return -1  # does not consume significant additional memory beyond the parameters

    def _store_related_region_mappings(self, original_conn_gid, new_connectivity_ht):
        result = []

        linked_region_mappings = dao.get_generic_entity(RegionMappingIndex, original_conn_gid, 'fk_connectivity_gid')
        for mapping in linked_region_mappings:
            original_rm = h5.load_from_index(mapping)
            surface_idx = dao.get_generic_entity(SurfaceIndex, mapping.fk_surface_gid, 'gid')[0]
            surface = h5.load_from_index(surface_idx)

            new_rm = RegionMapping()
            new_rm.connectivity = new_connectivity_ht
            new_rm.surface = surface
            new_rm.array_data = original_rm.array_data

            result_rm_index = h5.store_complete(new_rm, self.storage_path)
            result.append(result_rm_index)

        return result

    def launch(self, view_model):
        """
        Method to be called when user submits changes on the
        Connectivity matrix in the Visualizer.
        """
        # note: is_branch is missing instead of false because browsers only send checked boxes in forms.
        original_connectivity_index = load_entity_by_gid(view_model.original_connectivity.hex)
        original_conn_ht = h5.load_from_index(original_connectivity_index)
        assert isinstance(original_conn_ht, Connectivity)

        if not view_model.is_branch:
            new_conn_ht = self._cut_connectivity(original_conn_ht, view_model.new_weights,
                                                 view_model.interest_area_indexes, view_model.new_tracts)
            return [h5.store_complete(new_conn_ht, self.storage_path)]

        else:
            result = []
            new_conn_ht = self._branch_connectivity(original_conn_ht, view_model.new_weights,
                                                    view_model.interest_area_indexes, view_model.new_tracts)
            new_conn_index = h5.store_complete(new_conn_ht, self.storage_path)
            result.append(new_conn_index)
            result.extend(self._store_related_region_mappings(view_model.original_connectivity.gid, new_conn_ht))
            return result

    @staticmethod
    def _reorder_arrays(original_conn, new_weights, interest_areas, new_tracts=None):
        """
        Returns ordered versions of the parameters according to the hemisphere permutation.
        """
        permutation = original_conn.hemisphere_order_indices
        inverse_permutation = numpy.argsort(permutation)  # trick to invert a permutation represented as an array
        interest_areas = inverse_permutation[interest_areas]
        # see :meth"`ordered_weights` for why [p:][:p]
        new_weights = new_weights[inverse_permutation, :][:, inverse_permutation]

        if new_tracts is not None:
            new_tracts = new_tracts[inverse_permutation, :][:, inverse_permutation]

        return new_weights, interest_areas, new_tracts

    def _branch_connectivity(self, original_conn, new_weights, interest_areas,
                             new_tracts=None):
        # type: (Connectivity, numpy.array, numpy.array, numpy.array) -> Connectivity
        """
        Generate new Connectivity based on a previous one, by changing weights (e.g. simulate lesion).
        The returned connectivity has the same number of nodes. The edges of unselected nodes will have weight 0.
        :param original_conn: Original Connectivity, to copy from
        :param new_weights: weights matrix for the new connectivity
        :param interest_areas: ndarray of the selected node id's
        :param new_tracts: tracts matrix for the new connectivity
        """

        new_weights, interest_areas, new_tracts = self._reorder_arrays(original_conn, new_weights,
                                                                       interest_areas, new_tracts)
        if new_tracts is None:
            new_tracts = original_conn.tract_lengths

        for i in range(len(original_conn.weights)):
            for j in range(len(original_conn.weights)):
                if i not in interest_areas or j not in interest_areas:
                    new_weights[i][j] = 0

        final_conn = Connectivity()
        final_conn.parent_connectivity = original_conn.gid.hex
        final_conn.saved_selection = interest_areas.tolist()
        final_conn.weights = new_weights
        final_conn.centres = original_conn.centres
        final_conn.region_labels = original_conn.region_labels
        final_conn.orientations = original_conn.orientations
        final_conn.cortical = original_conn.cortical
        final_conn.hemispheres = original_conn.hemispheres
        final_conn.areas = original_conn.areas
        final_conn.tract_lengths = new_tracts
        final_conn.configure()
        return final_conn

    def _cut_connectivity(self, original_conn, new_weights, interest_areas, new_tracts=None):
        # type: (Connectivity, numpy.array, numpy.array, numpy.array) -> Connectivity
        """
        Generate new Connectivity object based on current one, by removing nodes (e.g. simulate lesion).
        Only the selected nodes will get used in the result. The order of the indices in interest_areas matters.
        If indices are not sorted then the nodes will be permuted accordingly.

        :param original_conn: Original Connectivity(HasTraits), to cut nodes from
        :param new_weights: weights matrix for the new connectivity
        :param interest_areas: ndarray with the selected node id's.
        :param new_tracts: tracts matrix for the new connectivity
        """
        new_weights, interest_areas, new_tracts = self._reorder_arrays(original_conn, new_weights,
                                                                       interest_areas, new_tracts)
        if new_tracts is None:
            new_tracts = original_conn.tract_lengths[interest_areas, :][:, interest_areas]
        else:
            new_tracts = new_tracts[interest_areas, :][:, interest_areas]
        new_weights = new_weights[interest_areas, :][:, interest_areas]

        final_conn = Connectivity()
        final_conn.parent_connectivity = None
        final_conn.weights = new_weights
        final_conn.centres = original_conn.centres[interest_areas, :]
        final_conn.region_labels = original_conn.region_labels[interest_areas]
        if original_conn.orientations is not None and len(original_conn.orientations):
            final_conn.orientations = original_conn.orientations[interest_areas, :]
        if original_conn.cortical is not None and len(original_conn.cortical):
            final_conn.cortical = original_conn.cortical[interest_areas]
        if original_conn.hemispheres is not None and len(original_conn.hemispheres):
            final_conn.hemispheres = original_conn.hemispheres[interest_areas]
        if original_conn.areas is not None and len(original_conn.areas):
            final_conn.areas = original_conn.areas[interest_areas]
        final_conn.tract_lengths = new_tracts
        final_conn.saved_selection = []
        final_conn.configure()
        return final_conn
