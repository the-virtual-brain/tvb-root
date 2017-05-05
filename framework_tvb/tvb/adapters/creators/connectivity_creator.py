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

import numpy
from tvb.core.adapters.abcadapter import ABCAsynchronous
from tvb.core.entities.storage import dao
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.region_mapping import RegionMapping


class ConnectivityCreator(ABCAsynchronous):
    """
    This adapter creates a Connectivity.
    """

    def get_input_tree(self):
        return [{'name': 'original_connectivity', 'label': 'Parent connectivity',
                 'type': Connectivity, 'required': True},
                {'name': 'new_weights', 'label': 'Weights json array',
                 'type': 'array', 'elementType': 'float', 'required': True},
                {'name': 'new_tracts', 'label': 'Tracts json array',
                 'type': 'array', 'elementType': 'float', 'required': True},
                {'name': 'interest_area_indexes', 'label': 'Indices of selected nodes as json array',
                 'type': 'array', 'elementType': 'int', 'required': True},
                {'name': 'is_branch', 'label': 'Is it a branch',
                 'type': 'bool', 'required': True}]


    def get_output(self):
        return [Connectivity, RegionMapping]


    def get_required_disk_size(self, original_connectivity, new_weights, new_tracts, interest_area_indexes, **kwargs):
        n = len(new_weights) if kwargs.get('is_branch') else len(interest_area_indexes)
        matrices_nr_elems = 2 * n * n
        arrays_nr_elems = ( 1 + 3 + 1 + 3 ) * n  # areas, centres, hemispheres, orientations
        matrices_estimate = (matrices_nr_elems + arrays_nr_elems) * numpy.array(0).itemsize
        labels_guesstimate = n * numpy.array(['some label']).nbytes
        return (matrices_estimate + labels_guesstimate) / 1024


    def get_required_memory_size(self, **_):
        return -1  # does not consume significant additional memory beyond the parameters


    def launch(self, original_connectivity, new_weights, new_tracts, interest_area_indexes, **kwargs):
        """
        Method to be called when user submits changes on the
        Connectivity matrix in the Visualizer.
        """
        # note: is_branch is missing instead of false because browsers only send checked boxes in forms.
        if not kwargs.get('is_branch'):
            result_connectivity = original_connectivity.cut_new_connectivity_from_ordered_arrays(
                                        new_weights, interest_area_indexes, self.storage_path, new_tracts)
            return [result_connectivity]

        else:
            result = []
            result_connectivity = original_connectivity.branch_connectivity_from_ordered_arrays(
                new_weights, interest_area_indexes, self.storage_path, new_tracts)
            result.append(result_connectivity)

            linked_region_mappings = dao.get_generic_entity(RegionMapping, original_connectivity.gid, '_connectivity')
            for mapping in linked_region_mappings:
                result.append(mapping.generate_new_region_mapping(result_connectivity.gid, self.storage_path))

            return result
