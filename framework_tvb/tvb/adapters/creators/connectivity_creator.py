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
import json
from tvb.basic.traits import traited_interface
from tvb.core.adapters.abcadapter import ABCAsynchronous
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.projections import ProjectionRegionEEG
from tvb.datatypes.surfaces import RegionMapping


class ConnectivityCreator(ABCAsynchronous):
    """
    This adapter creates a Connectivity.
    """

    def get_input_tree(self):
        connectivity = Connectivity()
        connectivity.trait.bound = traited_interface.INTERFACE_ATTRIBUTES_ONLY
        inputList = connectivity.interface[traited_interface.INTERFACE_ATTRIBUTES]
        return inputList


    def get_output(self):
        return [Connectivity]


    def get_required_disk_size(self, original_connectivity, new_weights, new_tracts, interest_area_indexes, is_branch, **_):
        n = len(json.loads(new_weights)) if is_branch else len(json.loads(interest_area_indexes))
        matrices_nr_elems = 2 * n * n
        arrays_nr_elems = ( 1 + 3 + 1 + 3 ) * n  # areas, centres, hemispheres, orientations
        labels_guesstimate = n * numpy.array(['some label']).nbytes
        return (matrices_nr_elems + arrays_nr_elems) * numpy.array(0).itemsize + labels_guesstimate


    def get_required_memory_size(self, original_connectivity, new_weights, new_tracts, interest_area_indexes, is_branch, **_):
        return -1  # does not consume significant additional memory beyond the parameters


    def launch(self, original_connectivity, new_weights, new_tracts, interest_area_indexes, is_branch, **_):
        """
        Method to be called when user submits changes on the
        Connectivity matrix in the Visualizer.
        """
        conn = self.load_entity_by_gid(original_connectivity)
        self.meta_data[DataTypeMetaData.KEY_SUBJECT] = conn.subject

        new_weights = numpy.asarray(json.loads(new_weights), dtype=numpy.float64)
        new_tracts = numpy.asarray(json.loads(new_tracts), dtype=numpy.float64)
        interest_area_indexes = numpy.asarray(json.loads(interest_area_indexes))
        is_branch = json.loads(is_branch)

        if not is_branch:
            result_connectivity = conn.cut_new_connectivity_from_ordered_arrays(new_weights, interest_area_indexes,
                                                                                            self.storage_path, new_tracts)
            return [result_connectivity]
        else:
            result = []
            result_connectivity = conn.branch_connectivity_from_ordered_arrays(new_weights, interest_area_indexes,
                                                                                     self.storage_path, new_tracts)
            result.append(result_connectivity)

            linked_region_mappings = dao.get_generic_entity(RegionMapping, original_connectivity, '_connectivity')
            for mapping in linked_region_mappings:
                result.append(mapping.generate_new_region_mapping(result_connectivity.gid, self.storage_path))

            linked_projection = dao.get_generic_entity(ProjectionRegionEEG, original_connectivity, '_sources')
            for projection in linked_projection:
                result.append(projection.generate_new_projection(result_connectivity.gid, self.storage_path))
            return result
