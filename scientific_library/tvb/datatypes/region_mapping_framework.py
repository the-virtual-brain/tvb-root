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
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
module docstring
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
from tvb.datatypes.region_mapping_data import RegionMappingData


class RegionMappingFramework(RegionMappingData):
    """
    Framework methods regarding RegionMapping DataType.
    """
    __tablename__ = None

    def get_region_mapping_slice(self, start_idx, end_idx):
        """
        Get a slice of the region mapping as used by the region viewers.
        For each vertex on the surface, alpha-indices will be the closest
        region-index
        :param start_idx: vertex index on the surface
        :param end_idx: vertex index on the surface
        :return: NumPy array with [colosest_reg_idx ...]
        """
        if isinstance(start_idx, (str, unicode)):
            start_idx = int(start_idx)
        if isinstance(end_idx, (str, unicode)):
            end_idx = int(end_idx)

        return self.array_data[start_idx: end_idx].T


    def generate_new_region_mapping(self, connectivity_gid, storage_path):
        """
        Generate a new region mapping with the given connectivity gid from an
        existing mapping corresponding to the parent connectivity.
        """
        new_region_map = self.__class__()
        new_region_map.storage_path = storage_path
        new_region_map._connectivity = connectivity_gid
        new_region_map._surface = self._surface
        new_region_map.array_data = self.array_data
        return new_region_map