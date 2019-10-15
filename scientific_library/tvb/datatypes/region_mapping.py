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
DataTypes for mapping some TVB DataTypes to a Connectivity (regions).

In FreeSurfer terms, a RegionMapping is a parcellation and a VolumeMapping is a segmentation.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import numpy
from tvb.basic.readers import try_get_absolute_path, FileReader
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.surfaces import Surface
from tvb.datatypes.volumes import Volume
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import HasTraits, Attr, NArray

LOG = get_logger(__name__)


class RegionMapping(HasTraits):
    """
    An array (of length Surface.vertices). Each value is representing the index in Connectivity regions
    to which the current vertex is mapped.
    """

    array_data = NArray(dtype=int)

    connectivity = Attr(field_type=Connectivity)

    surface = Attr(field_type=Surface)

    @staticmethod
    def from_file(source_file="regionMapping_16k_76.txt", instance=None):

        if instance is None:
            result = RegionMapping()
        else:
            result = instance

        source_full_path = try_get_absolute_path("tvb_data.regionMapping", source_file)
        reader = FileReader(source_full_path)

        result.array_data = reader.read_array(dtype=numpy.int32)
        return result

    def get_region_mapping_slice(self, start_idx, end_idx):
        """
        Get a slice of the region mapping as used by the region viewers.
        For each vertex on the surface, alpha-indices will be the closest
        region-index
        :param start_idx: vertex index on the surface
        :param end_idx: vertex index on the surface
        :return: NumPy array with [closest_reg_idx ...]
        """
        if isinstance(start_idx, str):
            start_idx = int(start_idx)
        if isinstance(end_idx, str):
            end_idx = int(end_idx)

        return self.array_data[start_idx: end_idx].T


    def get_triangles_mapping(self):
        """
        :return Numpy array of length triangles and for each the region corresponding to one of its vertices.
        """
        triangles_no = self.surface.number_of_triangles
        result = []
        for i in range(triangles_no):
            result.append(self.array_data[self.surface.triangles[i][0]])
        return numpy.array(result)


class RegionVolumeMapping(HasTraits):
    """
    Each value is representing the index in Connectivity regions to which the current voxel is mapped.
    """

    array_data = NArray(dtype=int)

    connectivity = Attr(Connectivity)

    volume = Attr(Volume)

