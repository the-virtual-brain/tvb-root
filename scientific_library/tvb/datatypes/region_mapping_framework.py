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
DataTypes for mapping some TVB DataTypes to a Connectivity (regions).

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import numpy
from tvb.basic.traits import exceptions
from tvb.basic.logger.builder import get_logger
from tvb.basic.arguments_serialisation import parse_slice, preprocess_space_parameters
from tvb.datatypes.region_mapping_data import RegionMappingData, RegionVolumeMappingData

LOG = get_logger(__name__)


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
        :return: NumPy array with [closest_reg_idx ...]
        """
        if isinstance(start_idx, (str, unicode)):
            start_idx = int(start_idx)
        if isinstance(end_idx, (str, unicode)):
            end_idx = int(end_idx)

        return self.array_data[start_idx: end_idx].T


    def get_triangles_mapping(self):
        """
        :return Numpy array of length triangles and for each the region corresponding to one of its vertices.
        """
        triangles_no = self.surface.number_of_triangles
        result = []
        for i in xrange(triangles_no):
            result.append(self.array_data[self.surface.triangles[i][0]])
        return numpy.array(result)


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



class RegionVolumeMappingFramework(RegionVolumeMappingData):
    """
    Framework methods regarding RegionVolumeMapping DataType.
    """
    __tablename__ = None
    apply_corrections = True


    def write_data_slice(self, data):
        """
        We are using here the same signature as in TS, just to allow easier parsing code.
        This method will also validate the data range nd convert it to int, along with writing it is H5.

        :param data: 3D int array
        """

        LOG.info("Writing RegionVolumeMapping with min=%d, mix=%d" % (data.min(), data.max()))
        if self.apply_corrections:
            data = numpy.array(data, dtype=numpy.int32)
            data = data - 1
            data[data >= self.connectivity.number_of_regions] = -1
            LOG.debug("After corrections: RegionVolumeMapping min=%d, mix=%d" % (data.min(), data.max()))

        if data.min() < -1 or data.max() >= self.connectivity.number_of_regions:
            raise exceptions.ValidationException("Invalid Mapping array: [%d ... %d]" % (data.min(), data.max()))

        self.store_data("array_data", data)


    def get_volume_slice(self, x_plane, y_plane, z_plane):
        slices = slice(self.length_1d), slice(self.length_2d), slice(z_plane, z_plane + 1)
        slice_x = self.read_data_slice(slices)[:, :, 0]  # 2D
        slice_x = numpy.array(slice_x, dtype=int)

        slices = slice(x_plane, x_plane + 1), slice(self.length_2d), slice(self.length_3d)
        slice_y = self.read_data_slice(slices)[0, :, :][..., ::-1]
        slice_y = numpy.array(slice_y, dtype=int)

        slices = slice(self.length_1d), slice(y_plane, y_plane + 1), slice(self.length_3d)
        slice_z = self.read_data_slice(slices)[:, 0, :][..., ::-1]
        slice_z = numpy.array(slice_z, dtype=int)

        return [slice_x, slice_y, slice_z]


    def get_volume_view(self, x_plane, y_plane, z_plane, **kwargs):
        # Work with space inside Volume:
        x_plane, y_plane, z_plane = preprocess_space_parameters(x_plane, y_plane, z_plane, self.length_1d,
                                                                self.length_2d, self.length_3d)
        slice_x, slice_y, slice_z = self.get_volume_slice(x_plane, y_plane, z_plane)
        return [[slice_x.tolist()], [slice_y.tolist()], [slice_z.tolist()]]


    def get_voxel_region(self, x_plane, y_plane, z_plane):
        x_plane, y_plane, z_plane = preprocess_space_parameters(x_plane, y_plane, z_plane, self.length_1d,
                                                                self.length_2d, self.length_3d)
        slices = slice(x_plane, x_plane + 1), slice(y_plane, y_plane + 1), slice(z_plane, z_plane + 1)
        voxel = self.read_data_slice(slices)[0, 0, 0]
        if voxel != -1:
            return self.connectivity.region_labels[voxel]
        else:
            return 'background'


    def get_mapped_array_volume_view(self, mapped_array, x_plane, y_plane, z_plane, mapped_array_slice=None, **kwargs):
        x_plane, y_plane, z_plane = preprocess_space_parameters(x_plane, y_plane, z_plane, self.length_1d,
                                                                self.length_2d, self.length_3d)
        slice_x, slice_y, slice_z = self.get_volume_slice(x_plane, y_plane, z_plane)

        if mapped_array_slice:
            matrix_slice = parse_slice(mapped_array_slice)
            measure = mapped_array.get_data('array_data', matrix_slice)
        else:
            measure = mapped_array.get_data('array_data')

        if measure.shape != (self.connectivity.number_of_regions, ):
            raise ValueError('cannot project measure on the space')

        result_x = measure[slice_x]
        result_y = measure[slice_y]
        result_z = measure[slice_z]
        # Voxels outside the brain are -1. The indexing above is incorrect for those voxels as it
        # associates the values of the last region measure[-1] to them.
        # Here we replace those values with an out of scale value.
        result_x[slice_x==-1] = measure.min() - 1
        result_y[slice_y==-1] = measure.min() - 1
        result_z[slice_z==-1] = measure.min() - 1

        return [[result_x.tolist()],
                [result_y.tolist()],
                [result_z.tolist()]]
