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
from tvb.basic.traits import exceptions
from tvb.basic.readers import try_get_absolute_path, FileReader
import tvb.datatypes.arrays as arrays
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.surfaces import Surface
from tvb.datatypes.volumes import Volume
from tvb.basic.logger.builder import get_logger
from tvb.basic.arguments_serialisation import parse_slice, preprocess_space_parameters
from tvb.datatypes.structural import VolumetricDataMixin


LOG = get_logger(__name__)


class RegionMapping(arrays.MappedArray):
    """
    An array (of length Surface.vertices). Each value is representing the index in Connectivity regions
    to which the current vertex is mapped.
    """

    array_data = arrays.IndexArray()

    connectivity = Connectivity

    surface = Surface

    __generate_table__ = True

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
        for i in range(triangles_no):
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

    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(RegionMapping, self)._find_summary_info()
        summary.update({"Source Surface": self.surface.display_name,
                        "Source Surface GID": self.surface.gid,
                        "Connectivity GID": self.connectivity.gid,
                        "Connectivity": self.connectivity.display_name})
        return summary


class RegionVolumeMapping(VolumetricDataMixin, arrays.MappedArray):
    """
    Each value is representing the index in Connectivity regions to which the current voxel is mapped.
    """

    array_data = arrays.IndexArray()

    connectivity = Connectivity

    volume = Volume

    __generate_table__ = True

    apply_corrections = True
    mappings_file = None

    def write_data_slice(self, data):
        """
        We are using here the same signature as in TS, just to allow easier parsing code.
        This method will also validate the data range nd convert it to int, along with writing it is H5.

        :param data: 3D int array
        """

        LOG.info("Writing RegionVolumeMapping with min=%d, mix=%d" % (data.min(), data.max()))
        if self.apply_corrections:
            data = numpy.array(data, dtype=numpy.int32)
            data[data >= self.connectivity.number_of_regions] = -1
            data[data < -1] = -1
            LOG.debug("After corrections: RegionVolumeMapping min=%d, mix=%d" % (data.min(), data.max()))

        if self.mappings_file:
            try:
                mapping_data = numpy.loadtxt(self.mappings_file, dtype=numpy.str, usecols=(0, 2))
                mapping_data = {int(row[0]): int(row[1]) for row in mapping_data}
            except Exception:
                raise exceptions.ValidationException("Invalid Mapping File. Expected 3 columns (int, string, int)")

            if len(data.shape) != 3:
                raise exceptions.ValidationException('Invalid RVM data. Expected 3D.')

            not_matched = set()
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    for k in range(data.shape[2]):
                        val = data[i][j][k]
                        if not mapping_data.has_key(val):
                            not_matched.add(val)
                        data[i][j][k] = mapping_data.get(val, -1)

            LOG.info("Imported RM with values in interval [%d - %d]" % (data.min(), data.max()))
            if not_matched:
                LOG.warn("Not matched regions will be considered background: %s" % not_matched)

        if data.min() < -1 or data.max() >= self.connectivity.number_of_regions:
            raise exceptions.ValidationException("Invalid Mapping array: [%d ... %d]" % (data.min(), data.max()))

        self.store_data("array_data", data)


    def get_voxel_region(self, x_plane, y_plane, z_plane):
        x_plane, y_plane, z_plane = preprocess_space_parameters(x_plane, y_plane, z_plane, self.length_1d,
                                                                self.length_2d, self.length_3d)
        slices = slice(x_plane, x_plane + 1), slice(y_plane, y_plane + 1), slice(z_plane, z_plane + 1)
        voxel = self.read_data_slice(slices)[0, 0, 0]
        if voxel != -1:
            return self.connectivity.region_labels[int(voxel)]
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

    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(RegionVolumeMapping, self)._find_summary_info()
        summary.update({"Source Volume": self.volume.display_name,
                        "Source Volume GID": self.volume.gid,
                        "Connectivity GID": self.connectivity.gid,
                        "Connectivity": self.connectivity.display_name})
        return summary