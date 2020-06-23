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

from tvb.core.adapters.arguments_serialisation import preprocess_space_parameters
from tvb.adapters.datatypes.h5.spectral_h5 import DataTypeMatrixH5
from tvb.adapters.datatypes.h5.structural_h5 import VolumetricDataMixin
from tvb.core.entities.load import load_entity_by_gid
from tvb.core.neotraits.h5 import H5File, DataSet, Reference
from tvb.datatypes.region_mapping import RegionMapping, RegionVolumeMapping


class RegionMappingH5(H5File):

    def __init__(self, path):
        super(RegionMappingH5, self).__init__(path)
        self.array_data = DataSet(RegionMapping.array_data, self)
        self.connectivity = Reference(RegionMapping.connectivity, self)
        self.surface = Reference(RegionMapping.surface, self)

    def get_region_mapping_slice(self, start_idx, end_idx):
        """
        Get a slice of the region mapping as used by the region viewers.
        For each vertex on the surface, alpha-indices will be the closest
        region-index
        :param start_idx: vertex index on the surface
        :param end_idx: vertex index on the surface
        :return: NumPy array with [closest_reg_idx ...]
        """
        return self.array_data.load()[int(start_idx): int(end_idx)].T



class RegionVolumeMappingH5(VolumetricDataMixin, DataTypeMatrixH5):

    def __init__(self, path):
        super(RegionVolumeMappingH5, self).__init__(path)
        self.array_data = DataSet(RegionVolumeMapping.array_data, self)
        self.connectivity = Reference(RegionVolumeMapping.connectivity, self)
        self.volume = Reference(RegionVolumeMapping.volume, self)

    def get_voxel_region(self, x_plane, y_plane, z_plane):

        data_shape = self.array_data.shape
        x_plane, y_plane, z_plane = preprocess_space_parameters(x_plane, y_plane, z_plane, data_shape[0],
                                                                data_shape[1], data_shape[2])
        slices = slice(x_plane, x_plane + 1), slice(y_plane, y_plane + 1), slice(z_plane, z_plane + 1)
        voxel = self.array_data[slices][0, 0, 0]
        if voxel != -1:
            conn_index = load_entity_by_gid(self.connectivity.load().hex)
            return conn_index.region_labels[int(voxel)]
        else:
            return 'background'
