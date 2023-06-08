# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

from tvb.adapters.datatypes.h5.spectral_h5 import DataTypeMatrixH5
from tvb.adapters.datatypes.h5.structural_h5 import VolumetricDataMixin
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

    def read_data_shape(self):
        """
        The shape of the data
        """
        return self.array_data.shape

    def read_data_slice(self, data_slice):
        """
        Expose chunked-data access.
        """
        return self.array_data[data_slice]
