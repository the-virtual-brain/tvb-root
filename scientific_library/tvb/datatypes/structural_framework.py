# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
Framework methods for the Volume datatypes.
.. moduleauthor:: Andrei Mihai <mihai.andrei@codemart.ro>
"""
import numpy
from tvb.basic.arguments_serialisation import preprocess_space_parameters
from tvb.datatypes import structural_data


class VolumetricDataFrameworkMixin(object):

    def write_data_slice(self, data):
        """
        We are using here the same signature as in TS, just to allow easier parsing code.
        This is not a chunked write.
        """
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


    def get_min_max_values(self):
        """
        Retrieve the minimum and maximum values from the metadata.
        :returns: (minimum_value, maximum_value)
        """
        metadata = self.get_metadata('array_data')
        return metadata[self.METADATA_ARRAY_MIN], metadata[self.METADATA_ARRAY_MAX]


class StructuralMRIFramework(VolumetricDataFrameworkMixin, structural_data.StructuralMRIData):

    __tablename__ = None

