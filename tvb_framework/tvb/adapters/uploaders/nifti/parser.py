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

"""
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
import numpy
import nibabel
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import ParseException


class NIFTIParser(object):
    """
    This class reads content of a NIFTI file and writes a 4D array [time, x, y, z].
    """

    def __init__(self, data_file):

        self.logger = get_logger(__name__)

        if data_file is None:
            raise ParseException("Please select NIFTI file which contains data to import")

        if not os.path.exists(data_file):
            raise ParseException("Provided file %s does not exists" % data_file)

        try:
            self.nifti_image = nibabel.load(data_file)
        except nibabel.spatialimages.ImageFileError as e:
            self.logger.exception(e)
            msg = "File: %s does not have a valid NIFTI-1 format." % data_file
            raise ParseException(msg)

        nifti_image_hdr = self.nifti_image.header

        # Check if there is a time dimensions (4th dimension).
        nifti_data_shape = nifti_image_hdr.get_data_shape()
        self.nr_dims = len(nifti_data_shape)
        self.has_time_dimension = self.nr_dims > 3
        self.time_dim_size = nifti_data_shape[3] if self.has_time_dimension else 1

        # Extract sample unit measure
        self.units = nifti_image_hdr.get_xyzt_units()

        # Usually zooms defines values for x, y, z, time and other dimensions
        self.zooms = nifti_image_hdr.get_zooms()

    def parse(self):
        """
        Parse NIFTI file and write in result_dt a 4D or 3D array [time*, x, y, z].
        """

        # Copy data from NIFTI file to our TVB storage
        # In NIFTI format time is the 4th dimension, while our TimeSeries has
        # it as first dimension, so we have to adapt imported data

        nifti_data = self.nifti_image.dataobj
        return numpy.array(nifti_data, dtype=numpy.int32)
