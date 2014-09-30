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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""
import os
import nibabel as nib
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import ParseException
from tvb.datatypes.time_series import TimeSeriesVolume
from tvb.datatypes.volumes import Volume


class NIFTIParser():
    """
    This class reads content of a NIFTI file and builds / returns a TimeSeries instance 
    filled with details.
    """
    def __init__(self, storage_path, operation_id):
        self.logger = get_logger(__name__)
        self.storage_path = storage_path
        self.operation_id = operation_id
        
    def parse(self, data_file):
        """
        Parse NIFTI file and returns TimeSeries for it.
        """
        if data_file is None:
            raise ParseException("Please select NIFTI file which contains data to import")

        if not os.path.exists(data_file):
            raise ParseException("Provided file %s does not exists" % data_file)
        try:
            nifti_image = nib.load(data_file)
        except nib.spatialimages.ImageFileError, e:
            self.logger.exception(e)
            msg = "File: %s does not have a valid NIFTI-1 format." % data_file
            raise ParseException(msg)
        
        nifti_image_hdr = nifti_image.get_header()
        
        # Create volume for time series
        volume = Volume(storage_path=self.storage_path)
        volume.set_operation_id(self.operation_id)
        volume.origin = [[0.0, 0.0, 0.0]]
        
        # Now create TimeSeries and fill it with data from NIFTI image                
        time_series = TimeSeriesVolume(storage_path=self.storage_path)
        time_series.set_operation_id(self.operation_id)
        time_series.volume = volume
        time_series.title = "NIFTI Import - " + os.path.split(data_file)[1]
        time_series.labels_ordering = ["Time", "X", "Y", "Z"]
        time_series.start_time = 0.0
        
        
        # Copy data from NIFTI file to our TVB storage
        # In NIFTI format time si the 4th dimension, while our TimeSeries has
        # it as first dimension, so we have to adapt imported data
        
        # Check if there is a time dimensions (4th dimension).
        nifti_data_shape = nifti_image_hdr.get_data_shape()
        has_time_dimension = len(nifti_data_shape) > 3
        time_dim_size = nifti_data_shape[3] if has_time_dimension else 1
        
        nifti_data = nifti_image.get_data()
        if has_time_dimension:
            for i in range(time_dim_size):
                time_series.write_data_slice([nifti_data[:, :, :, i, ...]])
        else:
            time_series.write_data_slice([nifti_data])
        time_series.close_file()  # Force closing HDF5 file
            
        # Extract sample unit measure
        units = nifti_image_hdr.get_xyzt_units()
        if units is not None and len(units) == 2:
            volume.voxel_unit = units[0]
            time_series.sample_period_unit = units[1]
        
        # Extract sample rate
        # Usually zooms defines values for x, y, z, time and other dimensions
        zooms = nifti_image_hdr.get_zooms()
        if has_time_dimension:
            time_series.sample_period = float(zooms[3])
        else:
            time_series.sample_period = 1.0  # If no time dim, set sampling to 1 sec
              
        # Get voxtel dimensions for x,y, z
        volume.voxel_size = [zooms[0], zooms[1], zooms[2]]
        
        return time_series      
