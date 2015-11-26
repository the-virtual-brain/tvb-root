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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.adapters.uploaders.nifti.parser import NIFTIParser
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.core.entities.storage import transactional
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.region_mapping import RegionVolumeMapping
from tvb.datatypes.time_series import TimeSeriesVolume
from tvb.datatypes.volumes import Volume



class NIFTIImporter(ABCUploader):
    """
    This importer is responsible for loading of data from NIFTI format (nii or nii.gz files)
    and store them in TVB as TimeSeriesVolume or RegionVolumeMapping.
    """
    _ui_name = "NIFTI"
    _ui_subsection = "nifti_importer"
    _ui_description = "Import TimeSeries Volume from NIFTI"


    def get_upload_input_tree(self):
        """
        Take as input a GZ archive or NII file.
        """
        return [{'name': 'data_file', 'type': 'upload', 'required_type': '.nii, .gz, .zip',
                 'label': 'Please select file to import (gz or nii)', 'required': True},

                {'name': 'apply_corrections', 'label': 'Apply Corrections', 'type': 'bool', 'default': False,
                 'description': 'Check this when the NII mapping is not zero based'},

                {'name': 'connectivity', 'label': 'Connectivity',
                 'type': Connectivity, 'required': False, 'datatype': True,
                 'description': 'Optional Connectivity in case the NII file is a volume2regions mapping.'}]


    def get_output(self):
        return [Volume, TimeSeriesVolume, RegionVolumeMapping]


    def _create_volume(self):
        volume = Volume(storage_path=self.storage_path)
        volume.set_operation_id(self.operation_id)
        volume.origin = [[0.0, 0.0, 0.0]]
        volume.voxel_size = [self.parser.zooms[0], self.parser.zooms[1], self.parser.zooms[2]]
        if self.parser.units is not None and len(self.parser.units) > 0:
            volume.voxel_unit = self.parser.units[0]
        return volume


    def _create_time_series(self, volume):
        # Now create TimeSeries and fill it with data from NIFTI image
        time_series = TimeSeriesVolume(storage_path=self.storage_path)
        time_series.set_operation_id(self.operation_id)
        time_series.volume = volume
        time_series.title = "NIFTI Import - " + os.path.split(self.data_file)[1]
        time_series.labels_ordering = ["Time", "X", "Y", "Z"]
        time_series.start_time = 0.0

        if len(self.parser.zooms) > 3:
            time_series.sample_period = float(self.parser.zooms[3])
        else:
            # If no time dim, set sampling to 1 sec
            time_series.sample_period = 1

        if self.parser.units is not None and len(self.parser.units) > 1:
            time_series.sample_period_unit = self.parser.units[1]

        self.parser.parse(time_series, True)
        return time_series


    def _create_region_map(self, volume, connectivity, apply_corrections):
        region2volume_mapping = RegionVolumeMapping(storage_path=self.storage_path)
        region2volume_mapping.set_operation_id(self.operation_id)
        region2volume_mapping.volume = volume
        region2volume_mapping.connectivity = connectivity
        region2volume_mapping.title = "NIFTI Import - " + os.path.split(self.data_file)[1]
        region2volume_mapping.dimensions_labels = ["X", "Y", "Z"]
        region2volume_mapping.apply_corrections = apply_corrections

        self.parser.parse(region2volume_mapping, False)
        return region2volume_mapping


    @transactional
    def launch(self, data_file, apply_corrections=False, connectivity=None):
        """
        Execute import operations:
        """
        import pydevd
        pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)

        self.data_file = data_file

        try:
            self.parser = NIFTIParser(data_file)

            # Create volume DT
            volume = self._create_volume()

            if self.parser.has_time_dimension or not connectivity:
                time_series = self._create_time_series(volume)
                return [volume, time_series]
            else:
                rm = self._create_region_map(volume, connectivity, apply_corrections)
                return [volume, rm]

        except ParseException, excep:
            logger = get_logger(__name__)
            logger.exception(excep)
            raise LaunchException(excep)