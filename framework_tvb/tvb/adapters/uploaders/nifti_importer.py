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
from tvb.datatypes.structural import StructuralMRI
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

                {'name': 'apply_corrections', 'label': 'Apply auto Corrections', 'type': 'bool', 'default': False,
                 'description': 'Check this when the NII mapping has values outside [-1..N-1]. '
                                'All outside range will be set -1 (background).'},

                {'name': 'mappings_file', 'label': 'Mapping File', 'type': 'upload', 'required_type': '.txt',
                 'description': 'Fill this for Region Mappings, when the indices in the NII do not match the '
                                'Connectivity [0..N-1] indices'},

                {'name': 'connectivity', 'label': 'Connectivity',
                 'type': Connectivity, 'required': False, 'datatype': True,
                 'description': 'Optional Connectivity in case the NII file is a volume2regions mapping.'}]


    def get_output(self):
        return [Volume, StructuralMRI, TimeSeriesVolume, RegionVolumeMapping]


    def _create_volume(self):
        volume = Volume(storage_path=self.storage_path)
        volume.origin = [[0.0, 0.0, 0.0]]
        volume.voxel_size = [self.parser.zooms[0], self.parser.zooms[1], self.parser.zooms[2]]
        if self.parser.units is not None and len(self.parser.units) > 0:
            volume.voxel_unit = self.parser.units[0]
        return volume


    def _create_mri(self, volume):
        mri = StructuralMRI(storage_path=self.storage_path)
        mri.volume = volume
        mri.title = "NIFTI Import - " + os.path.split(self.data_file)[1]
        mri.dimensions_labels = ["X", "Y", "Z"]
        mri.weighting = "T1"
        self.parser.parse(mri, False)
        return mri


    def _create_time_series(self, volume):
        # Now create TimeSeries and fill it with data from NIFTI image
        time_series = TimeSeriesVolume(storage_path=self.storage_path)
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


    def _create_region_map(self, volume, connectivity, apply_corrections, mappings_file):
        region2volume_mapping = RegionVolumeMapping(storage_path=self.storage_path)
        region2volume_mapping.volume = volume
        region2volume_mapping.connectivity = connectivity
        region2volume_mapping.title = "NIFTI Import - " + os.path.split(self.data_file)[1]
        region2volume_mapping.dimensions_labels = ["X", "Y", "Z"]
        region2volume_mapping.apply_corrections = apply_corrections
        region2volume_mapping.mappings_file = mappings_file

        self.parser.parse(region2volume_mapping, False)
        return region2volume_mapping


    @transactional
    def launch(self, data_file, apply_corrections=False, mappings_file=None, connectivity=None):
        """
        Execute import operations:
        """
        self.data_file = data_file

        try:
            self.parser = NIFTIParser(data_file)

            volume = self._create_volume()

            if connectivity:
                rm = self._create_region_map(volume, connectivity, apply_corrections, mappings_file)
                return [volume, rm]

            if self.parser.has_time_dimension:
                time_series = self._create_time_series(volume)
                return [volume, time_series]

            # no connectivity and no time
            mri = self._create_mri(volume)
            return [volume, mri]

        except ParseException as excep:
            logger = get_logger(__name__)
            logger.exception(excep)
            raise LaunchException(excep)
