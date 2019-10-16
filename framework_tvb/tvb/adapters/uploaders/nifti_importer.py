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

import json
import os
import uuid
import numpy
from tvb.datatypes.region_mapping import RegionVolumeMapping
from tvb.datatypes.structural import StructuralMRI
from tvb.datatypes.time_series import TimeSeriesVolume
from tvb.datatypes.volumes import Volume
from tvb.adapters.uploaders.abcuploader import ABCUploader, ABCUploaderForm
from tvb.adapters.uploaders.nifti.parser import NIFTIParser
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.core.entities.file.datatypes.region_mapping_h5 import RegionVolumeMappingH5
from tvb.core.entities.file.datatypes.structural_h5 import StructuralMRIH5
from tvb.core.entities.file.datatypes.time_series import TimeSeriesVolumeH5
from tvb.core.entities.file.datatypes.volumes_h5 import VolumeH5
from tvb.core.entities.model.datatypes.connectivity import ConnectivityIndex
from tvb.core.entities.model.datatypes.region_mapping import RegionVolumeMappingIndex
from tvb.core.entities.model.datatypes.structural import StructuralMRIIndex
from tvb.core.entities.model.datatypes.time_series import TimeSeriesVolumeIndex
from tvb.core.entities.model.datatypes.volume import VolumeIndex
from tvb.core.entities.storage import transactional
from tvb.core.neotraits._forms import UploadField, SimpleBoolField, DataTypeSelectField
from tvb.core.neotraits.db import prepare_array_shape_meta
from tvb.interfaces.neocom._h5loader import DirLoader


class NIFTIImporterForm(ABCUploaderForm):

    def __init__(self, prefix='', project_id=None):
        super(NIFTIImporterForm, self).__init__(prefix, project_id)

        self.data_file = UploadField('.nii, .gz, .zip', self, name='data_file', required=True,
                                     label='Please select file to import (gz or nii)')
        self.apply_corrections = SimpleBoolField(self, name='apply_corrections', label='Apply auto Corrections',
                                                 doc='Check this when the NII mapping has values outside [-1..N-1]. '
                                                     'All outside range will be set -1 (background).')
        self.mappings_file = UploadField('.txt', self, name='mappings_file', label='Mapping File',
                                         doc='Fill this for Region Mappings, when the indices in the NII do not match '
                                             'the Connectivity [0..N-1] indices')
        self.connectivity = DataTypeSelectField(ConnectivityIndex, self, name='connectivity', label='Connectivity',
                                                doc='Optional Connectivity in case the NII file is a volume2regions mapping.')


class NIFTIImporter(ABCUploader):
    """
    This importer is responsible for loading of data from NIFTI format (nii or nii.gz files)
    and store them in TVB as TimeSeriesVolume or RegionVolumeMapping.
    """
    _ui_name = "NIFTI"
    _ui_subsection = "nifti_importer"
    _ui_description = "Import TimeSeries Volume from NIFTI"

    form = None

    def get_input_tree(self): return None

    def get_upload_input_tree(self): return None

    def get_form(self):
        if self.form is None:
            return NIFTIImporterForm
        return self.form

    def set_form(self, form):
        self.form = form

    def get_output(self):
        return [VolumeIndex, StructuralMRIIndex, TimeSeriesVolumeIndex, RegionVolumeMappingIndex]

    def _create_volume(self):
        volume = Volume()
        volume.origin = numpy.array([[0.0, 0.0, 0.0]])
        volume.voxel_size = numpy.array([self.parser.zooms[0], self.parser.zooms[1], self.parser.zooms[2]])
        if self.parser.units is not None and len(self.parser.units) > 0:
            volume.voxel_unit = self.parser.units[0]

        volume_idx = VolumeIndex()
        volume_idx.fill_from_has_traits(volume)

        volume_h5_path = self.loader.path_for(VolumeH5, volume_idx.gid)
        with VolumeH5(volume_h5_path) as volume_h5:
            volume_h5.store(volume)
            volume_h5.gid.store(uuid.UUID(volume_idx.gid))

        return volume_idx

    def _create_mri(self, volume):
        mri = StructuralMRI()
        mri.title = "NIFTI Import - " + os.path.split(self.data_file)[1]
        mri.dimensions_labels = ["X", "Y", "Z"]
        mri.weighting = "T1"

        mri_idx = StructuralMRIIndex()
        mri_idx.volume = volume

        mri_h5_path = self.loader.path_for(StructuralMRIH5, mri_idx.gid)
        nifti_data = self.parser.parse()

        with StructuralMRIH5(mri_h5_path) as mri_h5:
            mri_h5.weighting.store(mri.weighting)
            mri_h5.write_data_slice(nifti_data)
            mri_h5.volume.store(uuid.UUID(volume.gid))
            mri_h5.gid.store(uuid.UUID(mri_idx.gid))
            mri.array_data = mri_h5.array_data.load()

        mri_idx.fill_from_has_traits(mri)
        return mri_idx

    def _create_time_series(self, volume):
        # Now create TimeSeries and fill it with data from NIFTI image
        time_series = TimeSeriesVolume()
        time_series.title = "NIFTI Import - " + os.path.split(self.data_file)[1]
        time_series.labels_ordering = ["Time", "X", "Y", "Z"]
        time_series.start_time = 0.0

        if len(self.parser.zooms) > 3:
            time_series.sample_period = float(self.parser.zooms[3])
        else:
            # If no time dim, set sampling to 1 sec
            time_series.sample_period = 1
        time_series.sample_rate = 1 / time_series.sample_period

        if self.parser.units is not None and len(self.parser.units) > 1:
            time_series.sample_period_unit = self.parser.units[1]

        ts_idx = TimeSeriesVolumeIndex()
        ts_idx.time_series_type = type(time_series).__name__
        ts_idx.volume = volume
        ts_idx.has_volume_mapping = True

        ts_h5_path = self.loader.path_for(TimeSeriesVolumeH5, ts_idx.gid)
        nifti_data = self.parser.parse()

        ts_h5 = TimeSeriesVolumeH5(ts_h5_path)
        ts_h5.volume.store(uuid.UUID(volume.gid))
        ts_h5.gid.store(uuid.UUID(ts_idx.gid))
        ts_h5.store(time_series, scalars_only=True, store_references=False)
        for i in range(self.parser.time_dim_size):
            ts_h5.write_data_slice([nifti_data[:, :, :, i, ...]])

        data_shape = ts_h5.read_data_shape()
        ts_h5.close()

        ts_idx.title = time_series.title
        ts_idx.sample_period_unit = time_series.sample_period_unit
        ts_idx.sample_period = time_series.sample_period
        ts_idx.sample_rate = time_series.sample_rate
        ts_idx.labels_ordering = json.dumps(time_series.labels_ordering)
        ts_idx.data_ndim = len(data_shape)
        ts_idx.data_length_1d, ts_idx.data_length_2d, ts_idx.data_length_3d, ts_idx.data_length_4d = prepare_array_shape_meta(
            data_shape)

        return ts_idx

    def _create_region_map(self, volume, connectivity, apply_corrections, mappings_file):
        region2volume_mapping = RegionVolumeMapping()
        region2volume_mapping.title = "NIFTI Import - " + os.path.split(self.data_file)[1]
        region2volume_mapping.dimensions_labels = ["X", "Y", "Z"]

        rm_idx = RegionVolumeMappingIndex()
        rm_idx.connectivity = connectivity
        rm_idx.volume = volume

        rm_h5_path = self.loader.path_for(RegionVolumeMappingH5, rm_idx.gid)
        nifti_data = self.parser.parse()

        with RegionVolumeMappingH5(rm_h5_path) as rm_h5:
            rm_h5.write_data_slice(nifti_data, apply_corrections, mappings_file, connectivity.number_of_regions)
            rm_h5.connectivity.store(uuid.UUID(connectivity.gid))
            rm_h5.volume.store(uuid.UUID(volume.gid))
            rm_h5.gid.store(uuid.UUID(rm_idx.gid))
            region2volume_mapping.array_data = rm_h5.array_data.load()

        rm_idx.fill_from_has_traits(region2volume_mapping)
        return rm_idx

    @transactional
    def launch(self, data_file, apply_corrections=False, mappings_file=None, connectivity=None):
        """
        Execute import operations:
        """
        self.data_file = data_file

        try:
            self.parser = NIFTIParser(data_file)
            self.loader = DirLoader(self.storage_path)

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
