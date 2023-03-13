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
from tvb.adapters.uploaders.nifti.parser import NIFTIParser
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesVolumeH5
from tvb.adapters.datatypes.db.region_mapping import RegionVolumeMappingIndex
from tvb.adapters.datatypes.db.structural import StructuralMRIIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesVolumeIndex
from tvb.adapters.datatypes.db.volume import VolumeIndex
from tvb.basic.logger.builder import get_logger
from tvb.basic.exceptions import ValidationException
from tvb.basic.neotraits.api import Attr
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str, DataTypeGidAttr
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.core.entities.storage import transactional
from tvb.core.neotraits.forms import TraitUploadField, BoolField, TraitDataTypeSelectField
from tvb.core.neotraits.db import prepare_array_shape_meta
from tvb.core.neocom import h5
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.region_mapping import RegionVolumeMapping
from tvb.datatypes.structural import StructuralMRI
from tvb.datatypes.time_series import TimeSeriesVolume
from tvb.datatypes.volumes import Volume


class NIFTIImporterModel(UploaderViewModel):
    data_file = Str(
        label='Please select file to import (gz or nii)'
    )

    one_based = Attr(
        field_type=bool,
        required=False,
        label='Indexes from 1',
        doc='Check this when the NII mapping has values [0..N] with 0 the background, instead  of [-1..N-1]'
    )

    apply_corrections = Attr(
        field_type=bool,
        required=False,
        label='Apply auto Corrections',
        doc='Check this when the NII mapping has values outside [-1..N-1]. '
            'All outside range will be set -1 (background).'
    )

    mappings_file = Str(
        required=False,
        label='Mapping File',
        doc='Fill this for Region Mappings, when the indices in the NII do not match '
            'the Connectivity [0..N-1] indices'
    )

    connectivity = DataTypeGidAttr(
        linked_datatype=Connectivity,
        required=False,
        label='Connectivity',
        doc='Optional Connectivity if the NII file is a volume2regions mapping'
    )


class NIFTIImporterForm(ABCUploaderForm):

    def __init__(self):
        super(NIFTIImporterForm, self).__init__()

        self.data_file = TraitUploadField(NIFTIImporterModel.data_file, ('.nii', '.gz', '.zip'), 'data_file')
        self.apply_corrections = BoolField(NIFTIImporterModel.apply_corrections, name='apply_corrections')
        self.one_based = BoolField(NIFTIImporterModel.one_based, name='one_based')
        self.mappings_file = TraitUploadField(NIFTIImporterModel.mappings_file, '.txt', 'mappings_file')
        self.connectivity = TraitDataTypeSelectField(NIFTIImporterModel.connectivity, name='connectivity')

    @staticmethod
    def get_view_model():
        return NIFTIImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'data_file': ('.nii', '.gz', '.zip'),
            'mappings_file': '.txt'
        }


class NIFTIImporter(ABCUploader):
    """
    This importer is responsible for loading of data from NIFTI format (nii or nii.gz files)
    and store them in TVB as TimeSeriesVolume or RegionVolumeMapping.
    """
    _ui_name = "NIFTI"
    _ui_subsection = "nifti_importer"
    _ui_description = "Import TimeSeries Volume from NIFTI"

    def get_form_class(self):
        return NIFTIImporterForm

    def get_output(self):
        return [VolumeIndex, StructuralMRIIndex, TimeSeriesVolumeIndex, RegionVolumeMappingIndex]

    def _create_volume(self):
        volume = Volume()
        volume.origin = numpy.array([[0.0, 0.0, 0.0]])
        volume.voxel_size = numpy.array([self.parser.zooms[0], self.parser.zooms[1], self.parser.zooms[2]])
        if self.parser.units is not None and len(self.parser.units) > 0:
            volume.voxel_unit = self.parser.units[0]

        return self.store_complete(volume), volume

    def _create_mri(self, volume, title):
        mri = StructuralMRI()
        mri.title = title
        mri.dimensions_labels = ["X", "Y", "Z"]
        mri.weighting = "T1"
        mri.array_data = self.parser.parse()
        mri.volume = volume

        return self.store_complete(mri)

    def _create_time_series(self, volume, title):
        # Now create TimeSeries and fill it with data from NIFTI image
        time_series = TimeSeriesVolume()
        time_series.title = title
        time_series.labels_ordering = ["Time", "X", "Y", "Z"]
        time_series.start_time = 0.0
        time_series.volume = volume

        if len(self.parser.zooms) > 3:
            time_series.sample_period = float(self.parser.zooms[3])
        else:
            # If no time dim, set sampling to 1 sec
            time_series.sample_period = 1

        if self.parser.units is not None and len(self.parser.units) > 1:
            time_series.sample_period_unit = self.parser.units[1]

        ts_h5_path = self.path_for(TimeSeriesVolumeH5, time_series.gid)
        nifti_data = self.parser.parse()
        with TimeSeriesVolumeH5(ts_h5_path) as ts_h5:
            ts_h5.store(time_series, scalars_only=True, store_references=True)
            for i in range(self.parser.time_dim_size):
                ts_h5.write_data_slice([nifti_data[:, :, :, i, ...]])
            data_shape = ts_h5.read_data_shape()

        ts_idx = TimeSeriesVolumeIndex()
        ts_idx.fill_from_has_traits(time_series)
        ts_idx.data_ndim = len(data_shape)
        ts_idx.data_length_1d, ts_idx.data_length_2d, ts_idx.data_length_3d, ts_idx.data_length_4d = prepare_array_shape_meta(
            data_shape)
        return ts_idx

    def _create_region_map(self, volume, connectivity, view_model):

        nifti_data = self.parser.parse()
        nifti_data = self._apply_corrections_and_mapping(nifti_data, connectivity.number_of_regions, view_model)
        rvm = RegionVolumeMapping()
        rvm.title = view_model.title
        rvm.dimensions_labels = ["X", "Y", "Z"]
        rvm.volume = volume
        rvm.connectivity = h5.load_from_index(connectivity)
        rvm.array_data = nifti_data

        return self.store_complete(rvm)

    def _apply_corrections_and_mapping(self, data, conn_nr_regions, view_model):

        self.log.info("Writing RegionVolumeMapping with min=%d, mix=%d" % (data.min(), data.max()))

        if view_model.mappings_file:
            try:
                mapping_data = numpy.loadtxt(view_model.mappings_file, dtype=numpy.str_, usecols=(0, 2))
                mapping_data = {int(row[0]): int(row[1]) for row in mapping_data}
            except Exception:
                raise ValidationException("Invalid Mapping File. Expected 3 columns (int, string, int)")

            if len(data.shape) != 3:
                raise ValidationException('Invalid RVM data. Expected 3D.')

            not_matched = set()
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    for k in range(data.shape[2]):
                        val = data[i][j][k]
                        if val not in mapping_data:
                            not_matched.add(val)
                        data[i][j][k] = mapping_data.get(val, -1)

            self.log.info("Imported RM with values in interval [%d - %d]" % (data.min(), data.max()))
            if not_matched:
                self.log.warning("Not matched regions will be considered background: %s" % not_matched)
                if not view_model.apply_corrections:
                    raise ValidationException("Not matched regions were identified %s" % not_matched)

        if view_model.one_based:
            data = data - 1

        if view_model.apply_corrections:
            data[data >= conn_nr_regions] = -1
            data[data < -1] = -1
            self.log.debug("After corrections: RegionVolumeMapping min=%d, mix=%d" % (data.min(), data.max()))

        if data.min() < -1 or data.max() >= conn_nr_regions:
            raise ValidationException("Invalid Mapping array: [%d ... %d]" % (data.min(), data.max()))

        return data

    @transactional
    def launch(self, view_model):
        # type: (NIFTIImporterModel) -> [VolumeIndex, StructuralMRIIndex, TimeSeriesVolumeIndex, RegionVolumeMappingIndex]
        """
        Execute import operations:
        """
        try:
            self.parser = NIFTIParser(view_model.data_file)
            volume_idx, volume_ht = self._create_volume()
            title = "NIFTI Import - " + os.path.split(view_model.data_file)[1]

            if view_model.connectivity is not None:
                connectivity_index = self.load_entity_by_gid(view_model.connectivity)
                rm = self._create_region_map(volume_ht, connectivity_index, view_model)
                return [volume_idx, rm]

            if self.parser.has_time_dimension:
                time_series = self._create_time_series(volume_ht, title)
                return [volume_idx, time_series]

            # no connectivity and no time
            mri = self._create_mri(volume_ht, title)
            return [volume_idx, mri]

        except ParseException as excep:
            logger = get_logger(__name__)
            logger.exception(excep)
            raise LaunchException(excep)
