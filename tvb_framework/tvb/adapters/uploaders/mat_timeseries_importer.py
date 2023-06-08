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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import json
import uuid
import numpy
from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex
from tvb.adapters.uploaders.mat.parser import read_nested_mat_file
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesRegionH5, TimeSeriesEEGH5
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex, TimeSeriesEEGIndex
from tvb.basic.neotraits.api import Attr, Int
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str, DataTypeGidAttr
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.core.entities.storage import transactional, dao
from tvb.core.adapters.arguments_serialisation import parse_slice
from tvb.core.neotraits.forms import TraitUploadField, StrField, BoolField, IntField, TraitDataTypeSelectField
from tvb.core.neotraits.db import prepare_array_shape_meta
from tvb.core.neocom import h5
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.time_series import TimeSeriesRegion, TimeSeriesEEG

TS_REGION = "Region"
TS_EEG = "EEG"


class RegionMatTimeSeriesImporterModel(UploaderViewModel):
    data_file = Str(
        label='Please select file to import'
    )

    dataset_name = Str(
        label='Matlab dataset name',
        doc='Name of the MATLAB dataset where data is stored'
    )

    structure_path = Str(
        required=False,
        default='',
        label='For nested structures enter the field path (separated by .)'
    )

    transpose = Attr(
        field_type=bool,
        required=False,
        default=False,
        label='Transpose the array. Expected shape is (time, channel)'
    )

    slice = Str(
        required=False,
        default='',
        label='Slice of the array in numpy syntax. Expected shape is (time, channel)'
    )

    sampling_rate = Int(
        required=False,
        default=100,
        label='sampling rate (Hz)'
    )

    start_time = Int(
        default=0,
        label='starting time (ms)'
    )

    datatype = DataTypeGidAttr(
        linked_datatype=Connectivity,
        label='Connectivity'
    )


class RegionMatTimeSeriesImporterForm(ABCUploaderForm):

    def __init__(self):
        super(RegionMatTimeSeriesImporterForm, self).__init__()
        self.data_file = TraitUploadField(RegionMatTimeSeriesImporterModel.data_file, '.mat', 'data_file')
        self.dataset_name = StrField(RegionMatTimeSeriesImporterModel.dataset_name, name='dataset_name')
        self.structure_path = StrField(RegionMatTimeSeriesImporterModel.structure_path, name='structure_path')
        self.transpose = BoolField(RegionMatTimeSeriesImporterModel.transpose, name='transpose')
        self.slice = StrField(RegionMatTimeSeriesImporterModel.slice, name='slice')
        self.sampling_rate = IntField(RegionMatTimeSeriesImporterModel.sampling_rate, name='sampling_rate')
        self.start_time = IntField(RegionMatTimeSeriesImporterModel.start_time, name='start_time')
        self.datatype = TraitDataTypeSelectField(RegionMatTimeSeriesImporterModel.datatype, name='tstype_parameters')

    @staticmethod
    def get_view_model():
        return RegionMatTimeSeriesImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'data_file': '.mat'
        }


class RegionTimeSeriesImporter(ABCUploader):
    """
    Import time series from a .mat file.
    """
    _ui_name = "TimeSeries Region MAT"
    _ui_subsection = "mat_ts_importer"
    _ui_description = "Import time series from a .mat file."
    tstype = TS_REGION

    def get_form_class(self):
        return RegionMatTimeSeriesImporterForm

    def get_output(self):
        return [TimeSeriesRegionIndex, TimeSeriesEEGIndex]

    def create_region_ts(self, data_shape, connectivity):
        if connectivity.number_of_regions != data_shape[1]:
            raise LaunchException("Data has %d channels but the connectivity has %d nodes"
                                  % (data_shape[1], connectivity.number_of_regions))
        ts_idx = TimeSeriesRegionIndex()
        ts_idx.fk_connectivity_gid = connectivity.gid

        region_map_indexes = dao.get_generic_entity(RegionMappingIndex, connectivity.gid,
                                                    'fk_connectivity_gid')
        ts_idx.has_surface_mapping = False
        if len(region_map_indexes) > 0:
            ts_idx.fk_region_mapping_gid = region_map_indexes[0].gid
            ts_idx.has_surface_mapping = True

        ts_h5_path = self.path_for(TimeSeriesRegionH5, ts_idx.gid)
        ts_h5 = TimeSeriesRegionH5(ts_h5_path)
        ts_h5.connectivity.store(uuid.UUID(connectivity.gid))

        return TimeSeriesRegion(), ts_idx, ts_h5

    def create_eeg_ts(self, data_shape, sensors):
        if sensors.number_of_sensors != data_shape[1]:
            raise LaunchException("Data has %d channels but the sensors have %d"
                                  % (data_shape[1], sensors.number_of_sensors))

        ts_idx = TimeSeriesEEGIndex()
        ts_idx.fk_sensors_gid = sensors.gid

        ts_h5_path = self.path_for(TimeSeriesEEGH5, ts_idx.gid)
        ts_h5 = TimeSeriesEEGH5(ts_h5_path)
        ts_h5.sensors.store(uuid.UUID(sensors.gid))

        return TimeSeriesEEG(), ts_idx, ts_h5

    ts_builder = {TS_REGION: create_region_ts, TS_EEG: create_eeg_ts}

    @transactional
    def launch(self, view_model):
        # type: (RegionMatTimeSeriesImporterModel) -> [TimeSeriesRegionIndex, TimeSeriesEEGIndex]

        try:
            data = read_nested_mat_file(view_model.data_file, view_model.dataset_name, view_model.structure_path)

            if view_model.transpose:
                data = data.T
            if view_model.slice:
                data = data[parse_slice(view_model.slice)]

            datatype_index = self.load_entity_by_gid(view_model.datatype)
            ts, ts_idx, ts_h5 = self.ts_builder[self.tstype](self, data.shape, datatype_index)

            ts.start_time = view_model.start_time
            ts.sample_period_unit = 's'

            ts_h5.write_time_slice(numpy.r_[:data.shape[0]] * ts.sample_period)
            # we expect empirical data shape to be time, channel.
            # But tvb expects time, state, channel, mode. Introduce those dimensions
            ts_h5.write_data_slice(data[:, numpy.newaxis, :, numpy.newaxis])

            data_shape = ts_h5.read_data_shape()
            ts_h5.nr_dimensions.store(len(data_shape))
            ts_h5.gid.store(uuid.UUID(ts_idx.gid))
            ts_h5.sample_period.store(ts.sample_period)
            ts_h5.sample_period_unit.store(ts.sample_period_unit)
            ts_h5.sample_rate.store(ts.sample_rate)
            ts_h5.start_time.store(ts.start_time)
            ts_h5.labels_ordering.store(ts.labels_ordering)
            ts_h5.labels_dimensions.store(ts.labels_dimensions)
            ts_h5.title.store(ts.title)
            ts_h5.close()

            ts_idx.title = ts.title
            ts_idx.time_series_type = type(ts).__name__
            ts_idx.sample_period_unit = ts.sample_period_unit
            ts_idx.sample_period = ts.sample_period
            ts_idx.sample_rate = ts.sample_rate
            ts_idx.labels_dimensions = json.dumps(ts.labels_dimensions)
            ts_idx.labels_ordering = json.dumps(ts.labels_ordering)
            ts_idx.data_ndim = len(data_shape)
            ts_idx.data_length_1d, ts_idx.data_length_2d, ts_idx.data_length_3d, ts_idx.data_length_4d = prepare_array_shape_meta(
                data_shape)

            return ts_idx
        except ParseException as ex:
            self.log.exception(ex)
            raise LaunchException(ex)
