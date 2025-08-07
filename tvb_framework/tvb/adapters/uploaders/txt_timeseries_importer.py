# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Paula Prodan <paula.prodan@codemart.ro>
"""

import json
import uuid
import numpy

from tvb.basic.neotraits.api import Int
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.time_series import TimeSeriesRegion

from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesRegionH5
from tvb.adapters.uploaders.mat_timeseries_importer import TS_REGION, RegionTimeSeriesImporter
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str, DataTypeGidAttr
from tvb.core.adapters.abcuploader import ABCUploaderForm, ABCUploader
from tvb.core.entities.storage import dao
from tvb.core.entities.storage.session_maker import transactional
from tvb.core.neotraits.db import prepare_array_shape_meta
from tvb.core.neotraits.forms import TraitUploadField, IntField, TraitDataTypeSelectField


class TxtTimeseriesImporterModel(UploaderViewModel):
    data_file = Str(
        label='Please select file to import'
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


class TxtTimeseriesImporterForm(ABCUploaderForm):

    def __init__(self):
        super(TxtTimeseriesImporterForm, self).__init__()
        self.data_file = TraitUploadField(TxtTimeseriesImporterModel.data_file, '.txt', 'data_file')
        self.sampling_rate = IntField(TxtTimeseriesImporterModel.sampling_rate, name='sampling_rate')
        self.start_time = IntField(TxtTimeseriesImporterModel.start_time, name='start_time')
        self.datatype = TraitDataTypeSelectField(TxtTimeseriesImporterModel.datatype, name='tstype_parameters')

    @staticmethod
    def get_view_model():
        return TxtTimeseriesImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'data_file': '.txt'
        }


class TxtTimeseriesImporter(RegionTimeSeriesImporter):
    """
    Import time series from a .mat file.
    """
    _ui_name = "TimeSeries Region TXT"
    _ui_subsection = "txt_ts_importer"
    _ui_description = "Import time series from a .txt file."
    tstype = TS_REGION

    def get_form_class(self):
        return TxtTimeseriesImporterForm

    def get_output(self):
        return [TimeSeriesRegionIndex]

    @transactional
    def launch(self, view_model):
        # type: (TxtTimeseriesImporterModel) -> TimeSeriesRegionIndex

        # Read first row to compute the nr of regions
        with open(view_model.data_file, 'r') as file:
            for line in file:
                row = line.strip().split()
                break

        data_shape = (0, len(row))

        datatype_index = self.load_entity_by_gid(view_model.datatype)
        ts, ts_idx, ts_h5 = self.create_region_ts(data_shape, datatype_index)
        ts.start_time = view_model.start_time
        ts.sample_period_unit = 's'

        # Read all rows from txt and write them to H5
        with open(view_model.data_file, 'r') as file:
            nr_of_points = 0
            for line in file:
                row = line.strip().split()
                nr_of_points += 1
                row_floats = numpy.array(row, dtype=float)
                ts_h5.write_data_slice(row_floats[numpy.newaxis, numpy.newaxis, :, numpy.newaxis])
        ts_h5.write_time_slice(numpy.r_[:nr_of_points] * ts.sample_period)

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
