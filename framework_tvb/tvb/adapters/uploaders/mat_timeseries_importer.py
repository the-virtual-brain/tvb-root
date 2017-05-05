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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import numpy
from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.adapters.uploaders.mat.parser import read_nested_mat_file
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.core.entities.storage import transactional
from tvb.basic.arguments_serialisation import parse_slice
from tvb.datatypes.time_series import TimeSeriesRegion, TimeSeriesEEG


class MatTimeSeriesImporter(ABCUploader):
    """
    Import time series from a .mat file.
    """
    _ui_name = "Timeseries MAT"
    _ui_subsection = "mat_ts_importer"
    _ui_description = "Import time series from a .mat file."

    TS_REGION = 'region'
    TS_EEG = 'EEG'

    def get_upload_input_tree(self):
        return [
            {'name': 'data_file', 'type': 'upload', 'required_type': '.mat',
             'label': 'Please select file to import', 'required': True},

            {'name': 'dataset_name', 'type': 'str', 'required': True,
             'label': 'Matlab dataset name', 'description': 'Name of the MATLAB dataset where data is stored'},

            {'name': 'structure_path', 'type': 'str', 'default': '',
             'label': 'For nested structures enter the field path (separated by .)'},

            {'name': 'transpose', 'type': 'bool', 'default': False,
             'label': 'Transpose the array. Expected shape is (time, channel)'},

            {'name': 'slice', 'type': 'str', 'default': '',
             'label': 'Slice of the array in numpy syntax. Expected shape is (time, channel)'},

            {'name': 'tstype', 'type': 'select', 'required': True,
             'label': 'time series type',
             'options': [{'name': self.TS_REGION, 'value': self.TS_REGION,
                          'attributes': [{'name': 'connectivity', 'required': True, 'label': 'Connectivity',
                                          'type': 'tvb.datatypes.connectivity.Connectivity'}]},
                         {'name': self.TS_EEG, 'value': self.TS_EEG,
                          'attributes': [{'name': 'sensors', 'required': True, 'label': 'EEG Sensors',
                                          'type': 'tvb.datatypes.sensors.SensorsEEG'}]}
                         ]},

            {'name': 'sampling_rate', 'type': 'int', 'default': 1000,
             'label': 'sampling rate (Hz)'},

            {'name': 'start_time', 'type': 'int', 'default': 0,
             'label': 'starting time (ms)'},
        ]


    def get_output(self):
        return [TimeSeriesRegion, TimeSeriesEEG]


    def create_region_ts(self, data, connectivity):
        if connectivity.number_of_regions != data.shape[1]:
            raise LaunchException("Data has %d channels but the connectivity has %d nodes"
                                  % (data.shape[1], connectivity.number_of_regions))
        return TimeSeriesRegion(storage_path=self.storage_path, connectivity=connectivity)


    def create_eeg_ts(self, data, sensors):
        if sensors.number_of_sensors != data.shape[1]:
            raise LaunchException("Data has %d channels but the sensors have %d"
                                  % (data.shape[1], sensors.number_of_sensors))
        return TimeSeriesEEG(storage_path=self.storage_path, sensors=sensors)


    ts_builder = {TS_REGION: create_region_ts, TS_EEG: create_eeg_ts}


    @transactional
    def launch(self, data_file, dataset_name, structure_path='',
               transpose=False, slice=None, sampling_rate=1000,
               start_time=0, tstype=None, tstype_parameters=None):
        try:
            data = read_nested_mat_file(data_file, dataset_name, structure_path)

            if transpose:
                data = data.T
            if slice:
                data = data[parse_slice(slice)]

            ts = self.ts_builder[tstype](self, data, **tstype_parameters)

            ts.start_time = start_time
            ts.sample_period = 1.0 / sampling_rate
            ts.sample_period_unit = 's'
            ts.write_time_slice(numpy.r_[:data.shape[0]] * ts.sample_period)
            # we expect empirical data shape to be time, channel.
            # But tvb expects time, state, channel, mode. Introduce those dimensions
            ts.write_data_slice(data[:, numpy.newaxis, :, numpy.newaxis])
            ts.close_file()

            return ts
        except ParseException as ex:
            self.log.exception(ex)
            raise LaunchException(ex)

