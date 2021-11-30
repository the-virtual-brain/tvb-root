# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import numpy
from tvb.datatypes.sensors import SensorsEEG, SensorsMEG
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.datatypes import time_series


class TestTimeseries(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.time_series` module.
    """

    def test_timeseries(self):
        data = numpy.random.random((10, 10))
        dt = time_series.TimeSeries(data=data, title='meh if it counts')
        summary_info = dt.summary_info()
        assert summary_info['Dimensions'] == ('Time', 'State Variable', 'Space', 'Mode')
        assert summary_info['Length'] == 10.0
        assert summary_info['Sample period'] == 1.0
        assert summary_info['Time units'] == 'ms'
        assert summary_info['Time-series name'] == 'meh if it counts'
        assert summary_info['Time-series type'] == 'TimeSeries'
        assert dt.data.shape == (10, 10)
        assert dt.sample_period == 1.0
        assert dt.sample_rate == 1000
        assert dt.start_time == 0.0
        assert dt.time is None

    def test_timeserieseeg(self):
        data = numpy.random.random((10, 10))
        dt = time_series.TimeSeriesEEG(data=data, sensors=SensorsEEG())
        assert dt.data.shape == (10, 10)
        assert ("Time", "SV", "EEG Sensor", "Mode") == dt.labels_ordering
        assert dt.sample_period == 1.0
        assert dt.sample_rate == 1000
        assert dt.sensors is not None
        assert dt.start_time == 0.0
        assert dt.time is None

    def test_timeseriesmeg(self):
        data = numpy.random.random((10, 10))
        dt = time_series.TimeSeriesMEG(data=data, sensors=SensorsMEG(orientations=numpy.array([])))
        assert dt.data.shape == (10, 10)
        assert ("Time", "SV", "MEG Sensor", "Mode") == dt.labels_ordering
        assert dt.sample_period == 1.0
        assert dt.sample_rate == 1000
        assert dt.sensors is not None
        assert dt.start_time == 0.0
        assert dt.time is None

    def test_timeseriesregion(self):
        data = numpy.random.random((10, 10))
        dt = time_series.TimeSeriesRegion(data=data)
        assert dt.data.shape == (10, 10)
        assert dt.labels_ordering == ('Time', 'State Variable', 'Region', 'Mode')
        assert dt.sample_period == 1.0
        assert dt.sample_rate == 1000
        assert dt.start_time == 0.0
        assert dt.time is None

    def test_timeseriessurface(self):
        data = numpy.random.random((10, 10))
        dt = time_series.TimeSeriesSurface(data=data)
        assert dt.data.shape == (10, 10)
        assert dt.labels_ordering == ('Time', 'State Variable', 'Vertex', 'Mode')
        assert dt.sample_period == 1.0
        assert dt.sample_rate == 1000
        assert dt.start_time == 0.0
        assert dt.time is None

    def test_timeseriesvolume(self):
        data = numpy.random.random((10, 10))
        dt = time_series.TimeSeriesVolume(data=data)
        assert dt.data.shape == (10, 10)
        assert dt.labels_ordering == ('Time', 'X', 'Y', 'Z')
        assert dt.sample_period == 1.0
        assert dt.sample_rate == 1000
        assert dt.start_time == 0.0
        assert dt.time is None
