# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env
    setup_test_console_env()
    
import numpy
import unittest
from tvb.datatypes import time_series
from tvb.tests.library.base_testcase import BaseTestCase


class TimeseriesTest(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.time_series` module.
    """
    
    def test_timeseries(self):
        data = numpy.random.random((10, 10))
        dt = time_series.TimeSeries(data=data)
        summary_info = dt.summary_info
        self.assertEqual(summary_info['Dimensions'], ['Time', 'State Variable', 'Space', 'Mode'])
        self.assertEqual(summary_info['Length'], 10.0)
        self.assertEqual(summary_info['Sample period'], 1.0)
        self.assertEqual(summary_info['Time units'], 'ms')
        self.assertEqual(summary_info['Time-series name'], '')
        self.assertEqual(summary_info['Time-series type'], 'TimeSeries') 
        self.assertEqual(dt.data.shape, (10, 10))
        self.assertEqual(dt.sample_period, 1.0)
        self.assertEqual(dt.sample_rate, 0.0)
        self.assertEqual(dt.start_time, 0.0)
        self.assertEqual(dt.time.shape, (0,))
        
        
    def test_timeserieseeg(self):
        data = numpy.random.random((10, 10))
        dt = time_series.TimeSeriesEEG(data=data)
        self.assertEqual(dt.data.shape, (10, 10))
        self.assertEqual(['Time', '1', 'EEG Sensor', '1'], dt.labels_ordering)
        self.assertEqual(dt.sample_period, 1.0)
        self.assertEqual(dt.sample_rate, 0.0)
        self.assertTrue(dt.sensors is None)
        self.assertEqual(dt.start_time, 0.0)
        self.assertEqual(dt.time.shape, (0,))
        
        
    def test_timeseriesmeg(self):
        data = numpy.random.random((10, 10))
        dt = time_series.TimeSeriesMEG(data=data)
        self.assertEqual(dt.data.shape, (10, 10))
        self.assertEqual(['Time', '1', 'MEG Sensor', '1'], dt.labels_ordering)
        self.assertEqual(dt.sample_period, 1.0)
        self.assertEqual(dt.sample_rate, 0.0)
        self.assertTrue(dt.sensors is None)
        self.assertEqual(dt.start_time, 0.0)
        self.assertEqual(dt.time.shape, (0,))
        
        
    def test_timeseriesregion(self):
        data = numpy.random.random((10, 10))
        dt = time_series.TimeSeriesRegion(data=data)
        self.assertEqual(dt.data.shape, (10, 10))
        self.assertEqual(dt.labels_ordering, ['Time', 'State Variable', 'Region', 'Mode'])
        self.assertEqual(dt.sample_period, 1.0)
        self.assertEqual(dt.sample_rate, 0.0)
        self.assertEqual(dt.start_time, 0.0)
        self.assertEqual(dt.time.shape, (0,))
        
        
    def test_timeseriessurface(self):
        data = numpy.random.random((10, 10))
        dt = time_series.TimeSeriesSurface(data=data)
        self.assertEqual(dt.data.shape, (10, 10))
        self.assertEqual(dt.labels_ordering, ['Time', 'State Variable', 'Vertex', 'Mode'])
        self.assertEqual(dt.sample_period, 1.0)
        self.assertEqual(dt.sample_rate, 0.0)
        self.assertEqual(dt.start_time, 0.0)
        self.assertEqual(dt.time.shape, (0,))
        
        
    def test_timeseriesvolume(self):
        data = numpy.random.random((10, 10))
        dt = time_series.TimeSeriesVolume(data=data)
        self.assertEqual(dt.data.shape, (10, 10))
        self.assertEqual(dt.labels_ordering, ['Time', 'X', 'Y', 'Z'])
        self.assertEqual(dt.sample_period, 1.0)
        self.assertEqual(dt.sample_rate, 0.0)
        self.assertEqual(dt.start_time, 0.0)
        self.assertEqual(dt.time.shape, (0,))  
        
        
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TimeseriesTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE) 