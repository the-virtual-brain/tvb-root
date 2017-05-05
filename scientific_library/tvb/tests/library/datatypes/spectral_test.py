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
Created on Mar 20, 2013

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env
    setup_test_console_env()

import numpy
import unittest

from tvb.datatypes import spectral, time_series
from tvb.tests.library.base_testcase import BaseTestCase
        
class SpectralTest(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.spectral` module.
    """
    
    def test_fourierspectrum(self):
        data = numpy.random.random((10, 10))
        ts = time_series.TimeSeries(data=data)
        dt = spectral.FourierSpectrum(source=ts,
                                      segment_length=100)
        dt.configure()
        summary_info = dt.summary_info
        self.assertEqual(summary_info['Frequency step'], 0.01)     
        self.assertEqual(summary_info['Maximum frequency'], 0.5)  
        self.assertEqual(summary_info['Segment length'], 100)  
        self.assertEqual(summary_info['Windowing function'], '')  
        self.assertEqual(summary_info['Source'], '')  
        self.assertEqual(summary_info['Spectral type'], 'FourierSpectrum')  
        self.assertTrue(dt.aggregation_functions is None)
        self.assertEqual(dt.normalised_average_power.shape, (0, ))
        self.assertEqual(dt.segment_length, 100.0)
        self.assertEqual(dt.shape, (0, ))
        self.assertTrue(dt.source is not None)
        self.assertEqual(dt.windowing_function, '')
        
        
    def test_waveletcoefficients(self):
        data = numpy.random.random((10, 10))
        ts = time_series.TimeSeries(data=data)
        dt = spectral.WaveletCoefficients(source=ts,
                                          mother='morlet',
                                          sample_period=7.8125,
                                          frequencies=[0.008, 0.028, 0.048, 0.068],
                                          normalisation="energy",
                                          q_ratio=5.0,
                                          array_data=numpy.random.random((10, 10)),)
        dt.configure()
        summary_info = dt.summary_info
        self.assertEqual(summary_info['Maximum frequency'], 0.068)  
        self.assertEqual(summary_info['Minimum frequency'], 0.008)  
        self.assertEqual(summary_info['Normalisation'], 'energy')  
        self.assertEqual(summary_info['Number of scales'], 4)
        self.assertEqual(summary_info['Q-ratio'], 5.0)  
        self.assertEqual(summary_info['Sample period'], 7.8125) 
        self.assertEqual(summary_info['Spectral type'], 'WaveletCoefficients')  
        self.assertEqual(summary_info['Wavelet type'], 'morlet')     
        self.assertEqual(dt.q_ratio, 5.0)
        self.assertEqual(dt.sample_period, 7.8125)
        self.assertEqual(dt.shape, (10, 10))
        self.assertTrue(dt.source is not None)
        
        
    def test_coherencespectrum(self):
        data = numpy.random.random((10, 10))
        ts = time_series.TimeSeries(data=data)
        dt = spectral.CoherenceSpectrum(source=ts,
                                        nfft = 4,
                                        array_data = numpy.random.random((10, 10)),
                                        frequency = numpy.random.random((10,)))
        summary_info = dt.summary_info 
        self.assertEqual(summary_info['Number of frequencies'], 10)
        self.assertEqual(summary_info['Spectral type'], 'CoherenceSpectrum')
        self.assertEqual(summary_info['FFT length (time-points)'], 4)
        self.assertEqual(summary_info['Source'], '')
        self.assertEqual(dt.nfft, 4)
        self.assertEqual(dt.shape, (10, 10))
        self.assertTrue(dt.source is not None)
        
        
    def test_complexcoherence(self):
        data = numpy.random.random((10, 10))
        ts = time_series.TimeSeries(data=data)
        dt = spectral.ComplexCoherenceSpectrum(source=ts,
                                               array_data = numpy.random.random((10, 10)),
                                               cross_spectrum = numpy.random.random((10, 10)),
                                               epoch_length = 10,
                                               segment_length = 5)
        summary_info = dt.summary_info
        self.assertEqual(summary_info['Frequency step'], 0.2)     
        self.assertEqual(summary_info['Maximum frequency'], 0.5)  
        self.assertEqual(summary_info['Source'], '')  
        self.assertEqual(summary_info['Spectral type'], 'ComplexCoherenceSpectrum')     
        self.assertTrue(dt.aggregation_functions is None)
        self.assertEqual(dt.epoch_length, 10)
        self.assertEqual(dt.segment_length, 5)
        self.assertEqual(dt.shape, (10, 10))
        self.assertTrue(dt.source is not None)
        self.assertEqual(dt.windowing_function, '')
        
        
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(SpectralTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE) 