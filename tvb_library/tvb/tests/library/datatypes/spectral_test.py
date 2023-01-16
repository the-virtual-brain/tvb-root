# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
Created on Mar 20, 2013

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import numpy
from tvb.datatypes.spectral import WindowingFunctionsEnum
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.datatypes import spectral, time_series


class TestSpectral(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.spectral` module.
    """

    def test_fourierspectrum(self):
        data = numpy.random.random((10, 10))
        ts = time_series.TimeSeries(data=data, title='meh')
        dt = spectral.FourierSpectrum(source=ts,
                                      segment_length=100, array_data=numpy.array([]),
                                      windowing_function=WindowingFunctionsEnum.HAMMING)
        summary_info = dt.summary_info()
        assert summary_info['Frequency step'] == 0.01
        assert summary_info['Maximum frequency'] == 0.5
        assert summary_info['Segment length'] == 100
        assert summary_info['Windowing function'] is not None
        assert summary_info['Source'] == 'meh'
        assert summary_info['Spectral type'] == 'FourierSpectrum'
        assert dt.normalised_average_power is None
        assert dt.segment_length == 100.0
        assert dt.array_data is not None
        assert dt.source is not None
        assert dt.windowing_function is not None

    def test_waveletcoefficients(self):
        data = numpy.random.random((10, 10))
        ts = time_series.TimeSeries(data=data)
        dt = spectral.WaveletCoefficients(source=ts,
                                          mother='morlet',
                                          sample_period=7.8125,
                                          frequencies=numpy.array([0.008, 0.028, 0.048, 0.068]),
                                          normalisation="energy",
                                          q_ratio=5.0,
                                          array_data=numpy.random.random((10, 10)), )
        # dt.configure()
        summary_info = dt.summary_info()
        assert summary_info['Maximum frequency'] == 0.068
        assert summary_info['Minimum frequency'] == 0.008
        assert summary_info['Normalisation'], 'energy'
        assert summary_info['Number of scales'] == 4
        assert summary_info['Q-ratio'] == 5.0
        assert summary_info['Sample period'] == 7.8125
        assert summary_info['Spectral type'] == 'WaveletCoefficients'
        assert summary_info['Wavelet type'] == 'morlet'
        assert dt.q_ratio == 5.0
        assert dt.sample_period == 7.8125
        assert dt.array_data.shape == (10, 10)
        assert dt.source is not None

    def test_coherencespectrum(self):
        data = numpy.random.random((10, 10))
        ts = time_series.TimeSeries(data=data, title='meh')
        dt = spectral.CoherenceSpectrum(source=ts,
                                        nfft=4,
                                        array_data=numpy.random.random((10, 10)),
                                        frequency=numpy.random.random((10,)))
        summary_info = dt.summary_info()
        assert summary_info['Number of frequencies'] == 10
        assert summary_info['Spectral type'] == 'CoherenceSpectrum'
        assert summary_info['FFT length (time-points)'] == 4
        assert summary_info['Source'] == 'meh'
        assert dt.nfft == 4
        assert dt.array_data.shape == (10, 10)
        assert dt.source is not None

    def test_complexcoherence(self):
        data = numpy.random.random((10, 10))
        ts = time_series.TimeSeries(data=data, title='meh')
        dt = spectral.ComplexCoherenceSpectrum(source=ts,
                                               windowing_function=str(''),
                                               array_data=numpy.random.random((10, 10)),
                                               cross_spectrum=numpy.random.random((10, 10)),
                                               epoch_length=10,
                                               segment_length=5)
        summary_info = dt.summary_info()
        assert summary_info['Frequency step'] == 0.2
        assert summary_info['Maximum frequency'] == 0.5
        assert summary_info['Source'] == 'meh'
        assert summary_info['Spectral type'] == 'ComplexCoherenceSpectrum'
        assert dt.epoch_length == 10
        assert dt.segment_length == 5
        assert dt.array_data.shape, (10 == 10)
        assert dt.source is not None
        assert dt.windowing_function is not None
