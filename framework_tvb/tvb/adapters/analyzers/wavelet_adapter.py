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
Adapter that uses the traits module to generate interfaces for
ContinuousWaveletTransform Analyzer.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""

import uuid
import numpy
from tvb.analyzers.wavelet import ContinuousWaveletTransform
from tvb.basic.neotraits.api import Range
from tvb.datatypes.time_series import TimeSeries
from tvb.datatypes.spectral import WaveletCoefficients
from tvb.core.adapters.abcadapter import ABCAsynchronous
from tvb.basic.filters.chain import FilterChain
from tvb.basic.logger.builder import get_logger

from tvb.core.entities.file.datatypes.spectral_h5 import WaveletCoefficientsH5
from tvb.core.entities.file.datatypes.time_series import TimeSeriesH5
from tvb.core.entities.model.datatypes.spectral import WaveletCoefficientsIndex
from tvb.core.neotraits.db import from_ndarray
from tvb.interfaces.neocom._h5loader import DirLoader

LOG = get_logger(__name__)


class ContinuousWaveletTransformAdapter(ABCAsynchronous):
    """
    TVB adapter for calling the ContinuousWaveletTransform algorithm.
    """

    _ui_name = "Continuous Wavelet Transform"
    _ui_description = "Compute Wavelet Tranformation for a TimeSeries input DataType."
    _ui_subsection = "wavelet"


    def get_input_tree(self):
        """
        Return a list of lists describing the interface to the analyzer. This
        is used by the GUI to generate the menus and fields necessary for
        defining current analysis.
        """
        algorithm = ContinuousWaveletTransform()
        algorithm.trait.bound = self.INTERFACE_ATTRIBUTES_ONLY
        tree = algorithm.interface[self.INTERFACE_ATTRIBUTES]
        for node in tree:
            if node['name'] == 'time_series':
                node['conditions'] = FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                                 operations=["=="], values=[4])
        return tree


    def get_output(self):
        return [WaveletCoefficients]


    def configure(self, time_series, mother=None, sample_period=None, normalisation=None, q_ratio=None,
                  frequencies='Range', frequencies_parameters=None):
        """
        Store the input shape to be later used to estimate memory usage. Also create the algorithm instance.
        """
        self.input_time_series_index = time_series
        self.input_shape = (self.input_time_series_index.data.length_1d,
                            self.input_time_series_index.data.length_2d,
                            self.input_time_series_index.data.length_3d,
                            self.input_time_series_index.data.length_4d)
        LOG.debug("Time series shape is %s" % str(self.input_shape))
        ##-------------------- Fill Algorithm for Analysis -------------------##
        algorithm = ContinuousWaveletTransform()
        if mother is not None:
            algorithm.mother = mother

        if sample_period is not None:
            algorithm.sample_period = sample_period

        if (frequencies_parameters is not None and 'lo' in frequencies_parameters
                and 'hi' in frequencies_parameters and frequencies_parameters['hi'] != frequencies_parameters['lo']):
            algorithm.frequencies = Range(**frequencies_parameters)

        if normalisation is not None:
            algorithm.normalisation = normalisation

        if q_ratio is not None:
            algorithm.q_ratio = q_ratio

        self.algorithm = algorithm

    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        used_shape = (self.input_shape[0],
                      self.input_shape[1],
                      1,
                      self.input_shape[3])
        input_size = numpy.prod(used_shape) * 8.0
        output_size = self.algorithm.result_size(used_shape, self.input_time_series_index.sample_period)
        return input_size + output_size

    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter.(in kB)
        """
        used_shape = (self.input_shape[0],
                      self.input_shape[1],
                      1,
                      self.input_shape[3])
        return self.array_size2kb(self.algorithm.result_size(used_shape, self.input_time_series_index.sample_period))

    def launch(self, time_series, mother=None, sample_period=None, normalisation=None, q_ratio=None,
               frequencies='Range', frequencies_parameters=None):
        """ 
        Launch algorithm and build results. 
        """
        ##--------- Prepare a WaveletCoefficients object for result ----------##
        frequencies_array = numpy.array([])
        if self.algorithm.frequencies is not None:
            frequencies_array = self.algorithm.frequencies.to_array()

        wavelet_index = WaveletCoefficientsIndex()
        gid = uuid.uuid4()  # wavelet_index.gid
        wavelet_index.gid = gid

        loader = DirLoader(self.storage_path)
        ts_gid = self.input_time_series_index.gid
        source_path = loader.path_for(TimeSeriesH5, ts_gid)
        time_series_h5 = TimeSeriesH5(path=source_path)

        dest_path = loader.path_for(WaveletCoefficientsH5, gid)

        wavelet_h5 = WaveletCoefficientsH5(path=dest_path)
        wavelet_h5.source.store(time_series_h5.gid.load())
        wavelet_h5.mother.store(self.algorithm.mother)
        wavelet_h5.q_ratio.store(self.algorithm.q_ratio)
        wavelet_h5.sample_period.store(self.algorithm.sample_period)
        wavelet_h5.frequencies.store(frequencies_array)
        wavelet_h5.normalisation.store(self.algorithm.normalisation)

        ##------------- NOTE: Assumes 4D, Simulator timeSeries. --------------##
        node_slice = [slice(self.input_shape[0]), slice(self.input_shape[1]), None, slice(self.input_shape[3])]

        ##---------- Iterate over slices and compose final result ------------##
        small_ts = TimeSeries()
        small_ts.sample_rate = time_series_h5.sample_rate.load()
        small_ts.sample_period = time_series_h5.sample_period.load()
        for node in range(self.input_shape[2]):
            node_slice[2] = slice(node, node + 1)
            small_ts.data = time_series_h5.read_data_slice(tuple(node_slice))
            self.algorithm.time_series = small_ts
            partial_wavelet = self.algorithm.evaluate()
            wavelet_h5.write_data_slice(partial_wavelet)

        wavelet_h5.close()

        wavelet_index.source = self.input_time_series_index
        wavelet_index.mother = self.algorithm.mother
        wavelet_index.normalisation = self.algorithm.normalisation
        wavelet_index.q_ratio = self.algorithm.q_ratio
        wavelet_index.sample_period = self.algorithm.sample_period
        wavelet_index.number_of_scales = frequencies_array.shape[0]
        wavelet_index.frequencies_min, wavelet_index.frequencies_max, _ = from_ndarray(frequencies_array)

        return wavelet_index
