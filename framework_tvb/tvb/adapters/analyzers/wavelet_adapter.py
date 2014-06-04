# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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

import numpy
from tvb.basic.config.settings import TVBSettings
from tvb.analyzers.wavelet import ContinuousWaveletTransform
from tvb.datatypes.time_series import TimeSeries
from tvb.datatypes.spectral import WaveletCoefficients
from tvb.core.adapters.abcadapter import ABCAsynchronous
from tvb.basic.traits.types_basic import Range
from tvb.basic.traits.util import log_debug_array
from tvb.basic.filters.chain import FilterChain
from tvb.basic.logger.builder import get_logger

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
        self.input_shape = time_series.read_data_shape()
        log_debug_array(LOG, time_series, "time_series")
        
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
        self.algorithm.time_series = time_series


    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        used_shape = (self.input_shape[0], self.input_shape[1], 1, self.input_shape[3])
        input_size = numpy.prod(used_shape) * 8.0
        output_size = self.algorithm.result_size(used_shape)
        return input_size + output_size   


    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter.(in kB)
        """
        used_shape = (self.input_shape[0], self.input_shape[1], 1, self.input_shape[3])
        return self.algorithm.result_size(used_shape) * TVBSettings.MAGIC_NUMBER / 8 / 2 ** 10


    def launch(self, time_series, mother=None, sample_period=None, normalisation=None, q_ratio=None,
               frequencies='Range', frequencies_parameters=None):
        """ 
        Launch algorithm and build results. 
        """
        ##--------- Prepare a WaveletCoefficients object for result ----------##
        frequencies_array = numpy.array([])
        if self.algorithm.frequencies is not None:
            frequencies_array = numpy.array(list(self.algorithm.frequencies))
        wavelet = WaveletCoefficients(source=time_series, mother=self.algorithm.mother, q_ratio=self.algorithm.q_ratio,
                                      sample_period=self.algorithm.sample_period, frequencies=frequencies_array,
                                      normalisation=self.algorithm.normalisation, storage_path=self.storage_path)
        
        ##------------- NOTE: Assumes 4D, Simulator timeSeries. --------------##
        node_slice = [slice(self.input_shape[0]), slice(self.input_shape[1]), None, slice(self.input_shape[3])]
        
        ##---------- Iterate over slices and compose final result ------------##
        small_ts = TimeSeries(use_storage=False)
        small_ts.sample_rate = time_series.sample_rate
        small_ts.sample_period = time_series.sample_period
        for node in range(self.input_shape[2]):
            node_slice[2] = slice(node, node + 1)
            small_ts.data = time_series.read_data_slice(tuple(node_slice))
            self.algorithm.time_series = small_ts
            partial_wavelet = self.algorithm.evaluate()
            wavelet.write_data_slice(partial_wavelet)
        
        wavelet.close_file()
        return wavelet


