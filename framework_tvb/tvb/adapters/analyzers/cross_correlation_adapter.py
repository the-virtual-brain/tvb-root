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
Adapter that uses the traits module to generate interfaces for ... Analyzer.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""

import numpy
from tvb.core.adapters.abcadapter import ABCAsynchronous
from tvb.core.adapters.exceptions import LaunchException
from tvb.basic.config.settings import TVBSettings
from tvb.basic.logger.builder import get_logger
from tvb.basic.filters.chain import FilterChain
from tvb.basic.traits.util import log_debug_array
from tvb.datatypes.time_series import TimeSeries, TimeSeriesEEG, TimeSeriesMEG, TimeSeriesSEEG
from tvb.datatypes.temporal_correlations import CrossCorrelation
from tvb.datatypes.graph import CorrelationCoefficients
from tvb.analyzers.cross_correlation import CrossCorrelate
from tvb.analyzers.correlation_coefficient import CorrelationCoefficient

LOG = get_logger(__name__)


class CrossCorrelateAdapter(ABCAsynchronous):
    """ TVB adapter for calling the CrossCorrelate algorithm. """
    
    _ui_name = "Cross-correlation of nodes"
    _ui_description = "Cross-correlate two one-dimensional arrays."
    _ui_subsection = "crosscorr"


    def get_input_tree(self):
        """
        Return a list of lists describing the interface to the analyzer. This
        is used by the GUI to generate the menus and fields necessary for
        defining a simulation.
        """
        algorithm = CrossCorrelate()
        algorithm.trait.bound = self.INTERFACE_ATTRIBUTES_ONLY
        tree = algorithm.interface[self.INTERFACE_ATTRIBUTES]
        tree[0]['conditions'] = FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                            operations=["=="], values=[4])
        return tree
    
    
    def get_output(self):
        return [CrossCorrelation]


    def configure(self, time_series):
        """
        Store the input shape to be later used to estimate memory usage. Also create the algorithm instance.

        :param time_series: the input time-series for which cross correlation should be computed
        """
        self.input_shape = time_series.read_data_shape()
        log_debug_array(LOG, time_series, "time_series")
        
        ##-------------------- Fill Algorithm for Analysis -------------------##
        self.algorithm = CrossCorrelate()
        
    
    def get_required_memory_size(self, **kwargs):
        """
        Returns the required memory to be able to run the adapter.
        """
        #Not all the data is loaded into memory at one time here.
        used_shape = (self.input_shape[0], 1, self.input_shape[2], self.input_shape[3])
        input_size = numpy.prod(used_shape) * 8.0
        output_size = self.algorithm.result_size(used_shape)
        return input_size + output_size
    
    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter (in kB).
        """
        used_shape = (self.input_shape[0], 1, self.input_shape[2], self.input_shape[3])
        return self.algorithm.result_size(used_shape) * TVBSettings.MAGIC_NUMBER / 8 / 2 ** 10
    
    def launch(self, time_series):
        """ 
        Launch algorithm and build results.

        :param time_series: the input time series for which the correlation should be computed
        :returns: the cross correlation for the given time series
        :rtype: `CrossCorrelation`
        """
        ##--------- Prepare a CrossCorrelation object for result ------------##
        cross_corr = CrossCorrelation(source=time_series,
                                      storage_path=self.storage_path)
        
        node_slice = [slice(self.input_shape[0]), None, slice(self.input_shape[2]), slice(self.input_shape[3])]
        ##---------- Iterate over slices and compose final result ------------##
        small_ts = TimeSeries(use_storage=False)
        small_ts.sample_period = time_series.sample_period
        partial_cross_corr = None
        for var in range(self.input_shape[1]):
            node_slice[1] = slice(var, var + 1)
            small_ts.data = time_series.read_data_slice(tuple(node_slice))
            self.algorithm.time_series = small_ts
            partial_cross_corr = self.algorithm.evaluate()
            cross_corr.write_data_slice(partial_cross_corr)
        cross_corr.time = partial_cross_corr.time
        cross_corr.labels_ordering[1] = time_series.labels_ordering[2]
        cross_corr.labels_ordering[2] = time_series.labels_ordering[2]
        cross_corr.close_file()
        return cross_corr


class PearsonCorrelationCoefficientAdapter(ABCAsynchronous):
    """ TVB adapter for calling the Pearson CrossCorrelation algorithm. """

    _ui_name = "Pearson correlation coefficients"
    _ui_description = "Cross Correlation"
    _ui_subsection = "ccpearson"


    def get_input_tree(self):
        """
        Return a list of lists describing the interface to the analyzer. This
        is used by the GUI to generate the menus and fields necessary for
        defining a simulation.
        """
        algorithm = CorrelationCoefficient()
        algorithm.trait.bound = self.INTERFACE_ATTRIBUTES_ONLY
        tree = algorithm.interface[self.INTERFACE_ATTRIBUTES]
        tree[0]['conditions'] = FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                            operations=["=="], values=[4])
        return tree


    def get_output(self):
        return [CorrelationCoefficients]


    def configure(self, time_series, t_start, t_end):
        """
        Store the input shape to be later used to estimate memory usage. Also create the algorithm instance.

        :param time_series: the input time-series for which correlation coefficient should be computed
        :param t_start: the physical time interval start for the analysis
        :param t_end: physical time, interval end
        """
        if t_start >= t_end or t_start < 0:
            raise LaunchException("Can not launch operation without monitors selected !!!")

        shape_tuple = time_series.read_data_shape()
        self.input_shape = [shape_tuple[0], shape_tuple[1], shape_tuple[2], shape_tuple[3]]
        self.input_shape[0] = int((t_end - t_start) / time_series.sample_period)
        log_debug_array(LOG, time_series, "time_series")

        self.algorithm = CorrelationCoefficient(time_series=time_series, t_start=t_start, t_end=t_end)


    def get_required_memory_size(self, **kwargs):
        """
        Returns the required memory to be able to run this adapter.
        """
        in_memory_input = [self.input_shape[0], 1, self.input_shape[2], 1]
        input_size = numpy.prod(in_memory_input) * 8.0
        output_size = self.algorithm.result_size(self.input_shape)
        return input_size + output_size


    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter (in kB).
        """
        output_size = self.algorithm.result_size(self.input_shape)
        return output_size * TVBSettings.MAGIC_NUMBER / 8 / 2 ** 10


    def launch(self, time_series, t_start, t_end):
        """
        Launch algorithm and build results.

        :param time_series: the input time-series for which correlation coefficient should be computed
        :param t_start: the physical time interval start for the analysis
        :param t_end: physical time, interval end
        :returns: the correlation coefficient for the given time series
        :rtype: `CorrelationCoefficients`
        """
        not_stored_result = self.algorithm.evaluate()
        result = CorrelationCoefficients(storage_path=self.storage_path, source=time_series)
        result.array_data = not_stored_result.array_data

        if isinstance(time_series, TimeSeriesEEG) or isinstance(time_series, TimeSeriesMEG) \
                or isinstance(time_series, TimeSeriesSEEG):
            result.labels_ordering = ["Sensor", "Sensor", "1", "1"]
        else:
            result.labels_ordering[0] = time_series.labels_ordering[2]
            result.labels_ordering[1] = time_series.labels_ordering[2]

        return result