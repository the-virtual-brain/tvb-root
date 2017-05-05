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
Adapter that uses the traits module to generate interfaces for BalloonModel Analyzer.

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy
from tvb.analyzers.fmri_balloon import BalloonModel
from tvb.datatypes.time_series import TimeSeries
from tvb.datatypes.time_series import TimeSeriesRegion
from tvb.core.adapters.abcadapter import ABCAsynchronous
from tvb.basic.traits.util import log_debug_array
from tvb.basic.filters.chain import FilterChain
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)


class BalloonModelAdapter(ABCAsynchronous):
    """
    TVB adapter for calling the BalloonModel algorithm.
    """
    
    _ui_name = "Balloon Model "
    _ui_description = "Compute BOLD signals for a TimeSeries input DataType."
    _ui_subsection = "balloon"
    
    
    def get_input_tree(self):
        """
        Return a list of lists describing the interface to the analyzer. This
        is used by the GUI to generate the menus and fields necessary for
        defining current analysis.
        """
        algorithm = BalloonModel()
        algorithm.trait.bound = self.INTERFACE_ATTRIBUTES_ONLY
        tree = algorithm.interface[self.INTERFACE_ATTRIBUTES]
        for node in tree:
            if node['name'] == 'time_series':
                node['conditions'] = FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                                 operations=["=="], values=[4])
        return tree
    
    
    def get_output(self):
        return [TimeSeriesRegion]


    def configure(self, time_series, dt=None, bold_model=None, RBM=None, neural_input_transformation=None):
        """
        Store the input shape to be later used to estimate memory usage. Also
        create the algorithm instance.
        """
        self.input_shape = time_series.read_data_shape()
        log_debug_array(LOG, time_series, "time_series")
        
        ##-------------------- Fill Algorithm for Analysis -------------------##
        algorithm = BalloonModel()
        
        if dt is not None:
            algorithm.dt = dt
        else:
            algorithm.dt = time_series.sample_period / 1000.

        if bold_model is not None:
            algorithm.bold_model = bold_model
        if RBM is not None:
            algorithm.RBM = RBM
        if neural_input_transformation is not None:
            algorithm.neural_input_transformation = neural_input_transformation
        
        self.algorithm = algorithm
        self.algorithm.time_series = time_series


    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        used_shape = self.input_shape
        input_size = numpy.prod(used_shape) * 8.0
        output_size = self.algorithm.result_size(used_shape)
        return input_size + output_size   


    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter.(in kB)
        """
        used_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
        return self.array_size2kb(self.algorithm.result_size(used_shape))


    def launch(self, time_series, dt=None, bold_model=None, RBM=None, neural_input_transformation=None):
        """
        Launch algorithm and build results.

        :param time_series: the input time-series used as neural activation in the Balloon Model
        :returns: the simulated BOLD signal
        :rtype: `TimeSeries`
        """
        time_line = time_series.read_time_page(0, self.input_shape[0])
        bold_signal = TimeSeriesRegion(storage_path=self.storage_path,
                                       sample_period=time_series.sample_period,
                                       start_time=time_series.start_time,
                                       connectivity=time_series.connectivity)

        ##---------- Iterate over slices and compose final result ------------##

        node_slice = [slice(self.input_shape[0]), slice(self.input_shape[1]), None, slice(self.input_shape[3])]
        small_ts = TimeSeries(use_storage=False, sample_period=time_series.sample_period, time=time_line)
        
        for node in range(self.input_shape[2]):
            node_slice[2] = slice(node, node + 1)
            small_ts.data = time_series.read_data_slice(tuple(node_slice))
            self.algorithm.time_series = small_ts
            partial_bold = self.algorithm.evaluate()
            bold_signal.write_data_slice(partial_bold.data, grow_dimension=2)

        bold_signal.write_time_slice(time_line)
        bold_signal.close_file()
        return bold_signal