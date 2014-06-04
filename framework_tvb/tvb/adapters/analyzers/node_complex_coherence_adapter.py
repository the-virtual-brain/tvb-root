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
Adapter that uses the traits module to generate interfaces for FFT Analyzer.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Paula Sanz Leon <paula@tvb.invalid>

"""

import numpy
from tvb.basic.config.settings import TVBSettings
from tvb.analyzers.node_complex_coherence import NodeComplexCoherence
from tvb.core.adapters.abcadapter import ABCAsynchronous
from tvb.datatypes.time_series import TimeSeries
from tvb.datatypes.spectral import ComplexCoherenceSpectrum
from tvb.basic.filters.chain import FilterChain
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)



class NodeComplexCoherenceAdapter(ABCAsynchronous):
    """ TVB adapter for calling the NodeComplexCoherence algorithm. """
    
    _ui_name = "Complex Coherence of Nodes"
    _ui_description = "Compute the node complex (imaginary) coherence for a TimeSeries input DataType."
    _ui_subsection = "complexcoherence"
    
    
    def get_input_tree(self):
        """
        Return a list of lists describing the interface to the analyzer. This
        is used by the GUI to generate the menus and fields necessary for
        defining a simulation.
        """
        algorithm = NodeComplexCoherence()
        algorithm.trait.bound = self.INTERFACE_ATTRIBUTES_ONLY
        tree = algorithm.interface[self.INTERFACE_ATTRIBUTES]
        for node in tree:
            if node['name'] == 'time_series':
                node['conditions'] = FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                                 operations=["=="], values=[4])
        return tree
    
    
    def get_output(self):
        return [ComplexCoherenceSpectrum]
    

    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """        
        used_shape = self.algorithm.time_series.read_data_shape()
        input_size = numpy.prod(used_shape) * 8.0
        output_size = self.algorithm.result_size(used_shape, self.algorithm.max_freq,
                                                 self.algorithm.epoch_length,
                                                 self.algorithm.segment_length,
                                                 self.algorithm.segment_shift,
                                                 self.algorithm.time_series.sample_period,
                                                 self.algorithm.zeropad,
                                                 self.algorithm.average_segments)
                                                 
        return input_size + output_size
        

    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter (in kB).
        """
        used_shape = self.algorithm.time_series.read_data_shape()
        return self.algorithm.result_size(used_shape, self.algorithm.max_freq,
                                          self.algorithm.epoch_length,
                                          self.algorithm.segment_length,
                                          self.algorithm.segment_shift,
                                          self.algorithm.time_series.sample_period,
                                          self.algorithm.zeropad,
                                          self.algorithm.average_segments) * TVBSettings.MAGIC_NUMBER / 8 / 2 ** 10
        
        
    def configure(self, time_series):
        """
        Do any configuration needed before launching and create an instance of the algorithm.
        """
        shape = time_series.read_data_shape()
        LOG.debug("time_series shape is %s" % (str(shape)))
        ##-------------------- Fill Algorithm for Analysis -------------------##
        self.algorithm = NodeComplexCoherence()
        self.algorithm.time_series = time_series
        self.memory_factor = 1
        
    
    def launch(self, time_series):
        """
        Launch algorithm and build results.

        :returns: the `ComplexCoherenceSpectrum` built with the given time-series
        """
        shape = time_series.read_data_shape()
        
        ##------- Prepare a ComplexCoherenceSpectrum object for result -------##
        spectra = ComplexCoherenceSpectrum(source=time_series,
                                           storage_path=self.storage_path)
        
        ##------------------- NOTE: Assumes 4D TimeSeries. -------------------##
        node_slice = [slice(shape[0]), slice(shape[1]), slice(shape[2]), slice(shape[3])]
        
        ##---------- Iterate over slices and compose final result ------------##
        small_ts = TimeSeries(use_storage=False)
        small_ts.sample_rate = time_series.sample_rate
        small_ts.data = time_series.read_data_slice(tuple(node_slice))
        self.algorithm.time_series = small_ts
        
        partial_result = self.algorithm.evaluate()
        LOG.debug("got partial_result")
        LOG.debug("partial segment_length is %s" % (str(partial_result.segment_length)))
        LOG.debug("partial epoch_length is %s" % (str(partial_result.epoch_length)))
        LOG.debug("partial windowing_function is %s" % (str(partial_result.windowing_function)))
        #LOG.debug("partial frequency vector is %s" % (str(partial_result.frequency)))
        
        spectra.write_data_slice(partial_result)
        spectra.segment_length = partial_result.segment_length
        spectra.epoch_length = partial_result.epoch_length
        spectra.windowing_function = partial_result.windowing_function
        #spectra.frequency = partial_result.frequency
        spectra.close_file()
        return spectra
    
    
    