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
Calculate temporal cross correlation on a TimeSeries datatype and return a 
temporal_correlations.CrossCorrelation dataype.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import numpy
import tvb.datatypes.time_series as time_series
import tvb.datatypes.temporal_correlations as temporal_correlations
import tvb.basic.traits.core as core
import tvb.basic.traits.util as util
from scipy.signal.signaltools import correlate
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)




class CrossCorrelate(core.Type):
    """
    Compute the node-pairwise cross-correlation of the given input 4D TimeSeries DataType.
    
    Return a CrossCorrelation DataType. It contains the cross-correlation
    sequences for all possible combinations of the nodes.
    
    See: http://www.scipy.org/doc/api_docs/SciPy.signal.signaltools.html#correlate
    """

    time_series = time_series.TimeSeries(
        label="Time Series",
        required=True,
        doc="""The time-series for which the cross correlation sequences are calculated.""")
    
    
    def evaluate(self):
        """
        Cross-correlate two one-dimensional arrays.
        """
        cls_attr_name = self.__class__.__name__ + ".time_series"
        self.time_series.trait["data"].log_debug(owner=cls_attr_name)
        
        #(tpts, nodes, nodes, state-variables, modes)
        result_shape = self.result_shape(self.time_series.data.shape)
        LOG.info("result shape will be: %s" % str(result_shape))
        
        result = numpy.zeros(result_shape)
        
        #TODO: For region level, 4s, 2000Hz, this takes ~3hours...(which makes node_coherence seem positively speedy...)
        # Probably best to add a keyword for offsets, so we just compute +- some "small" range...
        # One inter-node correlation, across offsets, for each state-var & mode.
        for mode in range(result_shape[4]):
            for var in range(result_shape[3]):
                data = self.time_series.data[:, var, :, mode]
                data = data - data.mean(axis=0)[numpy.newaxis, :]
                #TODO: Work out a way around the 4 level loop:
                for n1 in range(result_shape[1]):
                    for n2 in range(result_shape[2]):
                        result[:, n1, n2, var, mode] = correlate(data[:, n1], data[:, n2], mode="same")
        
        util.log_debug_array(LOG, result, "result")
        
        offset = (self.time_series.sample_period *
                  numpy.arange(-numpy.floor(result_shape[0] / 2.0), numpy.ceil(result_shape[0] / 2.0)))

        cross_corr = temporal_correlations.CrossCorrelation(
            source=self.time_series,
            array_data=result,
            time=offset,
            use_storage=False)
        
        return cross_corr
    
    
    def result_shape(self, input_shape):
        """Returns the shape of the main result of ...."""
        result_shape = (input_shape[0], input_shape[2], input_shape[2], input_shape[1], input_shape[3])
        return result_shape
    
    
    def result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the main result of .
        """
        result_size = numpy.sum(map(numpy.prod, self.result_shape(input_shape))) * 8.0  # Bytes
        return result_size



