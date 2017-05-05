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
Calculate a ... on a .. datatype and return a ...

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import numpy
#TODO: Currently built around the Simulator's 4D timeseries -- generalise...
import tvb.datatypes.time_series as time_series
import tvb.datatypes.graph as graph
import tvb.basic.traits.core as core
import tvb.basic.traits.util as util
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)




class NodeCovariance(core.Type):
    """
    Compute the temporal covariance of nodes in a TimeSeries dataType.
    A nodes x nodes matrix is returned for each (state-variable, mode).
    """

    time_series = time_series.TimeSeries(
        label="Time Series",
        required=True,
        doc="""The timeseries to which the NodeCovariance is to be applied.""")
    
    
    def evaluate(self):
        """
        Compute the temporal covariance between nodes in the time_series.
        """
        cls_attr_name = self.__class__.__name__ + ".time_series"
        self.time_series.trait["data"].log_debug(owner=cls_attr_name)
        
        data_shape = self.time_series.data.shape
        
        #(nodes, nodes, state-variables, modes)
        result_shape = (data_shape[2], data_shape[2], data_shape[1], data_shape[3])
        LOG.info("result shape will be: %s" % str(result_shape))
        
        result = numpy.zeros(result_shape)
        
        #One inter-node temporal covariance matrix for each state-var & mode.
        for mode in range(data_shape[3]):
            for var in range(data_shape[1]):
                data = self.time_series.data[:, var, :, mode]
                data = data - data.mean(axis=0)[numpy.newaxis, 0]
                result[:, :, var, mode] = numpy.cov(data.T)

        util.log_debug_array(LOG, result, "result")

        covariance = graph.Covariance(source=self.time_series,
                                      array_data=result,
                                      use_storage=False)
        return covariance
    
    
    def result_shape(self, input_shape):
        """
        Returns the shape of the main result of the NodeCovariance analysis.
        """
        result_shape = (input_shape[2], input_shape[2], input_shape[1], input_shape[3])
        return result_shape
    
    
    def result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the NodeCovariance result.
        """
        result_size = numpy.prod(self.result_shape(input_shape)) * 8.0  # Bytes
        return result_size
    
    
    def extended_result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the NodeCovariance extended result.
        That is, it includes storage of the evaluated PrincipleComponents
        attributes such as norm_source, component_time_series, etc.
        """
        extend_size = self.result_size(input_shape)  # Currently no derived attributes.
        return extend_size


