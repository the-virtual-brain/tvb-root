# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
Perform Principal Component Analysis (PCA) on a TimeSeries datatype and return
a PrincipalComponents datatype.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

#TODO: Make an appropriate datatype for the output, include properties to
#      project source timesereis to component timeserries, etc

import numpy
import matplotlib.mlab as mlab
#TODO: Currently built around the Simulator's 4D timeseries -- generalise...
import tvb.datatypes.time_series as time_series
import tvb.datatypes.mode_decompositions as mode_decompositions
import tvb.basic.traits.core as core
import tvb.basic.traits.util as util
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)



class PCA(core.Type):
    """
    Return principal component weights and the fraction of the variance that 
    they explain. 
    
    PCA takes time-points as observations and nodes as variables.
    
    NOTE: The TimeSeries must be longer(more time-points) than the number of
          nodes -- Mostly a problem for TimeSeriesSurface datatypes, which, if 
          sampled at 1024Hz, would need to be greater than 16 seconds long.
    """
    
    time_series = time_series.TimeSeries(
        label = "Time Series",
        required = True,
        doc = """The timeseries to which the PCA is to be applied. NOTE: The 
            TimeSeries must be longer(more time-points) than the number of nodes
            -- Mostly a problem for surface times-series, which, if sampled at
            1024Hz, would need to be greater than 16 seconds long.""")
    
    #TODO: Maybe should support first N components or neccessary components to
    #      explain X% of the variance. NOTE: For default surface the weights
    #      matrix has a size ~ 2GB * modes * vars...
    
    def evaluate(self):
        """
        Compute the temporal covariance between nodes in the time_series. 
        """
        cls_attr_name = self.__class__.__name__+".time_series"
        self.time_series.trait["data"].log_debug(owner = cls_attr_name)
        
        ts_shape = self.time_series.data.shape
        
        #Need more measurements than variables
        if ts_shape[0] < ts_shape[2]:
            msg = "PCA requires a longer timeseries (tpts > number of nodes)."
            LOG.error(msg)
            raise Exception, msg
        
        #(nodes, nodes, state-variables, modes)
        weights_shape = (ts_shape[2], ts_shape[2], ts_shape[1], ts_shape[3])
        LOG.info("weights shape will be: %s" % str(weights_shape))
        
        fractions_shape = (ts_shape[2], ts_shape[1], ts_shape[3])
        LOG.info("fractions shape will be: %s" % str(fractions_shape))
        
        weights = numpy.zeros(weights_shape)
        fractions = numpy.zeros(fractions_shape)
        
        #One inter-node temporal covariance matrix for each state-var & mode.
        for mode in range(ts_shape[3]):
            for var in range(ts_shape[1]):
                data = self.time_series.data[:, var, :, mode]
                data_pca = mlab.PCA(data)
                fractions[:, var, mode ] = data_pca.fracs
                weights[:, :, var, mode] = data_pca.Wt
        
        util.log_debug_array(LOG, fractions, "fractions")
        util.log_debug_array(LOG, weights, "weights")
        
        pca_result = mode_decompositions.PrincipalComponents(
            source = self.time_series,
            fractions = fractions,
            weights = weights,
            use_storage = False)
        
        return pca_result
    
    
    def result_shape(self, input_shape):
        """
        Returns the shape of the main result of the PCA analysis -- compnnent 
        weights matrix and a vector of fractions.
        """
        weights_shape = (input_shape[2], input_shape[2], input_shape[1],
                         input_shape[3])
        fractions_shape = (input_shape[2], input_shape[1], input_shape[3])
        return [weights_shape, fractions_shape]
    
    
    def result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the results of the PCA analysis.
        """
        result_size = numpy.sum(map(numpy.prod,
                                    self.result_shape(input_shape))) * 8.0 #Bytes
        return result_size
    
    
    def extended_result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the extended result of the PCA.
        That is, it includes storage of the evaluated PrincipleComponents
        attributes such as norm_source, component_time_series, etc.
        """
        result_size = self.result_size(input_shape)
        extend_size = result_size #Main arrays
        extend_size = extend_size + numpy.prod(input_shape) * 8.0 #norm_source
        extend_size = extend_size + numpy.prod(input_shape) * 8.0 #component_time_series
        extend_size = extend_size + numpy.prod(input_shape) * 8.0 #normalised_component_time_series
        return extend_size


