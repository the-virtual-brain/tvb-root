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
Perform Independent Component Analysis on a TimeSeries Object and returns an
IndependentComponents datatype.

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

#Third party libraries
import numpy
from sklearn.decomposition import fastica

# TVB
#TODO: Currently built around the Simulator's 4D timeseries (as PCA) -- generalise
import tvb.datatypes.time_series as time_series
import tvb.datatypes.mode_decompositions as mode_decompositions
import tvb.basic.traits.core as core
import tvb.basic.traits.types_basic as basic
import tvb.basic.traits.util as util
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)


class fastICA(core.Type):
    """
    Takes a TimeSeries datatype (x) and returns the unmixed temporal sources (S) 
    and the estimated mixing matrix (A).
    
    :math: x = AS
    
    ICA takes time-points as observations and nodes as variables.
    
    It uses the fastICA algorithm implemented in the scikit-learn toolkit, and 
    its intended usage is as a `blind source separation` method.
    
    See also: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.fastica.html#sklearn.decomposition.fastica
    
    Before the fastICA algorithm can be applied, the input vector data 
    should be whitened (`sphering`). This means that any correlations in the 
    data are removed, i.e. the signals are forced to be uncorrelated. To this end,
    the `whiten` parameter is always set to `True`.
    
    NOTE: As for PCA the TimeSeries datatype must be longer (more time-points)
          than the number of nodes -- Mostly a problem for TimeSeriesSurface 
          datatypes, which, if sampled at 1024Hz, would need to be greater than 
          16 seconds long.
    """
    
    time_series = time_series.TimeSeries(
        label = "Time Series",
        required = True,
        doc = """The timeseries to which the ICA is to be applied. NOTE: The 
            TimeSeries must be longer(more time-points) than the number of nodes
            -- Mostly a problem for surface times-series, which, if sampled at
            1024Hz, would need to be greater than 16 seconds long.""")
            
    n_components = basic.Integer(
        label = "Number of components to extract",
        required = False,
        default = None,
        doc = """Number of components to extract and to perform dimension reduction.
            The number of components must be less than the number of variables.
            By default it takes number of components = number of nodes. Definitely
            a problem for surface time-series.""")
    
    # NOTE: For default surface the weights matrix has a size ~ 2GB * modes * vars...
    
    def evaluate(self):
        """
        Compute the independent sources 
        """
        cls_attr_name = self.__class__.__name__+".time_series"
        self.time_series.trait["data"].log_debug(owner = cls_attr_name)
        
        ts_shape = self.time_series.data.shape
        
        #Need more observations than variables
        if ts_shape[0] < ts_shape[2]:
            msg = "ICA requires a longer timeseries (tpts > number of nodes)."
            LOG.error(msg)
            raise Exception, msg
            
        #Need more variables than components
        if self.n_components > ts_shape[2]:
            msg = "ICA requires more variables than components to extract (number of nodes > number of components)."
            LOG.error(msg)
            raise Exception, msg
        
        if self.n_components is None:
            self.n_components = ts_shape[2]
        
        #(n_components, n_components, state-variables, modes) --  unmixing matrix
        unmixing_matrix_shape = (self.n_components, self.n_components, ts_shape[1], ts_shape[3])
        LOG.info("unmixing matrix shape will be: %s" % str(unmixing_matrix_shape))
        
        # (n_components, nodes, state_variables, modes) -- prewhitening matrix
        prewhitening_matrix_shape = (self.n_components, ts_shape[2], ts_shape[1], ts_shape[3])
        LOG.info("prewhitening matrix shape will be: %s" % str(prewhitening_matrix_shape))
        
        
        unmixing_matrix = numpy.zeros(unmixing_matrix_shape)
        prewhitening_matrix = numpy.zeros(prewhitening_matrix_shape)
        
        
        #(tpts, n_components, state_variables, modes) -- unmixed sources time series
        data_ica = numpy.zeros((ts_shape[0], self.n_components, ts_shape[1], ts_shape[3]))
        
        #One un/mixing matrix for each state-var & mode.
        for mode in range(ts_shape[3]):
            for var in range(ts_shape[1]):
                # Assumes data must be whitened
                ica = fastica(self.time_series.data[:, var, :, mode], 
                                            n_components = self.n_components,
                                            whiten = True)
                # unmixed sources - component_time_series
                data_ica[:, :, var, mode] = ica[2]
                # prewhitening matrix
                prewhitening_matrix[:, :, var, mode] = ica[0]
                # unmixing matrix
                unmixing_matrix[:, :, var, mode] = ica[1]
        
        util.log_debug_array(LOG, prewhitening_matrix, "whitening_matrix")
        util.log_debug_array(LOG, unmixing_matrix, "unmixing_matrix")

        
        ica_result = mode_decompositions.IndependentComponents(source = self.time_series,
                                         component_time_series = data_ica, 
                                         #mixing_matrix = mixing_matrix,
                                         prewhitening_matrix = prewhitening_matrix,
                                         unmixing_matrix = unmixing_matrix,
                                         n_components = self.n_components, 
                                         use_storage = False)
        
        return ica_result
    
    
    def result_shape(self, input_shape):
        """
        Returns the shape of the main result of the ICA analysis -- component
        mixing matrix.
        """
        unmixing_matrix_shape = ((self.n_components or input_shape[2]), 
                                 (self.n_components or input_shape[2]),
                                 input_shape[1], input_shape[3])
        return unmixing_matrix_shape
    
    
    def result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the results of the ICA analysis.
        """
        result_size = numpy.sum(map(numpy.prod, self.result_shape(input_shape))) * 8.0 #Bytes
        return result_size
    
    
    def extended_result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the extended result of the ICA.
        That is, it includes storage of the evaluated IndependentComponents
        attributes such as norm_source, component_time_series, etc.
        """
        result_size = self.result_size(input_shape)
        extend_size = result_size #Main arrays
        extend_size = extend_size + numpy.prod(input_shape) * 8.0 #norm_source
        extend_size = extend_size + numpy.prod(input_shape) * 8.0 #component_time_series
        extend_size = extend_size + numpy.prod(input_shape) * 8.0 #normalised_component_time_series
        return extend_size


