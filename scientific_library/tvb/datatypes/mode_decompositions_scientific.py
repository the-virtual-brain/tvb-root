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
Scientific methods for the Spectral datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy
import tvb.datatypes.mode_decompositions_data as mode_decompositions_data
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)


class PrincipalComponentsScientific(mode_decompositions_data.PrincipalComponentsData):
    """
    This class exists to add scientific methods to PrincipalComponentsData.
    """
    __tablename__ = None
    
    
    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialization.
        """
        super(PrincipalComponentsScientific, self).configure()
        
        if self.trait.use_storage is False and sum(self.get_data_shape('weights')) != 0:
            if self.norm_source.size == 0:
                self.compute_norm_source()
            
            if self.component_time_series.size == 0:
                self.compute_component_time_series()
            
            if self.normalised_component_time_series.size == 0:
                self.compute_normalised_component_time_series()
    
    
    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this datatype.
        """
        summary = {"Mode decomposition type": self.__class__.__name__}
        summary["Source"] = self.source.title
        #summary["Number of variables"] = self...
        #summary["Number of mewasurements"] = self...
        #summary["Number of components"] = self...
        #summary["Number required for 95%"] = self...
        return summary
    
    
    def compute_norm_source(self):
        """Normalised source time-series."""
        self.norm_source = ((self.source.data - self.source.data.mean(axis=0)) /
                            self.source.data.std(axis=0))
        self.trait["norm_source"].log_debug(owner=self.__class__.__name__)
    
    
    #TODO: ??? Any value in making this a TimeSeries datatypes ???
    def compute_component_time_series(self):
        """Compnent time-series."""
        #TODO: Generalise -- it currently assumes 4D TimeSeriesSimulator...
        ts_shape = self.source.data.shape
        component_ts = numpy.zeros(ts_shape)
        for var in range(ts_shape[1]):
            for mode in range(ts_shape[3]):
                w = self.weights[:, :, var, mode]
                ts = self.source.data[:, var, :, mode]
                component_ts[:, var, :, mode] = numpy.dot(w, ts.T).T
        
        self.component_time_series = component_ts
        self.trait["component_time_series"].log_debug(owner=self.__class__.__name__)
    
    
    #TODO: ??? Any value in making this a TimeSeries datatypes ???
    def compute_normalised_component_time_series(self):
        """normalised_Compnent time-series."""
        #TODO: Generalise -- it currently assumes 4D TimeSeriesSimulator...
        ts_shape = self.source.data.shape
        component_ts = numpy.zeros(ts_shape)
        for var in range(ts_shape[1]):
            for mode in range(ts_shape[3]):
                w = self.weights[:, :, var, mode]
                nts = self.norm_source[:, var, :, mode]
                component_ts[:, var, :, mode] = numpy.dot(w, nts.T).T
        
        self.normalised_component_time_series = component_ts
        self.trait["normalised_component_time_series"].log_debug(owner=self.__class__.__name__)



class IndependentComponentsScientific(mode_decompositions_data.IndependentComponentsData):
    """
    This class exists to add scientific methods to IndependentComponentsData.
    
    """
    __tablename__ = None
    
    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialisation.
        """
        super(IndependentComponentsScientific, self).configure()
        
        if self.trait.use_storage is False and sum(self.get_data_shape('unmixing_matrix')) != 0:
            if self.norm_source.size == 0:
                self.compute_norm_source()
            
            if self.component_time_series.size == 0:
                self.compute_component_time_series()
            
            if self.normalised_component_time_series.size == 0:
                self.compute_normalised_component_time_series()
                
                
    def compute_norm_source(self):
        """Normalised source time-series."""
        self.norm_source = ((self.source.data - self.source.data.mean(axis=0)) /
                            self.source.data.std(axis=0))
        self.trait["norm_source"].log_debug(owner=self.__class__.__name__)
        
    #TODO: ??? Any value in making this a TimeSeries datatypes ??? -- 
    # (PAULA) >> a component time-series datatype?
    def compute_component_time_series(self):
        """Component time-series."""
        #TODO: Generalise -- it currently assumes 4D TimeSeriesSimulator...
        ts_shape = self.source.data.shape
        component_ts_shape = (ts_shape[0], ts_shape[1], self.n_components, ts_shape[3])
        component_ts = numpy.zeros(component_ts_shape)
        for var in range(ts_shape[1]):
            for mode in range(ts_shape[3]):
                w = self.unmixing_matrix[:, :, var, mode]
                k = self.prewhitening_matrix[:, :, var, mode]
                ts = self.source.data[:, var, :, mode]
                component_ts[:, var, : , mode] = numpy.dot(w, numpy.dot(k, ts.T)).T         
        
        self.component_time_series = component_ts
        self.trait["component_time_series"].log_debug(owner=self.__class__.__name__)
        
        
    #TODO: ??? Any value in making this a TimeSeries datatypes ???
    def compute_normalised_component_time_series(self):
        """normalised_component time-series."""
        #TODO: Generalise -- it currently assumes 4D TimeSeriesSimulator...
        ts_shape = self.source.data.shape
        component_ts_shape = (ts_shape[0], ts_shape[1], self.n_components, ts_shape[3])
        component_nts = numpy.zeros(component_ts_shape)
        for var in range(ts_shape[1]):
            for mode in range(ts_shape[3]):
                w = self.unmixing_matrix[:, :, var, mode]
                k = self.prewhitening_matrix[:, :, var, mode]
                nts = self.norm_source[:, var, :, mode]
                component_nts[:, var, :, mode] = numpy.dot(w, numpy.dot(k, nts.T)).T              
        self.normalised_component_time_series = component_nts
        self.trait["normalised_component_time_series"].log_debug(owner=self.__class__.__name__)
        
    def compute_mixing_matrix(self):
        """ 
        Compute the linear mixing matrix A, so X = A * S , 
        where X is the observed data and S contain the independent components 
            """
        ts_shape = self.source.data.shape
        mixing_matrix_shape = (ts_shape[2], self.n_components, ts_shape[1], ts_shape[3])
        mixing_matrix = numpy.zeros(mixing_matrix_shape)
        for var in range(ts_shape[1]):
            for mode in range(ts_shape[3]):
                w = self.unmixing_matrix[:, :, var, mode]
                k = self.prewhitening_matrix[:, :, var, mode]
                temp = numpy.matrix(numpy.dot(w, k))
                mixing_matrix[:, :, var, mode] = numpy.array(numpy.dot(temp.T ,(numpy.dot(temp, temp.T)).I))
        self.mixing_matrix = mixing_matrix
        self.trait["mixing_matrix"].log_debug(owner=self.__class__.__name__)
    
    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this datatype.
        """
        summary = {"Mode decomposition type": self.__class__.__name__}
        summary["Source"] = self.source.title
        #summary["Number of variables"] = str(self.source.read_data_shape()[2])
        #summary["Number of measurements"] = str(self.source.read_data_shape()[0])
        #summary["Number of components"] = str(self.n_components)
        #summary.update(self.get_info_about_array('array_data'))
        return summary


