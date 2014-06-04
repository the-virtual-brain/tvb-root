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
Scientific methods for the LookUpTables datatypes.

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy
import tvb.datatypes.lookup_tables_data as lookup_tables_data
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)


class LookUpTableScientific(lookup_tables_data.LookUpTableData):
    """
    This class primarily exists to add scientific methods to the 
    LookUpTablesData class, if any ...
    
    """
    __tablename__ = None
    


class PsiTableScientific(lookup_tables_data.PsiTableData, LookUpTableScientific):
    
    __tablename__ = None
    
    
    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialization.
        """
        super(PsiTableScientific, self).configure()
        
        # Check if dx and invdx have been computed
        if self.number_of_values == 0:
            self.number_of_values = self.data.shape[0]

        if self.dx.size == 0:
            self.compute_search_indices()

        
        
    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this dataType, if any ... 
        """
        summary = {"Number of values": self.number_of_values}
        return summary
        
    def compute_search_indices(self):
        """
        ...
        """
        self.dx = ((self.xmax - self.xmin) / (self.number_of_values) - 1)
        self.invdx = 1 / self.dx
        
        self.trait["dx"].log_debug(owner=self.__class__.__name__)
        self.trait["invdx"].log_debug(owner=self.__class__.__name__)
    
    def search_value(self, val):
        """ 
        Search a value in this look up table
        """
         
        if self.xmin: 
            y = val - self.xmin
        else: 
            y = val
            
        ind = numpy.array(y * self.invdx, dtype=int)
        try:
            return self.data[ind] + self.df[ind]*(y - ind * self.dx)
        except IndexError: # out of bounds
            return numpy.NaN 
            # NOTE: not sure if we should return a NaN or make val = self.max
            
    pass
    

class NerfTableScientific(lookup_tables_data.NerfTableData, LookUpTableScientific):
    __tablename__ = None
    
    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialization.
        """
        super(NerfTableScientific, self).configure()

        if self.number_of_values == 0:
            self.number_of_values = self.data.shape[0]

        if self.dx.size == 0:
            self.compute_search_indices()
            
            
        
        
    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this dataType, if any ... 
        """
        summary = {"Number of values": self.number_of_values}
        return summary
        
    def compute_search_indices(self):
        """
        ...
        """
        self.dx = ((self.xmax - self.xmin) / (self.number_of_values) - 1)
        self.invdx = 1 / self.dx
        
    
    def search_value(self, val):
        """ 
        Search a value in this look up table
        """ 
        
        if self.xmin: 
            y = val - self.xmin
        else: 
            y = val
            
        ind = numpy.array(y * self.invdx, dtype=int)
        
        try:
            return self.data[ind] + self.df[ind]*(y - ind * self.dx)
        except IndexError: # out of bounds
            return numpy.NaN 
            # NOTE: not sure if we should return a NaN or make val = self.max
            # At the moment, we force the input values to be within a known range
    pass
