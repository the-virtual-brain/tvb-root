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

The LookUpTable datatype. This brings together the scientific and framework 
methods that are associated with the precalculated look up tables.

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import tvb.datatypes.lookup_tables_scientific as lookup_tables_scientific
import tvb.datatypes.lookup_tables_framework as lookup_tables_framework
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)


class LookUpTable(lookup_tables_scientific.LookUpTableScientific,
                  lookup_tables_framework.LookUpTableFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the LookUpTable datatype.
    
    ::
        
                          LookUpTableData
                                 |
                                / \\
          LookUpTableFramework     LookUpTableScientific
                                \ /
                                 |
                            LookUpTable
        
    
    """
    
    pass


class PsiTable(lookup_tables_scientific.PsiTableScientific,
               lookup_tables_framework.PsiTableFramework, LookUpTable):
    """ 
    This class brings together the scientific and framework methods that are
    associated with the PsiTable dataType.
    
    ::
        
                             PsiTableData
                                 |
                                / \\
               PsiTableFramework   PsiTableScientific
                                \ /
                                 |
                               PsiTable
        
    
    """
    pass


class NerfTable(lookup_tables_scientific.NerfTableScientific,
               lookup_tables_framework.NerfTableFramework, LookUpTable):
    """ 
    This class brings together the scientific and framework methods that are
    associated with the NerfTable dataType.
    
    ::
        
                             NerfTableData
                                 |
                                / \\
              NerfTableFramework   NerfTableScientific
                                \ /
                                 |
                               NerfTable
        
    
    """
    pass

