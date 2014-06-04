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
Created on Nov 1, 2011

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

####### Default factory is empty
FACTORY_DICTIONARY = {}


def get_remover(datatype_name):
    """
    :param datatype_name: class-name; to search for a remover class for this. 
    """
    if isinstance(datatype_name, str) or isinstance(datatype_name, unicode):
        datatype_name = datatype_name.split('.')[-1]
    else:
        datatype_name = datatype_name.__name__
    
    if datatype_name in FACTORY_DICTIONARY:
        return FACTORY_DICTIONARY[datatype_name]
    else:
        return FACTORY_DICTIONARY['default']
    
    
def update_dictionary(new_dict):
    """
    Update removers dictionary.
    """
    FACTORY_DICTIONARY.update(new_dict)
    
    
    