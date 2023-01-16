# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#
"""
Created on Jan 15, 2013

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""



class BaseStorageException(Exception):
    """
    Base class for all TVB storage exceptions.
    """

    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message


    def __repr__(self):
        return self.message



class NestedTransactionUnsupported(BaseStorageException):
    """
    Nested transactions are not supported unless in testing.
    """

    def __init__(self, message):
        BaseStorageException.__init__(self, message)



class InvalidTransactionAccess(BaseStorageException):
    """
    Exception raised in case you have any faulty access to a transaction.
    """

    def __init__(self, message):
        BaseStorageException.__init__(self, message)
        
    
    