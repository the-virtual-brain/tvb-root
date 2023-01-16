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
Exceptions for services layer of the application. 
   
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
from tvb.basic.exceptions import TVBException


class ServicesBaseException(TVBException):
    """
    Base Exception class for Services layer in the application.
    """


class StructureException(ServicesBaseException):
    """
    Exception to be thrown in case of a problem related to Structure Storage.
    """


class OperationException(ServicesBaseException):
    """
    Exception to be thrown in case of a problem related to Launching 
    and Executing TVB specific Operations.
    """


class UsernameException(ServicesBaseException):
    """
    Exception to be thrown in case of a problem related to creating
    or managing a user.
    """


class ProjectServiceException(ServicesBaseException):
    """
    Exception to be thrown in case of a problem in the projectservice
    module.
    """


class ImportException(ServicesBaseException):
    """
    Exception to be thrown in case of a problem at project import.
    """


class MissingReferenceException(ImportException):
    """
    Exception to be thrown in case there are missing references when importing a H5 file.
    """


class DatatypeGroupImportException(ImportException):
    """
    Exception to be thrown in case there are issues when importing a datatype group.
    """


class BurstServiceException(ServicesBaseException):
    """
    Exception to be thrown in case of a problem at project import.
    """


class InvalidSettingsException(ServicesBaseException):
    """
    Exception to be thrown in case of a problem at project import.
    """


class RemoveDataTypeException(ServicesBaseException):
    """
    Exception to be thrown in case some one tries to remove an
    entity that is used by other entities.
    """
