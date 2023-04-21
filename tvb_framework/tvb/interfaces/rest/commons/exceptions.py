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

from abc import abstractmethod

from tvb.basic.exceptions import TVBException
from tvb.interfaces.rest.commons.status_codes import HTTP_STATUS_BAD_REQUEST, HTTP_STATUS_NOT_FOUND, \
    HTTP_STATUS_SERVER_ERROR, HTTP_STATUS_DENIED


class BaseRestException(TVBException):
    def __init__(self, message=None, code=None, payload=None):
        Exception.__init__(self)
        self.message = message if message is not None and message.strip() else self.get_default_message()
        self.code = code
        self.payload = payload

    def to_dict(self):
        payload = dict(self.payload or ())
        payload['message'] = self.message
        payload['code'] = self.code
        return payload

    @abstractmethod
    def get_default_message(self):
        return None


class BadRequestException(BaseRestException):
    def __init__(self, message, payload=None):
        super(BadRequestException, self).__init__(message, code=HTTP_STATUS_BAD_REQUEST, payload=payload)

    def get_default_message(self):
        return "Bad request error"


class InvalidIdentifierException(BaseRestException):
    def __init__(self, message=None, payload=None):
        super(InvalidIdentifierException, self).__init__(message, code=HTTP_STATUS_NOT_FOUND, payload=payload)

    def get_default_message(self):
        return "No data found for the given identifier"


class AuthorizationRequestException(BaseRestException):
    def __init__(self, message=None, code=HTTP_STATUS_DENIED):
        super(AuthorizationRequestException, self).__init__(message, code)

    def get_default_message(self):
        return "Token is missing."


class InvalidInputException(BadRequestException):
    def __init__(self, message=None, payload=None):
        super(InvalidInputException, self).__init__(message, payload=payload)

    def get_default_message(self):
        return "The input file is incomplete"


class ServiceException(BaseRestException):
    message_prefix = "Something went wrong on the server side"

    def __init__(self, message, code=HTTP_STATUS_SERVER_ERROR, payload=None):
        super(ServiceException, self).__init__(message, code, payload)
        self.message = self.message_prefix + ": " + message

    def get_default_message(self):
        return self.message_prefix


class ClientException(BaseRestException):
    def __init__(self, message, code=HTTP_STATUS_BAD_REQUEST):
        super(ClientException, self).__init__(message, code)

    def get_default_message(self):
        return "There was an error on client request"
