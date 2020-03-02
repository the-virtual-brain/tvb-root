# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
import json
from functools import wraps

from tvb.interfaces.rest.commons.decoders import CustomDecoder
from tvb.interfaces.rest.commons.exceptions import ClientException


def handle_response(func):
    @wraps(func)
    def decorator(*a, **b):
        result = func(*a, **b)
        response = result
        classz = None

        if isinstance(result, tuple):
            response = result[0]
            classz = result[1]

        content = response.content
        successful_call = response.ok

        if successful_call:
            if classz is not None:
                return json.loads(content.decode('utf-8'),
                                  object_hook=lambda d: classz(**d) if '__type__' not in d
                                  else CustomDecoder.custom_hook(d))
            return json.loads(content.decode('utf-8'), cls=CustomDecoder)

        decoded_dict = json.loads(content.decode('utf-8'))
        try:
            error_message = decoded_dict['message']
        except KeyError:
            error_message = response.text
        raise ClientException(error_message, response.status_code)

    return decorator
