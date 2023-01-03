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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from datetime import datetime
from tvb.core.utils import date2string


class Exportable(object):
    
    def to_dict(self, excludes=None):
        """
        For a model entity, return a equivalent dictionary.
        """
        if excludes is None:
            excludes = ["id"]

        dict_equivalent = {}
        for key in self.__dict__:
            if '_sa_' not in key[:5] and key not in excludes:
                if isinstance(self.__dict__[key], datetime):
                    dict_equivalent[key] = date2string(self.__dict__[key])
                else:
                    dict_equivalent[key] = self.__dict__[key]
        return self.__class__.__name__, dict_equivalent

    def from_dict(self, dictionary):
        pass
