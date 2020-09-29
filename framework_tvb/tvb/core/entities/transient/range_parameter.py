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

import numpy
from tvb.basic.neotraits.api import HasTraits, Range


class RangeParameter(object):
    KEY_FLOAT_LO = 'lo'
    KEY_FLOAT_STEP = 'step'
    KEY_FLOAT_HI = 'hi'

    def __init__(self, name, type, range_definition, is_array=False, range_values=None):
        self.name = name
        self.type = type
        self.is_array = is_array
        # range_definition is a Range object when type=float and a FilterChain otherwise
        self.range_definition = range_definition
        self.range_values = range_values

    def to_json(self):
        if self.type is float:
            return json.dumps([self.name, {self.KEY_FLOAT_LO: self.range_definition.lo,
                                           self.KEY_FLOAT_STEP: self.range_definition.step,
                                           self.KEY_FLOAT_HI: self.range_definition.hi}])

        return json.dumps([self.name, self.range_definition])

    @staticmethod
    def from_json(range_json):
        range_list = json.loads(range_json)
        if isinstance(range_list[1], dict):
            return RangeParameter(range_list[0], float,
                                  Range(range_list[1][RangeParameter.KEY_FLOAT_LO],
                                        range_list[1][RangeParameter.KEY_FLOAT_HI],
                                        range_list[1][RangeParameter.KEY_FLOAT_STEP]))

        return RangeParameter(range_list[0], HasTraits, range_list[1])

    def get_range_values(self):
        if self.type is float:
            self.range_values = self.range_definition.to_array()
            if self.is_array:
                self.range_values = [numpy.array([value]) for value in self.range_values]
        return self.range_values

    def fill_from_default(self, range_param):
        # type: (RangeParameter) -> None
        self.type = range_param.type
        self.is_array = range_param.is_array
