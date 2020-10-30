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
from abc import abstractmethod
from tvb.basic.neotraits.api import NArray
from tvb.core.entities.transient.range_parameter import RangeParameter
from tvb.core.neotraits.forms import Form


class FormWithRanges(Form):
    @staticmethod
    @abstractmethod
    def get_params_configurable_in_phase_plane():
        """Return a list with all parameter names that can be configured in Phase Plane viewer for the current model"""

    def _get_parameters_for_pse(self):
        pass

    def _gather_parameters_with_range_defined(self):
        parameters_with_range = []
        for trait_field in self.trait_fields:
            if isinstance(trait_field.trait_attribute, NArray) and trait_field.trait_attribute.domain:
                parameters_with_range.append(trait_field.trait_attribute)
        return parameters_with_range

    def get_range_parameters(self):
        parameters_for_pse = self._gather_parameters_with_range_defined()

        result = []
        for range_param in parameters_for_pse:
            range_obj = RangeParameter(range_param.field_name, float, range_param.domain,
                                       isinstance(range_param, NArray))
            result.append(range_obj)
        return result
