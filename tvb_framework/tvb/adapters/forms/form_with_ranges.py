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
from tvb.basic.neotraits.api import NArray
from tvb.core.entities.transient.range_parameter import RangeParameter
from tvb.core.neotraits.forms import Form
from tvb.simulator.integrators import IntegratorStochastic
from tvb.simulator.simulator import Simulator


class FormWithRanges(Form):
    COUPLING_FRAGMENT_KEY = 'coupling_fragment'
    CORTEX_FRAGMENT_KEY = 'cortex_fragment'
    MODEL_FRAGMENT_KEY = 'model_fragment'
    NOISE_FRAGMENT_KEY = 'noise_fragment'

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

    def get_range_parameters(self, prefix):
        parameters_for_pse = self._gather_parameters_with_range_defined()

        result = []
        for range_param in parameters_for_pse:
            range_obj = RangeParameter(range_param.label, float, range_param.domain,
                                       isinstance(range_param, NArray))
            self.ensure_correct_prefix_for_param_name(range_obj, prefix)
            result.append(range_obj)
        return result

    @staticmethod
    def ensure_correct_prefix_for_param_name(param, prefix):
        prefix = prefix + '.'
        if not param.name.startswith(prefix):
            param_full_name = prefix + param.name
            param.name = param_full_name

    @staticmethod
    def get_range_param_prefixes_dict():
        return {
            FormWithRanges.COUPLING_FRAGMENT_KEY: Simulator.coupling.field_name,
            FormWithRanges.CORTEX_FRAGMENT_KEY: Simulator.surface.field_name,
            FormWithRanges.MODEL_FRAGMENT_KEY: Simulator.model.field_name,
            FormWithRanges.NOISE_FRAGMENT_KEY:
                Simulator.integrator.field_name + "." + IntegratorStochastic.noise.field_name
        }
