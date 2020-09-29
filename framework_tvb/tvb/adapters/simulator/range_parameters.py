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
from collections import OrderedDict

from tvb.basic.neotraits.api import NArray, Range
from tvb.core.entities.transient.range_parameter import RangeParameter
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.surfaces import Surface
from tvb.simulator.integrators import IntegratorStochastic
from tvb.simulator.simulator import Simulator


class SimulatorRangeParameters(object):

    def __init__(self, connectivity_filters=None, surface_filters=None, coupling=None, model=None,
                 integrator_noise=None):
        self.connectivity_filters = connectivity_filters
        self.surface_filters = surface_filters
        self.coupling_parameters = coupling
        self.model_parameters = model
        self.integrator_noise_parameters = integrator_noise

    def _default_range_parameters(self):
        conduction_speed = RangeParameter(Simulator.conduction_speed.field_name, float,
                                          # TODO: Float should support ranges
                                          Range(lo=0.01, hi=100.0, step=1.0),
                                          isinstance(Simulator.conduction_speed, NArray))
        connectivity = RangeParameter(Simulator.connectivity.field_name, Connectivity, self.connectivity_filters)
        surface = RangeParameter(Simulator.surface.field_name, Surface, self.connectivity_filters)

        return OrderedDict({Simulator.conduction_speed.field_name: conduction_speed,
                            Simulator.connectivity.field_name: connectivity,
                            Simulator.surface.field_name: surface})

    def _ensure_correct_prefix_for_param_name(self, prefix, param):
        prefix = prefix + '.'
        if not param.name.startswith(prefix):
            param_full_name = prefix + param.name
            param.name = param_full_name

    def _prepare_dynamic_parameters(self, param_prefix, param_list):
        dynamic_parameters = {}
        if param_list is None:
            return dynamic_parameters

        for param in param_list:
            self._ensure_correct_prefix_for_param_name(param_prefix, param)
            dynamic_parameters.update({param.name: param})
        return dynamic_parameters

    def _prepare_model_parameters(self):
        return self._prepare_dynamic_parameters(Simulator.model.field_name, self.model_parameters)

    def _prepare_coupling_parameters(self):
        return self._prepare_dynamic_parameters(Simulator.coupling.field_name, self.coupling_parameters)

    def _prepare_integrator_noise_parameters(self):
        return self._prepare_dynamic_parameters(
            Simulator.integrator.field_name + '.' + IntegratorStochastic.noise.field_name,
            self.integrator_noise_parameters)

    def get_all_range_parameters(self):
        all_range_parameters = self._default_range_parameters()
        all_range_parameters.update(self._prepare_coupling_parameters())
        all_range_parameters.update(self._prepare_model_parameters())
        all_range_parameters.update(self._prepare_integrator_noise_parameters())

        return all_range_parameters

    def add_connectivity_filter(self, filter):
        # TODO: add to FilterChain not to list
        if self.connectivity_filters is None:
            self.connectivity_filters = []
        self.connectivity_filters.append(filter)

    def add_surface_filter(self, filter):
        if self.surface_filters is None:
            self.surface_filters = []
        self.surface_filters.append(filter)
