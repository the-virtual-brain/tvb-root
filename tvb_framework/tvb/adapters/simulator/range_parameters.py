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
from tvb.adapters.forms.form_with_ranges import FormWithRanges
from tvb.basic.neotraits.api import NArray, Range
from tvb.core.entities.transient.range_parameter import RangeParameter
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.simulator import Simulator


class RangeParametersCollector(object):

    def __init__(self):
        self.range_param_forms = dict()

    @staticmethod
    def _default_range_parameters():
        conduction_speed = RangeParameter(Simulator.conduction_speed.field_name, float,
                                          # TODO: Float should support ranges
                                          Range(lo=0.01, hi=100.0, step=1.0),
                                          isinstance(Simulator.conduction_speed, NArray))
        connectivity = RangeParameter(Simulator.connectivity.field_name, Connectivity)

        return [conduction_speed, connectivity]

    def get_all_range_parameters(self):
        all_range_parameters = self._default_range_parameters()
        param_prefixes_dict = FormWithRanges.get_range_param_prefixes_dict()
        for param_category_name, form in self.range_param_forms.items():
            if form is not None:
                prefix = param_prefixes_dict[param_category_name]
                range_params = form.get_range_parameters(prefix)
                all_range_parameters.extend(range_params)

        return all_range_parameters
