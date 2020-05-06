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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

from enum import Enum
from functools import partial

from tvb.adapters.simulator.equation_forms import get_form_for_equation
from tvb.adapters.simulator.integrator_forms import get_form_for_integrator
from tvb.adapters.simulator.noise_forms import get_form_for_noise
from tvb.adapters.simulator.subforms_mapping import SubformsEnum


class SubformHelper(object):
    class FormToConfigEnum(Enum):
        INTEGRATOR = partial(get_form_for_integrator)
        NOISE = partial(get_form_for_noise)
        EQUATION = partial(get_form_for_equation)

    @staticmethod
    def get_subform_for_field_value(ui_name, subform_key):
        current_class = SubformsEnum[subform_key].value().get(ui_name)
        return SubformHelper.FormToConfigEnum[subform_key].value(current_class)()

    @staticmethod
    def get_class_for_field_value(ui_name, subform_key):
        current_class = SubformsEnum[subform_key].value().get(ui_name)
        return current_class

