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
.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
"""

from tvb.adapters.forms.equation_forms import get_form_for_equation
from tvb.basic.neotraits.api import TupleEnum, EnumAttr
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.neotraits.forms import DynamicSelectField, SelectField, FormField
from tvb.core.neotraits.view_model import Str
from tvb.datatypes.equations import Gaussian, Sigmoid


class SurfaceModelEquationsEnum(TupleEnum):
    GAUSSIAN = (Gaussian, "Gaussian")
    SIGMOID = (Sigmoid, "Sigmoid")


### SESSION KEY for ContextModelParameter entity.
KEY_CONTEXT_MPS = "ContextForModelParametersOnSurface"


class SurfaceModelParametersForm(ABCAdapterForm):
    NAME_EQATION_PARAMS_DIV = 'equation_params'
    default_equation = SurfaceModelEquationsEnum.GAUSSIAN
    equation_field_label = 'Equation'

    def __init__(self, model_params):
        super(SurfaceModelParametersForm, self).__init__()

        model_labels = [param.name for param in model_params]
        model_mathjax_representations = [param.label for param in model_params]
        self.model_param = DynamicSelectField(Str(label='Model parameter'), choices=model_labels, name='model_param',
                                              ui_values=model_mathjax_representations)
        self.equation = SelectField(EnumAttr(label=self.equation_field_label, default=self.default_equation),
                                    name='equation', subform=get_form_for_equation(self.default_equation.value),
                                    session_key=KEY_CONTEXT_MPS)

    @staticmethod
    def get_required_datatype():
        return None

    @staticmethod
    def get_input_name():
        return None

    @staticmethod
    def get_filters():
        return None

    def fill_from_trait(self, trait):
        self.equation.data = type(trait)
        self.equation.subform_field = FormField(get_form_for_equation(type(trait)),
                                                self.NAME_EQATION_PARAMS_DIV)
        self.equation.subform_field.form.fill_from_trait(trait)