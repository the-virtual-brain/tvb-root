# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""
from tvb.adapters.simulator.equation_forms import get_form_for_equation, SurfaceModelEquationsEnum
from tvb.adapters.simulator.form_methods import SURFACE_EQ_KEY
from tvb.basic.neotraits.api import Attr, Float, EnumAttr, TupleEnum, TVBEnum
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.neotraits.forms import Form, FormField, SelectField, FloatField, DynamicSelectField
from tvb.core.neotraits.view_model import Str

### SESSION KEY for ContextModelParameter entity.
KEY_CONTEXT_MPS = "ContextForModelParametersOnSurface"


class SurfaceModelParametersForm(ABCAdapterForm):
    NAME_EQATION_PARAMS_DIV = 'equation_params'
    default_equation = SurfaceModelEquationsEnum.GAUSSIAN

    def __init__(self, model_params):
        super(SurfaceModelParametersForm, self).__init__()

        model_labels = [param.name for param in model_params]
        model_mathjax_representations = [param.label for param in model_params]
        self.model_param = DynamicSelectField(Str(label='Model parameter'), choices=model_labels, name='model_param',
                                              ui_values=model_mathjax_representations)
        self.equation = SelectField(EnumAttr(label='Equation', default=self.default_equation),
                                    name='equation',
                                    subform=get_form_for_equation(self.default_equation.value),
                                    session_key=KEY_CONTEXT_MPS, form_key=SURFACE_EQ_KEY)

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


class EquationPlotForm(Form):
    def __init__(self):
        super(EquationPlotForm, self).__init__()
        self.min_x = FloatField(Float(label='Min distance(mm)', default=0,
                                      doc="The minimum value of the x-axis for spatial equation plot."),
                                name='min_x')
        self.max_x = FloatField(Float(label='Max distance(mm)', default=100,
                                      doc="The maximum value of the x-axis for spatial equation plot."),
                                name='max_x')

    def fill_from_post(self, form_data):
        if self.min_x.name in form_data:
            self.min_x.fill_from_post(form_data)
        if self.max_x.name in form_data:
            self.max_x.fill_from_post(form_data)