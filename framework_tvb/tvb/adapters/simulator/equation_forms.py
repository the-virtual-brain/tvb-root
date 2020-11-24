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
from tvb.adapters.simulator.subforms_mapping import SubformsEnum, get_ui_name_to_equation_dict
from tvb.datatypes.equations import *
from tvb.core.neotraits.forms import Form, FloatField, StrField
from tvb.basic.neotraits.api import Float


def get_ui_name_for_equation(equation_class):
    equation_to_ui_name = dict((v, k) for k, v in get_ui_name_to_equation_dict().items())
    return equation_to_ui_name.get(equation_class)


def get_ui_name_to_monitor_equation_dict():
    eq_name_to_class = {
        'HRF kernel: Gamma kernel': Gamma,
        'HRF kernel: Difference of Exponentials': DoubleExponential,
        'HRF kernel: Volterra Kernel': FirstOrderVolterra,
        'HRF kernel: Mixture of Gammas': MixtureOfGammas
    }
    return eq_name_to_class


def get_equation_to_form_dict():
    coupling_class_to_form = {
        Linear: LinearEquationForm,
        Gaussian: GaussianEquationForm,
        DoubleGaussian: DoubleGaussianEquationForm,
        Sigmoid: SigmoidEquationForm,
        GeneralizedSigmoid: GeneralizedSigmoidEquationForm,
        Sinusoid: SinusoidEquationForm,
        Cosine: CosineEquationForm,
        Alpha: AlphaEquationForm,
        PulseTrain: PulseTrainEquationForm,
        Gamma: GammaEquationForm,
        DoubleExponential: DoubleExponentialEquationForm,
        FirstOrderVolterra: FirstOrderVolterraEquationForm,
        MixtureOfGammas: MixtureOfGammasEquationForm
    }

    return coupling_class_to_form


def get_form_for_equation(equation_class):
    return get_equation_to_form_dict().get(equation_class)


class EquationForm(Form):

    def get_subform_key(self):
        return SubformsEnum.EQUATION.name

    def get_traited_equation(self):
        return Equation

    def __init__(self):
        super(EquationForm, self).__init__()
        self.equation = StrField(self.get_traited_equation().equation, self.project_id)
        for param_key, param in self.get_traited_equation().parameters.default().items():
            setattr(self, param_key, FloatField(Float(label=param_key, default=param), self.project_id, name=param_key))

    def fill_from_post(self, form_data):
        for field in self.fields:
            if field.name in form_data:
                field.fill_from_post(form_data)

    def fill_trait_partially(self, datatype, fields = None):
        if fields is None:
            fields = []

        for field_str in fields:
            datatype.parameters[field_str] = getattr(self, field_str).value


class LinearEquationForm(EquationForm):

    def get_traited_equation(self):
        return Linear

    def __init__(self):
        super(LinearEquationForm, self).__init__()

    def fill_trait(self, datatype):
        datatype.parameters['a'] = self.a.value
        datatype.parameters['b'] = self.b.value

    def fill_from_trait(self, trait):
        self.a.data = trait.parameters['a']
        self.b.data = trait.parameters['b']


class GaussianEquationForm(EquationForm):

    def get_traited_equation(self):
        return Gaussian

    def __init__(self):
        super(GaussianEquationForm, self).__init__()

    def fill_trait(self, datatype):
        datatype.parameters['amp'] = self.amp.value
        datatype.parameters['sigma'] = self.sigma.value
        datatype.parameters['midpoint'] = self.midpoint.value
        datatype.parameters['offset'] = self.offset.value

    def fill_from_trait(self, trait):
        self.amp.data = trait.parameters['amp']
        self.sigma.data = trait.parameters['sigma']
        self.midpoint.data = trait.parameters['midpoint']
        self.offset.data = trait.parameters['offset']


class DoubleGaussianEquationForm(EquationForm):

    def get_traited_equation(self):
        return DoubleGaussian

    def __init__(self):
        super(DoubleGaussianEquationForm, self).__init__()

    def fill_trait(self, datatype):
        datatype.parameters['amp_1'] = self.amp_1.value
        datatype.parameters['amp_2'] = self.amp_2.value
        datatype.parameters['sigma_1'] = self.sigma_1.value
        datatype.parameters['sigma_2'] = self.sigma_2.value
        datatype.parameters['midpoint_1'] = self.midpoint_1.value
        datatype.parameters['midpoint_2'] = self.midpoint_2.value

    def fill_from_trait(self, trait):
        self.amp_1.data = trait.parameters['amp_1']
        self.amp_2.data = trait.parameters['amp_2']
        self.sigma_1.data = trait.parameters['sigma_1']
        self.sigma_2.data = trait.parameters['sigma_2']
        self.midpoint_1.data = trait.parameters['midpoint_1']
        self.midpoint_2.data = trait.parameters['midpoint_2']


class SigmoidEquationForm(EquationForm):

    def get_traited_equation(self):
        return Sigmoid

    def __init__(self):
        super(SigmoidEquationForm, self).__init__()

    def fill_trait(self, datatype):
        datatype.parameters['amp'] = self.amp.value
        datatype.parameters['radius'] = self.radius.value
        datatype.parameters['sigma'] = self.sigma.value
        datatype.parameters['offset'] = self.offset.value

    def fill_from_trait(self, trait):
        self.amp.data = trait.parameters['amp']
        self.radius.data = trait.parameters['radius']
        self.sigma.data = trait.parameters['sigma']
        self.offset.data = trait.parameters['offset']


class GeneralizedSigmoidEquationForm(EquationForm):

    def get_traited_equation(self):
        return GeneralizedSigmoid

    def __init__(self):
        super(GeneralizedSigmoidEquationForm, self).__init__()

    def fill_trait(self, datatype):
        datatype.parameters['low'] = self.low.value
        datatype.parameters['high'] = self.high.value
        datatype.parameters['midpoint'] = self.midpoint.value
        datatype.parameters['sigma'] = self.sigma.value

    def fill_from_trait(self, trait):
        self.low.data = trait.parameters['low']
        self.high.data = trait.parameters['high']
        self.midpoint.data = trait.parameters['midpoint']
        self.sigma.data = trait.parameters['sigma']


class SinusoidEquationForm(EquationForm):

    def get_traited_equation(self):
        return Sinusoid

    def __init__(self):
        super(SinusoidEquationForm, self).__init__()

    def fill_trait(self, datatype):
        datatype.parameters['amp'] = self.amp.value
        datatype.parameters['frequency'] = self.frequency.value

    def fill_from_trait(self, trait):
        self.amp.data = trait.parameters['amp']
        self.frequency.data = trait.parameters['frequency']


class CosineEquationForm(EquationForm):

    def get_traited_equation(self):
        return Cosine

    def __init__(self):
        super(CosineEquationForm, self).__init__()

    def fill_trait(self, datatype):
        datatype.parameters['amp'] = self.amp.value
        datatype.parameters['frequency'] = self.frequency.value

    def fill_from_trait(self, trait):
        self.amp.data = trait.parameters['amp']
        self.frequency.data = trait.parameters['frequency']


class AlphaEquationForm(EquationForm):

    def get_traited_equation(self):
        return Alpha

    def __init__(self):
        super(AlphaEquationForm, self).__init__()

    def fill_trait(self, datatype):
        datatype.parameters['onset'] = self.onset.value
        datatype.parameters['alpha'] = self.alpha.value
        datatype.parameters['beta'] = self.beta.value

    def fill_from_trait(self, trait):
        self.onset.data = trait.parameters['onset']
        self.alpha.data = trait.parameters['alpha']
        self.beta.data = trait.parameters['beta']


class PulseTrainEquationForm(EquationForm):

    def get_traited_equation(self):
        return PulseTrain

    def __init__(self):
        super(PulseTrainEquationForm, self).__init__()

    def fill_trait(self, datatype):
        datatype.parameters['T'] = self.T.value
        datatype.parameters['tau'] = self.tau.value
        datatype.parameters['amp'] = self.amp.value
        datatype.parameters['onset'] = self.onset.value

    def fill_from_trait(self, trait):
        self.T.data = trait.parameters['T']
        self.tau.data = trait.parameters['tau']
        self.amp.data = trait.parameters['amp']
        self.onset.data = trait.parameters['onset']


class GammaEquationForm(EquationForm):

    def get_traited_equation(self):
        return Gamma

    def __init__(self):
        super(GammaEquationForm, self).__init__()

    def fill_trait(self, datatype):
        datatype.parameters['tau'] = self.tau.value
        datatype.parameters['n'] = self.n.value
        datatype.parameters['factorial'] = self.factorial.value
        datatype.parameters['a'] = self.a.value

    def fill_from_trait(self, trait):
        self.tau.data = trait.parameters['tau']
        self.n.data = trait.parameters['n']
        self.factorial.data = trait.parameters['factorial']
        self.a.data = trait.parameters['a']


class DoubleExponentialEquationForm(EquationForm):

    def get_traited_equation(self):
        return DoubleExponential

    def __init__(self):
        super(DoubleExponentialEquationForm, self).__init__()

    def fill_trait(self, datatype):
        datatype.parameters['tau_1'] = self.tau_1.value
        datatype.parameters['tau_2'] = self.tau_2.value
        datatype.parameters['a'] = self.a.value
        datatype.parameters['f_1'] = self.f_1.value
        datatype.parameters['f_2'] = self.f_2.value
        datatype.parameters['pi'] = self.pi.value
        datatype.parameters['amp_1'] = self.amp_1.value
        datatype.parameters['amp_2'] = self.amp_2.value

    def fill_from_trait(self, trait):
        self.tau_1.data = trait.parameters['tau_1']
        self.tau_2.data = trait.parameters['tau_2']
        self.a.data = trait.parameters['a']
        self.f_1.data = trait.parameters['f_1']
        self.f_2.data = trait.parameters['f_2']
        self.pi.data = trait.parameters['pi']
        self.amp_1.data = trait.parameters['amp_1']
        self.amp_2.data = trait.parameters['amp_2']


class FirstOrderVolterraEquationForm(EquationForm):

    def get_traited_equation(self):
        return FirstOrderVolterra

    def __init__(self):
        super(FirstOrderVolterraEquationForm, self).__init__()

    def fill_trait(self, datatype):
        datatype.parameters['tau_s'] = self.tau_s.value
        datatype.parameters['tau_f'] = self.tau_f.value
        datatype.parameters['k_1'] = self.k_1.value
        datatype.parameters['V_0'] = self.V_0.value

    def fill_from_trait(self, trait):
        self.tau_s.data = trait.parameters['tau_s']
        self.tau_f.data = trait.parameters['tau_f']
        self.k_1.data = trait.parameters['k_1']
        self.V_0.data = trait.parameters['V_0']


class MixtureOfGammasEquationForm(EquationForm):

    def get_traited_equation(self):
        return MixtureOfGammas

    def __init__(self):
        super(MixtureOfGammasEquationForm, self).__init__()

    def fill_trait(self, datatype):
        datatype.parameters['a_1'] = self.a_1.value
        datatype.parameters['a_2'] = self.a_2.value
        datatype.parameters['l'] = self.l.value
        datatype.parameters['c'] = self.c.value
        datatype.parameters['gamma_a_1'] = self.gamma_a_1.value
        datatype.parameters['gamma_a_2'] = self.gamma_a_2.value

    def fill_from_trait(self, trait):
        self.a_1.data = trait.parameters['a_1']
        self.a_2.data = trait.parameters['a_2']
        self.l.data = trait.parameters['l']
        self.c.data = trait.parameters['c']
        self.gamma_a_1.data = trait.parameters['gamma_a_1']
        self.gamma_a_2.data = trait.parameters['gamma_a_2']
