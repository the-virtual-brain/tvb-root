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

from tvb.datatypes.equations import *
from tvb.core.neotraits.forms import Form, FloatField, LabelField
from tvb.basic.neotraits.api import Float


class SpatialEquationsEnum(EquationsEnum):
    GAUSSIAN = (Gaussian, "Gaussian")
    MEXICAN_HAT = (DoubleGaussian, "Mexican-hat")
    SIGMOID = (Sigmoid, "Sigmoid")
    DISCRETE = (DiscreteEquation, "Discrete Equation")


class TemporalEquationsEnum(EquationsEnum):
    LINEAR = (Linear, "Linear")
    GAUSSIAN = (Gaussian, "Gaussian")
    MEXICAN_HAT = (DoubleGaussian, "Mexican-hat")
    SIGMOID = (Sigmoid, "Sigmoid")
    GENERALIZEDSIGMOID = (GeneralizedSigmoid, "GeneralizedSigmoid")
    SINUSOID = (Sinusoid, "Sinusoid")
    COSINE = (Cosine, "Cosine")
    ALPHA = (Alpha, "Alpha")
    PULSETRAIN = (PulseTrain, "PulseTrain")


class SurfaceModelEquationsEnum(TupleEnum):
    GAUSSIAN = (Gaussian, "Gaussian")
    SIGMOID = (Sigmoid, "Sigmoid")


class BoldMonitorEquationsEnum(EquationsEnum):
    Gamma_KERNEL = (Gamma, "Hrf Kernel: Gamma Kernel")
    DOUBLE_EXPONENTIAL_KERNEL = (DoubleExponential, "Hrf Kernel: Difference of Exponential")
    VOLTERRA_KERNEL = (FirstOrderVolterra, "Hrf Kernel: Volterra Kernel")
    MOG_KERNEL = (MixtureOfGammas, "Hrf Kernel: Mixture Of Gammas")


class TransferVectorEquationsEnum(EquationsEnum):
    IDENTITY = (Identity, "Identity")
    LINEAR = (Linear, "Linear")
    LINEAR_INTERVAL = (RescaleInterval, "Rescale to Interval")
    ABSOLUTE = (Absolute, "Absolute")
    LOGARITHM = (Logarithm, "Logarithm")


def get_ui_name_to_monitor_equation_dict():
    eq_name_to_class = {
        'HRF kernel: Gamma kernel': Gamma,
        'HRF kernel: Difference of Exponential': DoubleExponential,
        'HRF kernel: Volterra Kernel': FirstOrderVolterra,
        'HRF kernel: Mixture of Gammas': MixtureOfGammas
    }
    return eq_name_to_class


def get_equation_to_form_dict():
    coupling_class_to_form = {
        Linear: LinearEquationForm,
        RescaleInterval: RescaleIntervalEquationForm,
        Absolute: AbsoluteEquationForm,
        Identity: IdentityEquationForm,
        Logarithm: LogarithmEquationForm,
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

    @staticmethod
    def get_subform_key():
        return 'EQUATION'

    def get_traited_equation(self):
        return Equation

    def __init__(self):
        super(EquationForm, self).__init__()
        traited_equation = self.get_traited_equation().equation
        self.equation = LabelField(traited_equation, traited_equation.doc)
        for param_key, param in self.get_traited_equation().parameters.default().items():
            setattr(self, param_key, FloatField(Float(label=param_key, default=param), name=param_key))

    def fill_from_post(self, form_data):
        for field in self.fields:
            if field.name in form_data:
                field.fill_from_post(form_data)

    def fill_trait_partially(self, datatype, fields=None):
        if fields is None:
            fields = []

        for field_str in fields:
            datatype.parameters[field_str] = getattr(self, field_str).value

    def fill_trait(self, datatype):
        for param_key in datatype.parameters.keys():
            datatype.parameters[param_key] = getattr(self, param_key).value

    def fill_from_trait(self, trait):
        for param_key in self.get_traited_equation().parameters.default().keys():
            getattr(self, param_key).data = trait.parameters[param_key]


class LinearEquationForm(EquationForm):

    def get_traited_equation(self):
        return Linear


class RescaleIntervalEquationForm(EquationForm):

    def get_traited_equation(self):
        return RescaleInterval


class AbsoluteEquationForm(EquationForm):

    def get_traited_equation(self):
        return Absolute


class IdentityEquationForm(EquationForm):

    def get_traited_equation(self):
        return Identity


class LogarithmEquationForm(EquationForm):

    def get_traited_equation(self):
        return Logarithm


class GaussianEquationForm(EquationForm):

    def get_traited_equation(self):
        return Gaussian


class DoubleGaussianEquationForm(EquationForm):

    def get_traited_equation(self):
        return DoubleGaussian


class SigmoidEquationForm(EquationForm):

    def get_traited_equation(self):
        return Sigmoid


class GeneralizedSigmoidEquationForm(EquationForm):

    def get_traited_equation(self):
        return GeneralizedSigmoid


class SinusoidEquationForm(EquationForm):

    def get_traited_equation(self):
        return Sinusoid


class CosineEquationForm(EquationForm):

    def get_traited_equation(self):
        return Cosine


class AlphaEquationForm(EquationForm):

    def get_traited_equation(self):
        return Alpha


class PulseTrainEquationForm(EquationForm):

    def get_traited_equation(self):
        return PulseTrain


class GammaEquationForm(EquationForm):

    def get_traited_equation(self):
        return Gamma


class DoubleExponentialEquationForm(EquationForm):

    def get_traited_equation(self):
        return DoubleExponential


class FirstOrderVolterraEquationForm(EquationForm):

    def get_traited_equation(self):
        return FirstOrderVolterra


class MixtureOfGammasEquationForm(EquationForm):

    def get_traited_equation(self):
        return MixtureOfGammas
