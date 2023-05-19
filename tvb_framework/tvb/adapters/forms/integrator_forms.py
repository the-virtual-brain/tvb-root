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
from tvb.adapters.forms.noise_forms import get_form_for_noise
from tvb.basic.neotraits.api import TupleEnum, EnumAttr
from tvb.core.entities.file.simulator.view_model import HeunDeterministicViewModel, HeunStochasticViewModel, \
    EulerDeterministicViewModel, EulerStochasticViewModel, RungeKutta4thOrderDeterministicViewModel, IdentityViewModel, \
    VODEViewModel, VODEStochasticViewModel, Dopri5ViewModel, Dopri5StochasticViewModel, Dop853ViewModel, \
    Dop853StochasticViewModel, IntegratorViewModel, AdditiveNoiseViewModel, MultiplicativeNoiseViewModel
from tvb.core.entities.file.simulator.view_model import IntegratorStochasticViewModel
from tvb.core.neotraits.forms import Form, SelectField, FloatField


def get_integrator_name_list():
    return ['Heun', 'Stochastic Heun', 'Euler', 'Euler-Maruyama', 'Runge-Kutta 4th order', 'Difference equation',
            'Variable-order Adams / BDF', 'Stochastic variable-order Adams / BDF', 'Dormand-Prince, order(4, 5)',
            'Stochastic Dormand-Prince, order (4, 5)', 'Dormand-Prince, order 8 (5, 3)',
            'Stochastic Dormand-Prince, order 8 (5, 3)']


def get_integrator_to_form_dict():
    integrator_class_to_form = {
        HeunDeterministicViewModel: IntegratorForm,
        HeunStochasticViewModel: IntegratorStochasticForm,
        EulerDeterministicViewModel: IntegratorForm,
        EulerStochasticViewModel: IntegratorStochasticForm,
        RungeKutta4thOrderDeterministicViewModel: IntegratorForm,
        IdentityViewModel: IntegratorForm,
        VODEViewModel: IntegratorForm,
        VODEStochasticViewModel: IntegratorStochasticForm,
        Dopri5ViewModel: IntegratorForm,
        Dopri5StochasticViewModel: IntegratorStochasticForm,
        Dop853ViewModel: IntegratorForm,
        Dop853StochasticViewModel: IntegratorStochasticForm
    }
    return integrator_class_to_form


def get_form_for_integrator(integrator_class):
    return get_integrator_to_form_dict().get(integrator_class)


class NoiseTypesEnum(TupleEnum):
    ADDITIVE = (AdditiveNoiseViewModel, "Additive")
    MULTIPLICATIVE = (MultiplicativeNoiseViewModel, "Multiplicative")


class IntegratorForm(Form):

    @staticmethod
    def get_subform_key():
        return 'INTEGRATOR'

    def __init__(self, is_dt_disabled=False):
        super(IntegratorForm, self).__init__()
        self.is_dt_disabled = is_dt_disabled
        self.dt = FloatField(IntegratorViewModel.dt)

    def fill_from_trait(self, trait):
        # type: (IntegratorViewModel) -> None
        super(IntegratorForm, self).fill_from_trait(trait)

        if self.is_dt_disabled:
            self.dt.disabled = True


class IntegratorStochasticForm(IntegratorForm):
    template = 'form_fields/select_field.html'

    def __init__(self, is_dt_disabled=False):
        super(IntegratorStochasticForm, self).__init__(is_dt_disabled)

        self.noise = SelectField(EnumAttr(label='Noise', default=NoiseTypesEnum.ADDITIVE), name='noise',
                                 subform=get_form_for_noise(NoiseTypesEnum.ADDITIVE.value))

    def fill_trait(self, datatype):
        super(IntegratorStochasticForm, self).fill_trait(datatype)
        if self.noise.data and type(datatype.noise) != self.noise.data.value:
            datatype.noise = self.noise.data.instance

    def fill_from_trait(self, trait):
        # type: (IntegratorStochasticViewModel) -> None
        super(IntegratorStochasticForm, self).fill_from_trait(trait)
        self.noise.data = trait.noise.__class__
