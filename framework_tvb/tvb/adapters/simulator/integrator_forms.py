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
from tvb.adapters.simulator.noise_forms import get_form_for_noise
from tvb.adapters.simulator.subforms_mapping import SubformsEnum, get_ui_name_to_noise_dict
from tvb.basic.neotraits.api import Attr
from tvb.core.entities.file.simulator.view_model import HeunDeterministicViewModel, HeunStochasticViewModel, \
    EulerDeterministicViewModel, EulerStochasticViewModel, RungeKutta4thOrderDeterministicViewModel, IdentityViewModel, \
    VODEViewModel, VODEStochasticViewModel, Dopri5ViewModel, Dopri5StochasticViewModel, Dop853ViewModel, \
    Dop853StochasticViewModel, IntegratorViewModel, NoiseViewModel
from tvb.core.entities.file.simulator.view_model import IntegratorStochasticViewModel
from tvb.core.neotraits.forms import Form, ScalarField, SelectField


def get_integrator_to_form_dict():
    integrator_class_to_form = {
        HeunDeterministicViewModel: HeunDeterministicIntegratorForm,
        HeunStochasticViewModel: HeunStochasticIntegratorForm,
        EulerDeterministicViewModel: EulerDeterministicIntegratorForm,
        EulerStochasticViewModel: EulerStochasticIntegratorForm,
        RungeKutta4thOrderDeterministicViewModel: RungeKutta4thOrderDeterministicIntegratorForm,
        IdentityViewModel: IdentityIntegratorForm,
        VODEViewModel: VODEIntegratorForm,
        VODEStochasticViewModel: VODEStochasticIntegratorForm,
        Dopri5ViewModel: Dopri5IntegratorForm,
        Dopri5StochasticViewModel: Dopri5StochasticIntegratorForm,
        Dop853ViewModel: Dop853IntegratorForm,
        Dop853StochasticViewModel: Dop853StochasticIntegratorForm
    }
    return integrator_class_to_form


def get_form_for_integrator(integrator_class):
    return get_integrator_to_form_dict().get(integrator_class)


class IntegratorForm(Form):

    def get_subform_key(self):
        return SubformsEnum.INTEGRATOR.name

    def __init__(self, prefix=''):
        super(IntegratorForm, self).__init__(prefix)
        self.dt = ScalarField(IntegratorViewModel.dt, self)


class IntegratorStochasticForm(IntegratorForm):
    template = 'form_fields/select_field.html'

    def __init__(self, prefix=''):
        super(IntegratorStochasticForm, self).__init__(prefix)
        self.noise_choices = get_ui_name_to_noise_dict()
        default_noise = list(self.noise_choices.values())[0]

        self.noise = SelectField(Attr(NoiseViewModel, label='Noise', default=default_noise), self, name='noise',
                                 choices=self.noise_choices, subform=get_form_for_noise(default_noise))

    def fill_trait(self, datatype):
        super(IntegratorStochasticForm, self).fill_trait(datatype)
        if type(datatype.noise) != self.noise.data:
            datatype.noise = self.noise.data()

    def fill_from_trait(self, trait):
        # type: (IntegratorStochasticViewModel) -> None
        self.noise.data = trait.noise.__class__


class HeunDeterministicIntegratorForm(IntegratorForm):

    def __init__(self, prefix=''):
        super(HeunDeterministicIntegratorForm, self).__init__(prefix)


class HeunStochasticIntegratorForm(IntegratorStochasticForm):

    def __init__(self, prefix=''):
        super(HeunStochasticIntegratorForm, self).__init__(prefix)


class EulerDeterministicIntegratorForm(IntegratorForm):

    def __init__(self, prefix=''):
        super(EulerDeterministicIntegratorForm, self).__init__(prefix)


class EulerStochasticIntegratorForm(IntegratorStochasticForm):

    def __init__(self, prefix=''):
        super(EulerStochasticIntegratorForm, self).__init__(prefix)


class RungeKutta4thOrderDeterministicIntegratorForm(IntegratorForm):

    def __init__(self, prefix=''):
        super(RungeKutta4thOrderDeterministicIntegratorForm, self).__init__(prefix)


class IdentityIntegratorForm(IntegratorForm):

    def __init__(self, prefix=''):
        super(IdentityIntegratorForm, self).__init__(prefix)


class VODEIntegratorForm(IntegratorForm):

    def __init__(self, prefix=''):
        super(VODEIntegratorForm, self).__init__(prefix)


class VODEStochasticIntegratorForm(IntegratorStochasticForm):

    def __init__(self, prefix=''):
        super(VODEStochasticIntegratorForm, self).__init__(prefix)


class Dopri5IntegratorForm(IntegratorForm):

    def __init__(self, prefix=''):
        super(Dopri5IntegratorForm, self).__init__(prefix)


class Dopri5StochasticIntegratorForm(IntegratorStochasticForm):

    def __init__(self, prefix=''):
        super(Dopri5StochasticIntegratorForm, self).__init__(prefix)


class Dop853IntegratorForm(IntegratorForm):

    def __init__(self, prefix=''):
        super(Dop853IntegratorForm, self).__init__(prefix)


class Dop853StochasticIntegratorForm(IntegratorStochasticForm):

    def __init__(self, prefix=''):
        super(Dop853StochasticIntegratorForm, self).__init__(prefix)
