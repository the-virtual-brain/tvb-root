from tvb.simulator.integrators import Integrator, HeunDeterministic, HeunStochastic, EulerDeterministic, \
    EulerStochastic, Identity, VODE, VODEStochastic, Dopri5, Dopri5Stochastic, Dop853, Dop853Stochastic, \
    RungeKutta4thOrderDeterministic

from tvb.adapters.simulator.noise_forms import get_ui_name_to_noise_dict
from tvb.core.neotraits._forms import Form, ScalarField, ArrayField, SimpleSelectField


def get_integrator_to_form_dict():
    integrator_class_to_form = {
        HeunDeterministic: HeunDeterministicIntegratorForm,
        HeunStochastic: HeunStochasticIntegratorForm,
        EulerDeterministic: EulerDeterministicIntegratorForm,
        EulerStochastic: EulerStochasticIntegratorForm,
        RungeKutta4thOrderDeterministic: RungeKutta4thOrderDeterministicIntegratorForm,
        Identity: IdentityIntegratorForm,
        VODE: VODEIntegratorForm,
        VODEStochastic: VODEStochasticIntegratorForm,
        Dopri5: Dopri5IntegratorForm,
        Dopri5Stochastic: Dopri5StochasticIntegratorForm,
        Dop853: Dop853IntegratorForm,
        Dop853Stochastic: Dop853StochasticIntegratorForm
    }
    return integrator_class_to_form


def get_ui_name_to_integrator_dict():
    ui_name_to_integrator = {
        'Heun': HeunDeterministic,
        'Stochastic Heun': HeunStochastic,
        'Euler': EulerDeterministic,
        'Euler-Maruyama': EulerStochastic,
        'Runge-Kutta 4th order': RungeKutta4thOrderDeterministic,
        '"Difference equation': Identity,
        'Variable-order Adams / BDF': VODE,
        'Stochastic variable-order Adams / BDF': VODEStochastic,
        'Dormand-Prince, order (4, 5)': Dopri5,
        'Stochastic Dormand-Prince, order (4, 5)': Dopri5Stochastic,
        'Dormand-Prince, order 8 (5, 3)': Dop853,
        'Stochastic Dormand-Prince, order 8 (5, 3)': Dop853Stochastic,

    }
    return ui_name_to_integrator


def get_form_for_integrator(integrator_class):
    return get_integrator_to_form_dict().get(integrator_class)


class IntegratorForm(Form):

    def __init__(self, prefix=''):
        super(IntegratorForm, self).__init__(prefix)
        self.dt = ScalarField(Integrator.dt, self)
        self.clamped_state_variable_indices = ArrayField(Integrator.clamped_state_variable_indices, self)
        self.clamped_state_variable_values = ArrayField(Integrator.clamped_state_variable_values, self)


class IntegratorStochasticForm(IntegratorForm):
    template = 'select_field.jinja2'

    def __init__(self, prefix=''):
        super(IntegratorStochasticForm, self).__init__(prefix)
        # TODO: show select box with Noise types
        # self.noise = FormField(MultiplicativeNoiseForm, self, name='noise', label='Noise')
        self.noise_choices = get_ui_name_to_noise_dict()
        self.noise = SimpleSelectField(self.noise_choices, self, name='noise', required=True, label='Noise')

    def fill_trait(self, datatype):
        super(IntegratorStochasticForm, self).fill_trait(datatype)
        datatype.noise = self.noise.data()

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
