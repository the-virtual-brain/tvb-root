from tvb.simulator.noise import Noise, Additive, Multiplicative
from tvb.adapters.simulator.equation_forms import get_ui_name_to_equation_dict
from tvb.core.neotraits._forms import Form, ScalarField, ArrayField, SimpleSelectField


def get_form_for_noise(noise_class):
    noise_class_to_form = {
        Additive: AdditiveNoiseForm,
        Multiplicative: MultiplicativeNoiseForm,
    }

    return noise_class_to_form.get(noise_class)


def get_ui_name_to_noise_dict():
    ui_name_to_noise = {
        'Additive': Additive,
        'Multiplicative': Multiplicative
    }
    return ui_name_to_noise


class NoiseForm(Form):

    def __init__(self, prefix=''):
        super(NoiseForm, self).__init__(prefix)
        self.ntau = ScalarField(Noise.ntau, self)
        self.noise_seed = ScalarField(Noise.noise_seed, self)
        # TODO: should we display something for random_stream?
        # self.random_stream = ScalarField(Noise.random_stream)


class AdditiveNoiseForm(NoiseForm):

    def __init__(self, prefix=''):
        super(AdditiveNoiseForm, self).__init__(prefix)
        self.nsig = ArrayField(Additive.nsig, self)


class MultiplicativeNoiseForm(NoiseForm):

    def __init__(self, prefix=''):
        super(MultiplicativeNoiseForm, self).__init__(prefix)
        self.nsig = ArrayField(Multiplicative.nsig, self)
        self.equation_choices = get_ui_name_to_equation_dict()
        self.equation = SimpleSelectField(self.equation_choices, self, name='equation', required=True, label='Equation')

    def fill_trait(self, datatype):
        super(MultiplicativeNoiseForm, self).fill_trait(datatype)
        datatype.nsig = self.nsig.data
        datatype.b = self.equation.data()
