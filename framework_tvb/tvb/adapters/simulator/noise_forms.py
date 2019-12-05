from tvb.basic.neotraits.api import Range
from tvb.simulator.noise import Noise, Additive, Multiplicative
from tvb.adapters.simulator.equation_forms import get_ui_name_to_equation_dict
from tvb.adapters.simulator.form_with_ranges import FormWithRanges
from tvb.adapters.simulator.range_parameter import RangeParameter
from tvb.core.neotraits.forms import ScalarField, ArrayField, SimpleSelectField


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


class NoiseForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(NoiseForm, self).__init__(prefix)
        self.ntau = ScalarField(Noise.ntau, self)
        self.noise_seed = ScalarField(Noise.noise_seed, self)
        # TODO: should we display something for random_stream?
        # self.random_stream = ScalarField(Noise.random_stream)

    def fill_from_trait(self, trait):
        # type: (Noise) -> None
        # super(NoiseForm, self).fill_from_trait(trait)
        self.ntau.data = trait.ntau
        self.noise_seed.data = trait.noise_seed
        self.nsig.data = trait.nsig

class AdditiveNoiseForm(NoiseForm):

    def __init__(self, prefix=''):
        super(AdditiveNoiseForm, self).__init__(prefix)
        self.nsig = ArrayField(Additive.nsig, self)

    def get_range_parameters(self):
        ntau_range_param = RangeParameter(Noise.ntau.field_name, float, Range(lo=0.0, hi=20.0, step=1.0))
        params_with_range_defined = super(NoiseForm, self).get_range_parameters()
        params_with_range_defined.append(ntau_range_param)

        return params_with_range_defined


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
