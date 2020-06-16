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
from tvb.adapters.simulator.equation_forms import get_form_for_equation
from tvb.adapters.simulator.form_with_ranges import FormWithRanges
from tvb.adapters.simulator.subforms_mapping import SubformsEnum, get_ui_name_to_equation_dict
from tvb.basic.neotraits.api import Attr, Range
from tvb.core.entities.transient.range_parameter import RangeParameter
from tvb.core.neotraits.forms import ScalarField, ArrayField, SelectField
from tvb.datatypes.equations import Equation
from tvb.simulator.noise import Noise, Additive, Multiplicative


def get_form_for_noise(noise_class):
    noise_class_to_form = {
        Additive: AdditiveNoiseForm,
        Multiplicative: MultiplicativeNoiseForm,
    }

    return noise_class_to_form.get(noise_class)


class NoiseForm(FormWithRanges):

    def get_subform_key(self):
        return SubformsEnum.NOISE.name

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

    def get_range_parameters(self):
        ntau_range_param = RangeParameter(Noise.ntau.field_name, float, Range(lo=0.0, hi=20.0, step=1.0))
        params_with_range_defined = super(NoiseForm, self).get_range_parameters()
        params_with_range_defined.append(ntau_range_param)

        return params_with_range_defined


class MultiplicativeNoiseForm(NoiseForm):

    def __init__(self, prefix=''):
        super(MultiplicativeNoiseForm, self).__init__(prefix)
        self.equation_choices = get_ui_name_to_equation_dict()
        default_equation = list(self.equation_choices.values())[0]

        self.nsig = ArrayField(Multiplicative.nsig, self)
        self.equation = SelectField(Attr(Equation, label='Equation', default=default_equation), self, name='equation',
                                    choices=self.equation_choices, subform=get_form_for_equation(default_equation))

    def fill_trait(self, datatype):
        super(MultiplicativeNoiseForm, self).fill_trait(datatype)
        datatype.nsig = self.nsig.data
        datatype.b = self.equation.data()

    def fill_from_trait(self, trait):
        # type: (Noise) -> None
        super(MultiplicativeNoiseForm, self).fill_from_trait(trait)
        self.equation.data = trait.b.__class__
