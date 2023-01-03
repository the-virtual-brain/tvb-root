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
from tvb.adapters.forms.equation_forms import get_form_for_equation, TemporalEquationsEnum
from tvb.adapters.forms.form_with_ranges import FormWithRanges
from tvb.basic.neotraits.api import EnumAttr, Range
from tvb.core.entities.file.simulator.view_model import NoiseViewModel, AdditiveNoiseViewModel, \
    MultiplicativeNoiseViewModel
from tvb.core.entities.transient.range_parameter import RangeParameter
from tvb.core.neotraits.forms import ArrayField, SelectField, FloatField, IntField


def get_form_for_noise(noise_class):
    noise_class_to_form = {
        AdditiveNoiseViewModel: AdditiveNoiseForm,
        MultiplicativeNoiseViewModel: MultiplicativeNoiseForm,
    }

    return noise_class_to_form.get(noise_class)


class NoiseForm(FormWithRanges):

    @staticmethod
    def get_subform_key():
        return 'NOISE'

    def __init__(self):
        super(NoiseForm, self).__init__()
        self.ntau = FloatField(NoiseViewModel.ntau)
        self.noise_seed = IntField(NoiseViewModel.noise_seed)


class AdditiveNoiseForm(NoiseForm):

    def __init__(self):
        super(AdditiveNoiseForm, self).__init__()
        self.nsig = ArrayField(AdditiveNoiseViewModel.nsig)

    def get_range_parameters(self, prefix):
        ntau_range_param = RangeParameter(NoiseViewModel.ntau.field_name, float, Range(lo=0.0, hi=20.0, step=1.0))
        params_with_range_defined = super(NoiseForm, self).get_range_parameters(prefix)
        self.ensure_correct_prefix_for_param_name(ntau_range_param, prefix)
        params_with_range_defined.append(ntau_range_param)

        return params_with_range_defined


class MultiplicativeNoiseForm(NoiseForm):

    def __init__(self):
        super(MultiplicativeNoiseForm, self).__init__()
        self.nsig = ArrayField(MultiplicativeNoiseViewModel.nsig)
        self.equation = SelectField(EnumAttr(label='Equation', default=TemporalEquationsEnum.LINEAR),
                                    name='equation', subform=get_form_for_equation(TemporalEquationsEnum.LINEAR.value))

    def fill_trait(self, datatype):
        super(MultiplicativeNoiseForm, self).fill_trait(datatype)
        datatype.nsig = self.nsig.data
        if type(datatype.b) != self.equation.data.value:
            datatype.b = self.equation.data.instance

    def fill_from_trait(self, trait):
        # type: (NoiseViewModel) -> None
        super(MultiplicativeNoiseForm, self).fill_from_trait(trait)
        self.equation.data = type(trait.b)
