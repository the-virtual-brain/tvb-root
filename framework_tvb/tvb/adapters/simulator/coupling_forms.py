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
from tvb.simulator.coupling import *
from tvb.adapters.simulator.form_with_ranges import FormWithRanges
from tvb.core.neotraits.forms import ArrayField, BoolField


def get_coupling_to_form_dict():
    coupling_class_to_form = {
        Linear: LinearCouplingForm,
        Scaling: ScalingCouplingForm,
        HyperbolicTangent: HyperbolicTangentCouplingForm,
        Sigmoidal: SigmoidalCouplingForm,
        SigmoidalJansenRit: SigmoidalJansenRitForm,
        PreSigmoidal: PreSigmoidalCouplingForm,
        Difference: DifferenceCouplingForm,
        Kuramoto: KuramotoCouplingForm
    }
    return coupling_class_to_form


def get_ui_name_to_coupling_dict():
    ui_name_to_coupling = {}
    for coupling_class in get_coupling_to_form_dict():
        ui_name_to_coupling.update({coupling_class.__name__: coupling_class})

    return ui_name_to_coupling


def get_form_for_coupling(coupling_class):
    return get_coupling_to_form_dict().get(coupling_class)


class LinearCouplingForm(FormWithRanges):

    def __init__(self):
        super(LinearCouplingForm, self).__init__()
        self.a = ArrayField(Linear.a)
        self.b = ArrayField(Linear.b)


class ScalingCouplingForm(FormWithRanges):

    def __init__(self):
        super(ScalingCouplingForm, self).__init__()
        self.a = ArrayField(Scaling.a)


class HyperbolicTangentCouplingForm(FormWithRanges):

    def __init__(self):
        super(HyperbolicTangentCouplingForm, self).__init__()
        self.a = ArrayField(HyperbolicTangent.a)
        self.b = ArrayField(HyperbolicTangent.b)
        self.midpoint = ArrayField(HyperbolicTangent.midpoint)
        self.sigma = ArrayField(HyperbolicTangent.sigma)


class SigmoidalCouplingForm(FormWithRanges):

    def __init__(self):
        super(SigmoidalCouplingForm, self).__init__()
        self.cmin = ArrayField(Sigmoidal.cmin)
        self.cmax = ArrayField(Sigmoidal.cmax)
        self.midpoint = ArrayField(Sigmoidal.midpoint)
        self.a = ArrayField(Sigmoidal.a)
        self.sigma = ArrayField(Sigmoidal.sigma)


class SigmoidalJansenRitForm(FormWithRanges):

    def __init__(self):
        super(SigmoidalJansenRitForm, self).__init__()
        self.cmin = ArrayField(SigmoidalJansenRit.cmin)
        self.cmax = ArrayField(SigmoidalJansenRit.cmax)
        self.midpoint = ArrayField(SigmoidalJansenRit.midpoint)
        self.r = ArrayField(SigmoidalJansenRit.r)
        self.a = ArrayField(SigmoidalJansenRit.a)


class PreSigmoidalCouplingForm(FormWithRanges):

    def __init__(self):
        super(PreSigmoidalCouplingForm, self).__init__()
        self.H = ArrayField(PreSigmoidal.H)
        self.Q = ArrayField(PreSigmoidal.Q)
        self.G = ArrayField(PreSigmoidal.G)
        self.P = ArrayField(PreSigmoidal.P)
        self.theta = ArrayField(PreSigmoidal.theta)
        self.dynamic = BoolField(PreSigmoidal.dynamic)
        self.globalT= BoolField(PreSigmoidal.globalT)


class DifferenceCouplingForm(FormWithRanges):

    def __init__(self):
        super(DifferenceCouplingForm, self).__init__()
        self.a = ArrayField(Difference.a)


class KuramotoCouplingForm(FormWithRanges):

    def __init__(self):
        super(KuramotoCouplingForm, self).__init__()
        self.a = ArrayField(Kuramoto.a)