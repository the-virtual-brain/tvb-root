# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

from tvb.simulator.models import *
from tvb.adapters.simulator.form_with_ranges import FormWithRanges
from tvb.core.neotraits.forms import Form, ArrayField, MultiSelectField


def get_model_to_form_dict():
    model_class_to_form = {
        Generic2dOscillator: Generic2dOscillatorModelForm,
        Kuramoto: KuramotoModelForm,
        SupHopf: SupHopfModelForm,
        Hopfield: HopfieldModelForm,
        Epileptor: EpileptorModelForm,
        Epileptor2D: Epileptor2DModelForm,
        EpileptorCodim3: EpileptorCodim3ModelForm,
        EpileptorCodim3SlowMod: EpileptorCodim3SlowModModelForm,
        EpileptorRestingState: EpileptorRestingStateModelForm,
        JansenRit: JansenRitModelForm,
        ZetterbergJansen: ZetterbergJansenModelForm,
        ReducedWongWang: ReducedWongWangModelForm,
        ReducedWongWangExcInh: ReducedWongWangExcInhModelForm,
        ReducedSetFitzHughNagumo: ReducedSetFitzHughNagumoModelForm,
        ReducedSetHindmarshRose: ReducedSetHindmarshRoseModelForm,
        ZerlautFirstOrder: ZerlautFirstOrderModelForm,
        ZerlautSecondOrder: ZerlautSecondOrderModelForm,
        Linear: LinearModelForm,
        WilsonCowan: WilsonCowanModelForm,
        LarterBreakspear: LarterBreakspearModelForm
    }

    return model_class_to_form


def get_ui_name_to_model():
    ui_name_to_model = {
        'Generic 2d Oscillator': Generic2dOscillator,
        'Kuramoto Oscillator': Kuramoto,
        'supHopf': SupHopf,
        'Hopfield': Hopfield,
        'Epileptor': Epileptor,
        'Epileptor2D': Epileptor2D,
        'Epileptor codim 3': EpileptorCodim3,
        'Epileptor codim 3 ultra-slow modulations': EpileptorCodim3SlowMod,
        'EpileptorRestingState': EpileptorRestingState,
        'Jansen-Rit': JansenRit,
        'Zetterberg-Jansen': ZetterbergJansen,
        'Reduced Wong-Wang': ReducedWongWang,
        'Reduced Wong-Wang with Excitatory and Inhibitory Coupled Populations': ReducedWongWangExcInh,
        'Stefanescu-Jirsa 2D': ReducedSetFitzHughNagumo,
        'Stefanescu-Jirsa 3D': ReducedSetHindmarshRose,
        'Zerlaut adaptation first order': ZerlautFirstOrder,
        'Zerlaut adaptation second order': ZerlautSecondOrder,
        'Linear model': Linear,
        'Wilson-Cowan': WilsonCowan,
        'Larter-Breakspear': LarterBreakspear
    }
    return ui_name_to_model


def get_form_for_model(model_class):
    return get_model_to_form_dict().get(model_class)


class StateVariableRangesForm(Form):

    def __init__(self, prefix=''):
        super(StateVariableRangesForm, self).__init__(prefix)


class Generic2dOscillatorModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(Generic2dOscillatorModelForm, self).__init__(prefix)
        self.tau = ArrayField(Generic2dOscillator.tau, self)
        self.I = ArrayField(Generic2dOscillator.I, self)
        self.a = ArrayField(Generic2dOscillator.a, self)
        self.b = ArrayField(Generic2dOscillator.b, self)
        self.c = ArrayField(Generic2dOscillator.c, self)
        self.d = ArrayField(Generic2dOscillator.d, self)
        self.e = ArrayField(Generic2dOscillator.e, self)
        self.f = ArrayField(Generic2dOscillator.f, self)
        self.g = ArrayField(Generic2dOscillator.g, self)
        self.alpha = ArrayField(Generic2dOscillator.alpha, self)
        self.beta = ArrayField(Generic2dOscillator.beta, self)
        self.gamma = ArrayField(Generic2dOscillator.gamma, self)
        self.variables_of_interest = MultiSelectField(Generic2dOscillator.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['tau', 'a', 'b', 'c', 'I', 'd', 'e', 'f', 'g', 'alpha', 'beta', 'gamma']


class KuramotoModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(KuramotoModelForm, self).__init__(prefix)
        self.omega = ArrayField(Kuramoto.omega, self)
        self.variables_of_interest = MultiSelectField(Kuramoto.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['omega']


class SupHopfModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(SupHopfModelForm, self).__init__(prefix)
        self.a = ArrayField(SupHopf.a, self)
        self.omega = ArrayField(SupHopf.omega, self)
        self.variables_of_interest = MultiSelectField(SupHopf.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['a', 'omega']


class HopfieldModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(HopfieldModelForm, self).__init__(prefix)
        self.taux = ArrayField(Hopfield.taux, self)
        self.tauT = ArrayField(Hopfield.tauT, self)
        self.dynamic = ArrayField(Hopfield.dynamic, self)
        self.variables_of_interest = MultiSelectField(Hopfield.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['taux', 'tauT', 'dynamic']


class EpileptorModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(EpileptorModelForm, self).__init__(prefix)
        self.a = ArrayField(Epileptor.a, self)
        self.b = ArrayField(Epileptor.b, self)
        self.c = ArrayField(Epileptor.c, self)
        self.d = ArrayField(Epileptor.d, self)
        self.r = ArrayField(Epileptor.r, self)
        self.s = ArrayField(Epileptor.s, self)
        self.x0 = ArrayField(Epileptor.x0, self)
        self.Iext = ArrayField(Epileptor.Iext, self)
        self.slope = ArrayField(Epileptor.slope, self)
        self.Iext2 = ArrayField(Epileptor.Iext2, self)
        self.tau = ArrayField(Epileptor.tau, self)
        self.aa = ArrayField(Epileptor.aa, self)
        self.bb = ArrayField(Epileptor.bb, self)
        self.Kvf = ArrayField(Epileptor.Kvf, self)
        self.Kf = ArrayField(Epileptor.Kf, self)
        self.Ks = ArrayField(Epileptor.Ks, self)
        self.tt = ArrayField(Epileptor.tt, self)
        self.modification = ArrayField(Epileptor.modification, self)
        self.variables_of_interest = MultiSelectField(Epileptor.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ["Iext", "Iext2", "r", "x0", "slope"]


class Epileptor2DModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(Epileptor2DModelForm, self).__init__(prefix)
        self.a = ArrayField(Epileptor2D.a, self)
        self.b = ArrayField(Epileptor2D.b, self)
        self.c = ArrayField(Epileptor2D.c, self)
        self.d = ArrayField(Epileptor2D.d, self)
        self.r = ArrayField(Epileptor2D.r, self)
        self.x0 = ArrayField(Epileptor2D.x0, self)
        self.Iext = ArrayField(Epileptor2D.Iext, self)
        self.slope = ArrayField(Epileptor2D.slope, self)
        self.Kvf = ArrayField(Epileptor2D.Kvf, self)
        self.Ks = ArrayField(Epileptor2D.Ks, self)
        self.tt = ArrayField(Epileptor2D.tt, self)
        self.modification = ArrayField(Epileptor2D.modification, self)
        self.variables_of_interest = MultiSelectField(Epileptor2D.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ["r", "Iext", "x0"]


class EpileptorCodim3ModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(EpileptorCodim3ModelForm, self).__init__(prefix)
        self.mu1_start = ArrayField(EpileptorCodim3.mu1_start, self)
        self.mu2_start = ArrayField(EpileptorCodim3.mu2_start, self)
        self.nu_start = ArrayField(EpileptorCodim3.nu_start, self)
        self.mu1_stop = ArrayField(EpileptorCodim3.mu1_stop, self)
        self.mu2_stop = ArrayField(EpileptorCodim3.mu2_stop, self)
        self.nu_stop = ArrayField(EpileptorCodim3.nu_stop, self)
        self.b = ArrayField(EpileptorCodim3.b, self)
        self.R = ArrayField(EpileptorCodim3.R, self)
        self.c = ArrayField(EpileptorCodim3.c, self)
        self.dstar = ArrayField(EpileptorCodim3.dstar, self)
        self.Ks = ArrayField(EpileptorCodim3.Ks, self)
        self.N = ArrayField(EpileptorCodim3.N, self)
        self.modification = ArrayField(EpileptorCodim3.modification, self)
        self.variables_of_interest = MultiSelectField(EpileptorCodim3.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['mu1_start', 'mu2_start', 'nu_start', 'mu1_stop', 'mu2_stop', 'nu_stop', 'b', 'R', 'c', 'dstar', 'N',
                'Ks']


class EpileptorCodim3SlowModModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(EpileptorCodim3SlowModModelForm, self).__init__(prefix)
        self.mu1_Ain = ArrayField(EpileptorCodim3SlowMod.mu1_Ain, self)
        self.mu2_Ain = ArrayField(EpileptorCodim3SlowMod.mu2_Ain, self)
        self.nu_Ain = ArrayField(EpileptorCodim3SlowMod.nu_Ain, self)
        self.mu1_Bin = ArrayField(EpileptorCodim3SlowMod.mu1_Bin, self)
        self.mu2_Bin = ArrayField(EpileptorCodim3SlowMod.mu2_Bin, self)
        self.nu_Bin = ArrayField(EpileptorCodim3SlowMod.nu_Bin, self)
        self.mu1_Aend = ArrayField(EpileptorCodim3SlowMod.mu1_Aend, self)
        self.mu2_Aend = ArrayField(EpileptorCodim3SlowMod.mu2_Aend, self)
        self.nu_Aend = ArrayField(EpileptorCodim3SlowMod.nu_Aend, self)
        self.mu1_Bend = ArrayField(EpileptorCodim3SlowMod.mu1_Bend, self)
        self.mu2_Bend = ArrayField(EpileptorCodim3SlowMod.mu2_Bend, self)
        self.nu_Bend = ArrayField(EpileptorCodim3SlowMod.nu_Bend, self)
        self.b = ArrayField(EpileptorCodim3SlowMod.b, self)
        self.R = ArrayField(EpileptorCodim3SlowMod.R, self)
        self.c = ArrayField(EpileptorCodim3SlowMod.c, self)
        self.cA = ArrayField(EpileptorCodim3SlowMod.cA, self)
        self.cB = ArrayField(EpileptorCodim3SlowMod.cB, self)
        self.dstar = ArrayField(EpileptorCodim3SlowMod.dstar, self)
        self.Ks = ArrayField(EpileptorCodim3SlowMod.Ks, self)
        self.N = ArrayField(EpileptorCodim3SlowMod.N, self)
        self.modification = ArrayField(EpileptorCodim3SlowMod.modification, self)
        self.variables_of_interest = ArrayField(EpileptorCodim3SlowMod.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['mu1_Ain', 'mu2_Ain', 'nu_Ain', 'mu1_Bin', 'mu2_Bin', 'nu_Bin', 'mu1_Aend', 'mu2_Aend', 'nu_Aend',
                'mu1_Bend', 'mu2_Bend', 'nu_Bend', 'b', 'R', 'c', 'dstar', 'N']


class EpileptorRestingStateModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(EpileptorRestingStateModelForm, self).__init__(prefix)
        self.a = ArrayField(EpileptorRestingState.a, self)
        self.b = ArrayField(EpileptorRestingState.b, self)
        self.c = ArrayField(EpileptorRestingState.c, self)
        self.d = ArrayField(EpileptorRestingState.d, self)
        self.r = ArrayField(EpileptorRestingState.r, self)
        self.s = ArrayField(EpileptorRestingState.s, self)
        self.x0 = ArrayField(EpileptorRestingState.x0, self)
        self.Iext = ArrayField(EpileptorRestingState.Iext, self)
        self.slope = ArrayField(EpileptorRestingState.slope, self)
        self.Iext2 = ArrayField(EpileptorRestingState.Iext2, self)
        self.tau = ArrayField(EpileptorRestingState.tau, self)
        self.aa = ArrayField(EpileptorRestingState.aa, self)
        self.bb = ArrayField(EpileptorRestingState.bb, self)
        self.Kvf = ArrayField(EpileptorRestingState.Kvf, self)
        self.Kf = ArrayField(EpileptorRestingState.Kf, self)
        self.Ks = ArrayField(EpileptorRestingState.Ks, self)
        self.tt = ArrayField(EpileptorRestingState.tt, self)
        self.tau_rs = ArrayField(EpileptorRestingState.tau_rs, self)
        self.I_rs = ArrayField(EpileptorRestingState.I_rs, self)
        self.a_rs = ArrayField(EpileptorRestingState.a_rs, self)
        self.b_rs = ArrayField(EpileptorRestingState.b_rs, self)
        self.d_rs = ArrayField(EpileptorRestingState.d_rs, self)
        self.e_rs = ArrayField(EpileptorRestingState.e_rs, self)
        self.f_rs = ArrayField(EpileptorRestingState.f_rs, self)
        self.alpha_rs = ArrayField(EpileptorRestingState.alpha_rs, self)
        self.beta_rs = ArrayField(EpileptorRestingState.beta_rs, self)
        self.K_rs = ArrayField(EpileptorRestingState.K_rs, self)
        self.p = ArrayField(EpileptorRestingState.p, self)
        self.variables_of_interest = MultiSelectField(EpileptorRestingState.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['Iext', 'Iext2', 'r', 'x0', 'slope', 'tau_rs', 'a_rs', 'b_rs', 'I_rs', 'd_rs', 'e_rs', 'f_rs',
                'alpha_rs', 'beta_rs', 'gamma_rs']


class JansenRitModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(JansenRitModelForm, self).__init__(prefix)
        self.A = ArrayField(JansenRit.A, self)
        self.B = ArrayField(JansenRit.B, self)
        self.a = ArrayField(JansenRit.a, self)
        self.b = ArrayField(JansenRit.b, self)
        self.v0 = ArrayField(JansenRit.v0, self)
        self.nu_max = ArrayField(JansenRit.nu_max, self)
        self.r = ArrayField(JansenRit.r, self)
        self.J = ArrayField(JansenRit.J, self)
        self.a_1 = ArrayField(JansenRit.a_1, self)
        self.a_2 = ArrayField(JansenRit.a_2, self)
        self.a_3 = ArrayField(JansenRit.a_3, self)
        self.a_4 = ArrayField(JansenRit.a_4, self)
        self.p_min = ArrayField(JansenRit.p_min, self)
        self.p_max = ArrayField(JansenRit.p_max, self)
        self.mu = ArrayField(JansenRit.mu, self)
        self.variables_of_interest = ArrayField(JansenRit.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['A', 'B', 'a', 'b', 'v0', 'nu_max', 'r', 'J', 'a_1', 'a_2', 'a_3', 'a_4', 'p_min', 'p_max', 'mu']


class ZetterbergJansenModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(ZetterbergJansenModelForm, self).__init__(prefix)
        self.He = ArrayField(ZetterbergJansen.He, self)
        self.Hi = ArrayField(ZetterbergJansen.Hi, self)
        self.ke = ArrayField(ZetterbergJansen.ke, self)
        self.ki = ArrayField(ZetterbergJansen.ki, self)
        self.e0 = ArrayField(ZetterbergJansen.e0, self)
        self.rho_2 = ArrayField(ZetterbergJansen.rho_2, self)
        self.rho_1 = ArrayField(ZetterbergJansen.rho_1, self)
        self.gamma_1 = ArrayField(ZetterbergJansen.gamma_1, self)
        self.gamma_2 = ArrayField(ZetterbergJansen.gamma_2, self)
        self.gamma_3 = ArrayField(ZetterbergJansen.gamma_3, self)
        self.gamma_4 = ArrayField(ZetterbergJansen.gamma_4, self)
        self.gamma_5 = ArrayField(ZetterbergJansen.gamma_5, self)
        self.gamma_1T = ArrayField(ZetterbergJansen.gamma_1T, self)
        self.gamma_2T = ArrayField(ZetterbergJansen.gamma_2T, self)
        self.gamma_3T = ArrayField(ZetterbergJansen.gamma_3T, self)
        self.P = ArrayField(ZetterbergJansen.P, self)
        self.U = ArrayField(ZetterbergJansen.U, self)
        self.Q = ArrayField(ZetterbergJansen.Q, self)
        self.variables_of_interest = MultiSelectField(ZetterbergJansen.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['He', 'Hi', 'ke', 'ki', 'e0', 'rho_2', 'rho_1', 'gamma_1', 'gamma_2', 'gamma_3', 'gamma_4', 'gamma_5',
                'P', 'U', 'Q']


class ReducedWongWangModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(ReducedWongWangModelForm, self).__init__(prefix)
        self.a = ArrayField(ReducedWongWang.a, self)
        self.b = ArrayField(ReducedWongWang.b, self)
        self.d = ArrayField(ReducedWongWang.d, self)
        self.gamma = ArrayField(ReducedWongWang.gamma, self)
        self.tau_s = ArrayField(ReducedWongWang.tau_s, self)
        self.w = ArrayField(ReducedWongWang.w, self)
        self.J_N = ArrayField(ReducedWongWang.J_N, self)
        self.I_o = ArrayField(ReducedWongWang.I_o, self)
        self.sigma_noise = ArrayField(ReducedWongWang.sigma_noise, self)
        self.variables_of_interest = MultiSelectField(ReducedWongWang.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['a', 'b', 'd', 'gamma', 'tau_s', 'w', 'J_N', 'I_o']


class ReducedWongWangExcInhModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(ReducedWongWangExcInhModelForm, self).__init__(prefix)
        self.a_e = ArrayField(ReducedWongWangExcInh.a_e, self)
        self.b_e = ArrayField(ReducedWongWangExcInh.b_e, self)
        self.d_e = ArrayField(ReducedWongWangExcInh.d_e, self)
        self.gamma_e = ArrayField(ReducedWongWangExcInh.gamma_e, self)
        self.tau_e = ArrayField(ReducedWongWangExcInh.tau_e, self)
        self.w_p = ArrayField(ReducedWongWangExcInh.w_p, self)
        self.J_N = ArrayField(ReducedWongWangExcInh.J_N, self)
        self.W_e = ArrayField(ReducedWongWangExcInh.W_e, self)
        self.a_i = ArrayField(ReducedWongWangExcInh.a_i, self)
        self.b_i = ArrayField(ReducedWongWangExcInh.b_i, self)
        self.d_i = ArrayField(ReducedWongWangExcInh.d_i, self)
        self.gamma_i = ArrayField(ReducedWongWangExcInh.gamma_i, self)
        self.tau_i = ArrayField(ReducedWongWangExcInh.tau_i, self)
        self.J_i = ArrayField(ReducedWongWangExcInh.J_i, self)
        self.W_i = ArrayField(ReducedWongWangExcInh.W_i, self)
        self.I_o = ArrayField(ReducedWongWangExcInh.I_o, self)
        self.G = ArrayField(ReducedWongWangExcInh.G, self)
        self.lamda = ArrayField(ReducedWongWangExcInh.lamda, self)
        self.variables_of_interest = MultiSelectField(ReducedWongWangExcInh.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['a_e', 'b_e', 'd_e', 'gamma_e', 'tau_e', 'W_e', 'w_p', 'J_N', 'a_i', 'b_i', 'd_i', 'gamma_i', 'tau_i',
                'W_i', 'J_i', 'I_o', 'G', 'lamda']


class ReducedSetFitzHughNagumoModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(ReducedSetFitzHughNagumoModelForm, self).__init__(prefix)
        self.tau = ArrayField(ReducedSetFitzHughNagumo.tau, self)
        self.a = ArrayField(ReducedSetFitzHughNagumo.a, self)
        self.b = ArrayField(ReducedSetFitzHughNagumo.b, self)
        self.K11 = ArrayField(ReducedSetFitzHughNagumo.K11, self)
        self.K12 = ArrayField(ReducedSetFitzHughNagumo.K12, self)
        self.K21 = ArrayField(ReducedSetFitzHughNagumo.K21, self)
        self.sigma = ArrayField(ReducedSetFitzHughNagumo.sigma, self)
        self.mu = ArrayField(ReducedSetFitzHughNagumo.mu, self)
        self.variables_of_interest = MultiSelectField(ReducedSetFitzHughNagumo.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['tau', 'a', 'b', 'K11', 'K12', 'K21', 'sigma', 'mu']


class ReducedSetHindmarshRoseModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(ReducedSetHindmarshRoseModelForm, self).__init__(prefix)
        self.r = ArrayField(ReducedSetHindmarshRose.r, self)
        self.a = ArrayField(ReducedSetHindmarshRose.a, self)
        self.b = ArrayField(ReducedSetHindmarshRose.b, self)
        self.c = ArrayField(ReducedSetHindmarshRose.c, self)
        self.d = ArrayField(ReducedSetHindmarshRose.d, self)
        self.s = ArrayField(ReducedSetHindmarshRose.s, self)
        self.xo = ArrayField(ReducedSetHindmarshRose.xo, self)
        self.K11 = ArrayField(ReducedSetHindmarshRose.K11, self)
        self.K12 = ArrayField(ReducedSetHindmarshRose.K12, self)
        self.K21 = ArrayField(ReducedSetHindmarshRose.K21, self)
        self.sigma = ArrayField(ReducedSetHindmarshRose.sigma, self)
        self.mu = ArrayField(ReducedSetHindmarshRose.mu, self)
        self.variables_of_interest = MultiSelectField(ReducedSetHindmarshRose.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['r', 'a', 'b', 'c', 'd', 's', 'xo', 'K11', 'K12', 'K21', 'sigma', 'mu']


class ZerlautFirstOrderModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(ZerlautFirstOrderModelForm, self).__init__(prefix)
        self.g_L = ArrayField(ZerlautFirstOrder.g_L, self)
        self.E_L_e = ArrayField(ZerlautFirstOrder.E_L_e, self)
        self.E_L_i = ArrayField(ZerlautFirstOrder.E_L_i, self)
        self.C_m = ArrayField(ZerlautFirstOrder.C_m, self)
        self.b = ArrayField(ZerlautFirstOrder.b, self)
        self.tau_w = ArrayField(ZerlautFirstOrder.tau_w, self)
        self.E_e = ArrayField(ZerlautFirstOrder.E_e, self)
        self.E_i = ArrayField(ZerlautFirstOrder.E_i, self)
        self.Q_e = ArrayField(ZerlautFirstOrder.Q_e, self)
        self.Q_i = ArrayField(ZerlautFirstOrder.Q_i, self)
        self.tau_e = ArrayField(ZerlautFirstOrder.tau_e, self)
        self.tau_i = ArrayField(ZerlautFirstOrder.tau_i, self)
        self.N_tot = ArrayField(ZerlautFirstOrder.N_tot, self)
        self.p_connect = ArrayField(ZerlautFirstOrder.p_connect, self)
        self.g = ArrayField(ZerlautFirstOrder.g, self)
        self.T = ArrayField(ZerlautFirstOrder.T, self)
        self.P_e = ArrayField(ZerlautFirstOrder.P_e, self)
        self.P_i = ArrayField(ZerlautFirstOrder.P_i, self)
        self.external_input = ArrayField(ZerlautFirstOrder.external_input, self)
        self.variables_of_interest = MultiSelectField(ZerlautFirstOrder.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['g_L', 'E_L_e', 'E_L_i', 'C_m', 'b', 'tau_w', 'E_e', 'E_i', 'Q_e', 'Q_i', 'tau_e', 'tau_i', 'N_tot',
                'p_connect', 'g', 'T', 'external_input']


class ZerlautSecondOrderModelForm(ZerlautFirstOrderModelForm):

    def __init__(self, prefix=''):
        super(ZerlautSecondOrderModelForm, self).__init__(prefix)
        self.variables_of_interest = MultiSelectField(ZerlautSecondOrder.variables_of_interest, self)


class LinearModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(LinearModelForm, self).__init__(prefix)
        self.gamma = ArrayField(Linear.gamma, self)
        self.variables_of_interest = MultiSelectField(Linear.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['gamma']


class WilsonCowanModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(WilsonCowanModelForm, self).__init__(prefix)
        self.c_ee = ArrayField(WilsonCowan.c_ee, self)
        self.c_ie = ArrayField(WilsonCowan.c_ie, self)
        self.c_ei = ArrayField(WilsonCowan.c_ei, self)
        self.c_ii = ArrayField(WilsonCowan.c_ii, self)
        self.tau_e = ArrayField(WilsonCowan.tau_e, self)
        self.tau_i = ArrayField(WilsonCowan.tau_i, self)
        self.a_e = ArrayField(WilsonCowan.a_e, self)
        self.b_e = ArrayField(WilsonCowan.b_e, self)
        self.c_e = ArrayField(WilsonCowan.c_e, self)
        self.theta_e = ArrayField(WilsonCowan.theta_e, self)
        self.a_i = ArrayField(WilsonCowan.a_i, self)
        self.b_i = ArrayField(WilsonCowan.b_i, self)
        self.theta_i = ArrayField(WilsonCowan.theta_i, self)
        self.c_i = ArrayField(WilsonCowan.c_i, self)
        self.r_e = ArrayField(WilsonCowan.r_e, self)
        self.r_i = ArrayField(WilsonCowan.r_i, self)
        self.k_e = ArrayField(WilsonCowan.k_e, self)
        self.k_i = ArrayField(WilsonCowan.k_i, self)
        self.P = ArrayField(WilsonCowan.P, self)
        self.Q = ArrayField(WilsonCowan.Q, self)
        self.alpha_e = ArrayField(WilsonCowan.alpha_e, self)
        self.alpha_i = ArrayField(WilsonCowan.alpha_i, self)
        self.variables_of_interest = MultiSelectField(WilsonCowan.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['c_ee', 'c_ei', 'c_ie', 'c_ii', 'tau_e', 'tau_i', 'a_e', 'b_e', 'c_e', 'a_i', 'b_i', 'c_i', 'r_e',
                'r_i', 'k_e', 'k_i', 'P', 'Q', 'theta_e', 'theta_i', 'alpha_e', 'alpha_i']


class LarterBreakspearModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(LarterBreakspearModelForm, self).__init__(prefix)
        self.gCa = ArrayField(LarterBreakspear.gCa, self)
        self.gK = ArrayField(LarterBreakspear.gK, self)
        self.gL = ArrayField(LarterBreakspear.gL, self)
        self.phi = ArrayField(LarterBreakspear.phi, self)
        self.gNa = ArrayField(LarterBreakspear.gNa, self)
        self.TK = ArrayField(LarterBreakspear.TK, self)
        self.TCa = ArrayField(LarterBreakspear.TCa, self)
        self.TNa = ArrayField(LarterBreakspear.TNa, self)
        self.VCa = ArrayField(LarterBreakspear.VCa, self)
        self.VK = ArrayField(LarterBreakspear.VK, self)
        self.VL = ArrayField(LarterBreakspear.VL, self)
        self.VNa = ArrayField(LarterBreakspear.VNa, self)
        self.d_K = ArrayField(LarterBreakspear.d_K, self)
        self.tau_K = ArrayField(LarterBreakspear.tau_K, self)
        self.d_Na = ArrayField(LarterBreakspear.d_Na, self)
        self.d_Ca = ArrayField(LarterBreakspear.d_Ca, self)
        self.aei = ArrayField(LarterBreakspear.aei, self)
        self.aie = ArrayField(LarterBreakspear.aie, self)
        self.b = ArrayField(LarterBreakspear.b, self)
        self.C = ArrayField(LarterBreakspear.C, self)
        self.ane = ArrayField(LarterBreakspear.ane, self)
        self.ani = ArrayField(LarterBreakspear.ani, self)
        self.aee = ArrayField(LarterBreakspear.aee, self)
        self.Iext = ArrayField(LarterBreakspear.Iext, self)
        self.rNMDA = ArrayField(LarterBreakspear.rNMDA, self)
        self.VT = ArrayField(LarterBreakspear.VT, self)
        self.d_V = ArrayField(LarterBreakspear.d_V, self)
        self.ZT = ArrayField(LarterBreakspear.ZT, self)
        self.d_Z = ArrayField(LarterBreakspear.d_Z, self)
        self.QV_max = ArrayField(LarterBreakspear.QV_max, self)
        self.QZ_max = ArrayField(LarterBreakspear.QZ_max, self)
        self.t_scale = ArrayField(LarterBreakspear.t_scale, self)
        self.variables_of_interest = MultiSelectField(LarterBreakspear.variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['gCa', 'gK', 'gL', 'phi', 'gNa', 'TK', 'TCa', 'TNa', 'VCa', 'VK', 'VL', 'VNa', 'd_K', 'tau_K', 'd_Na',
                'd_Ca', 'aei', 'aie', 'b', 'C', 'ane', 'ani', 'aee', 'Iext', 'rNMDA', 'VT', 'd_V', 'ZT', 'd_Z',
                'QV_max', 'QZ_max', 't_scale']
