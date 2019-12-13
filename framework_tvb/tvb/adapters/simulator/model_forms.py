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

from tvb.adapters.simulator.form_with_ranges import FormWithRanges
from tvb.core.neotraits.forms import Form, ArrayField, MultiSelectField
from tvb.simulator.models import ModelsEnum


def get_model_to_form_dict():
    model_class_to_form = {
        ModelsEnum.GENERIC_2D_OSCILLATOR.get_class(): Generic2dOscillatorModelForm,
        ModelsEnum.KURAMOTO.get_class(): KuramotoModelForm,
        ModelsEnum.SUP_HOPF.get_class(): SupHopfModelForm,
        ModelsEnum.HOPFIELD.get_class(): HopfieldModelForm,
        ModelsEnum.EPILEPTOR.get_class(): EpileptorModelForm,
        ModelsEnum.EPILEPTOR_2D.get_class(): Epileptor2DModelForm,
        ModelsEnum.EPILEPTOR_CODIM_3.get_class(): EpileptorCodim3ModelForm,
        ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class(): EpileptorCodim3SlowModModelForm,
        ModelsEnum.EPILEPTOR_RS.get_class(): EpileptorRestingStateModelForm,
        ModelsEnum.JANSEN_RIT.get_class(): JansenRitModelForm,
        ModelsEnum.ZETTERBERG_JANSEN.get_class(): ZetterbergJansenModelForm,
        ModelsEnum.REDUCED_WONG_WANG.get_class(): ReducedWongWangModelForm,
        ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class(): ReducedWongWangExcInhModelForm,
        ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class(): ReducedSetFitzHughNagumoModelForm,
        ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class(): ReducedSetHindmarshRoseModelForm,
        ModelsEnum.ZERLAUT_FIRST_ORDER.get_class(): ZerlautAdaptationFirstOrderModelForm,
        ModelsEnum.ZERLAUT_SECOND_ORDER.get_class(): ZerlautAdaptationSecondOrderModelForm,
        ModelsEnum.LINEAR.get_class(): LinearModelForm,
        ModelsEnum.WILSON_COWAN.get_class(): WilsonCowanModelForm,
        ModelsEnum.LARTER_BREAKSPEAR.get_class(): LarterBreakspearModelForm
    }

    return model_class_to_form


def get_ui_name_to_model():
    ui_name_to_model = {
        'Generic 2d Oscillator': ModelsEnum.GENERIC_2D_OSCILLATOR.get_class(),
        'Kuramoto Oscillator': ModelsEnum.KURAMOTO.get_class(),
        'supHopf': ModelsEnum.KURAMOTO.get_class(),
        'Hopfield': ModelsEnum.HOPFIELD.get_class(),
        'Epileptor': ModelsEnum.EPILEPTOR.get_class(),
        'Epileptor2D': ModelsEnum.EPILEPTOR_2D.get_class(),
        'Epileptor codim 3': ModelsEnum.EPILEPTOR_CODIM_3.get_class(),
        'Epileptor codim 3 ultra-slow modulations': ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class(),
        'EpileptorRestingState': ModelsEnum.EPILEPTOR_RS.get_class(),
        'Jansen-Rit': ModelsEnum.JANSEN_RIT.get_class(),
        'Zetterberg-Jansen': ModelsEnum.ZETTERBERG_JANSEN.get_class(),
        'Reduced Wong-Wang': ModelsEnum.REDUCED_WONG_WANG.get_class(),
        'Reduced Wong-Wang with Excitatory and Inhibitory Coupled Populations': ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class(),
        'Stefanescu-Jirsa 2D': ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class(),
        'Stefanescu-Jirsa 3D': ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class(),
        'Zerlaut adaptation first order': ModelsEnum.ZERLAUT_FIRST_ORDER.get_class(),
        'Zerlaut adaptation second order': ModelsEnum.ZERLAUT_SECOND_ORDER.get_class(),
        'Linear model': ModelsEnum.LINEAR.get_class(),
        'Wilson-Cowan': ModelsEnum.WILSON_COWAN.get_class(),
        'Larter-Breakspear': ModelsEnum.LARTER_BREAKSPEAR.get_class()
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
        self.tau = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().tau, self)
        self.I = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().I, self)
        self.a = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().a, self)
        self.b = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().b, self)
        self.c = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().c, self)
        self.d = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().d, self)
        self.e = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().e, self)
        self.f = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().f, self)
        self.g = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().g, self)
        self.alpha = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().alpha, self)
        self.beta = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().beta, self)
        self.gamma = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().gamma, self)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().variables_of_interest,
            self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['tau', 'a', 'b', 'c', 'I', 'd', 'e', 'f', 'g', 'alpha', 'beta', 'gamma']


class KuramotoModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(KuramotoModelForm, self).__init__(prefix)
        self.omega = ArrayField(ModelsEnum.KURAMOTO.get_class().omega, self)
        self.variables_of_interest = MultiSelectField(ModelsEnum.KURAMOTO.get_class().variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['omega']


class SupHopfModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(SupHopfModelForm, self).__init__(prefix)
        self.a = ArrayField(ModelsEnum.SUP_HOPF.get_class().a, self)
        self.omega = ArrayField(ModelsEnum.SUP_HOPF.get_class().omega, self)
        self.variables_of_interest = MultiSelectField(ModelsEnum.SUP_HOPF.get_class().variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['a', 'omega']


class HopfieldModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(HopfieldModelForm, self).__init__(prefix)
        self.taux = ArrayField(ModelsEnum.HOPFIELD.get_class().taux, self)
        self.tauT = ArrayField(ModelsEnum.HOPFIELD.get_class().tauT, self)
        self.dynamic = ArrayField(ModelsEnum.HOPFIELD.get_class().dynamic, self)
        self.variables_of_interest = MultiSelectField(ModelsEnum.HOPFIELD.get_class().variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['taux', 'tauT', 'dynamic']


class EpileptorModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(EpileptorModelForm, self).__init__(prefix)
        self.a = ArrayField(ModelsEnum.EPILEPTOR.get_class().a, self)
        self.b = ArrayField(ModelsEnum.EPILEPTOR.get_class().b, self)
        self.c = ArrayField(ModelsEnum.EPILEPTOR.get_class().c, self)
        self.d = ArrayField(ModelsEnum.EPILEPTOR.get_class().d, self)
        self.r = ArrayField(ModelsEnum.EPILEPTOR.get_class().r, self)
        self.s = ArrayField(ModelsEnum.EPILEPTOR.get_class().s, self)
        self.x0 = ArrayField(ModelsEnum.EPILEPTOR.get_class().x0, self)
        self.Iext = ArrayField(ModelsEnum.EPILEPTOR.get_class().Iext, self)
        self.slope = ArrayField(ModelsEnum.EPILEPTOR.get_class().slope, self)
        self.Iext2 = ArrayField(ModelsEnum.EPILEPTOR.get_class().Iext2, self)
        self.tau = ArrayField(ModelsEnum.EPILEPTOR.get_class().tau, self)
        self.aa = ArrayField(ModelsEnum.EPILEPTOR.get_class().aa, self)
        self.bb = ArrayField(ModelsEnum.EPILEPTOR.get_class().bb, self)
        self.Kvf = ArrayField(ModelsEnum.EPILEPTOR.get_class().Kvf, self)
        self.Kf = ArrayField(ModelsEnum.EPILEPTOR.get_class().Kf, self)
        self.Ks = ArrayField(ModelsEnum.EPILEPTOR.get_class().Ks, self)
        self.tt = ArrayField(ModelsEnum.EPILEPTOR.get_class().tt, self)
        self.modification = ArrayField(ModelsEnum.EPILEPTOR.get_class().modification, self)
        self.variables_of_interest = MultiSelectField(ModelsEnum.EPILEPTOR.get_class().variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ["Iext", "Iext2", "r", "x0", "slope"]


class Epileptor2DModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(Epileptor2DModelForm, self).__init__(prefix)
        self.a = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().a, self)
        self.b = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().b, self)
        self.c = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().c, self)
        self.d = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().d, self)
        self.r = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().r, self)
        self.x0 = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().x0, self)
        self.Iext = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().Iext, self)
        self.slope = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().slope, self)
        self.Kvf = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().Kvf, self)
        self.Ks = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().Ks, self)
        self.tt = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().tt, self)
        self.modification = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().modification, self)
        self.variables_of_interest = MultiSelectField(ModelsEnum.EPILEPTOR_2D.get_class().variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ["r", "Iext", "x0"]


class EpileptorCodim3ModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(EpileptorCodim3ModelForm, self).__init__(prefix)
        self.mu1_start = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().mu1_start, self)
        self.mu2_start = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().mu2_start, self)
        self.nu_start = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().nu_start, self)
        self.mu1_stop = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().mu1_stop, self)
        self.mu2_stop = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().mu2_stop, self)
        self.nu_stop = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().nu_stop, self)
        self.b = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().b, self)
        self.R = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().R, self)
        self.c = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().c, self)
        self.dstar = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().dstar, self)
        self.Ks = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().Ks, self)
        self.N = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().N, self)
        self.modification = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().modification, self)
        self.variables_of_interest = MultiSelectField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().variables_of_interest,
                                                      self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['mu1_start', 'mu2_start', 'nu_start', 'mu1_stop', 'mu2_stop', 'nu_stop', 'b', 'R', 'c', 'dstar', 'N',
                'Ks']


class EpileptorCodim3SlowModModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(EpileptorCodim3SlowModModelForm, self).__init__(prefix)
        self.mu1_Ain = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu1_Ain, self)
        self.mu2_Ain = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu2_Ain, self)
        self.nu_Ain = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().nu_Ain, self)
        self.mu1_Bin = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu1_Bin, self)
        self.mu2_Bin = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu2_Bin, self)
        self.nu_Bin = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().nu_Bin, self)
        self.mu1_Aend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu1_Aend, self)
        self.mu2_Aend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu2_Aend, self)
        self.nu_Aend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().nu_Aend, self)
        self.mu1_Bend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu1_Bend, self)
        self.mu2_Bend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu2_Bend, self)
        self.nu_Bend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().nu_Bend, self)
        self.b = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().b, self)
        self.R = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().R, self)
        self.c = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().c, self)
        self.cA = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().cA, self)
        self.cB = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().cB, self)
        self.dstar = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().dstar, self)
        self.Ks = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().Ks, self)
        self.N = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().N, self)
        self.modification = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().modification, self)
        self.variables_of_interest = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().variables_of_interest,
                                                self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['mu1_Ain', 'mu2_Ain', 'nu_Ain', 'mu1_Bin', 'mu2_Bin', 'nu_Bin', 'mu1_Aend', 'mu2_Aend', 'nu_Aend',
                'mu1_Bend', 'mu2_Bend', 'nu_Bend', 'b', 'R', 'c', 'dstar', 'N']


class EpileptorRestingStateModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(EpileptorRestingStateModelForm, self).__init__(prefix)
        self.a = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().a, self)
        self.b = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().b, self)
        self.c = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().c, self)
        self.d = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().d, self)
        self.r = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().r, self)
        self.s = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().s, self)
        self.x0 = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().x0, self)
        self.Iext = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().Iext, self)
        self.slope = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().slope, self)
        self.Iext2 = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().Iext2, self)
        self.tau = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().tau, self)
        self.aa = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().aa, self)
        self.bb = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().bb, self)
        self.Kvf = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().Kvf, self)
        self.Kf = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().Kf, self)
        self.Ks = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().Ks, self)
        self.tt = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().tt, self)
        self.tau_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().tau_rs, self)
        self.I_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().I_rs, self)
        self.a_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().a_rs, self)
        self.b_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().b_rs, self)
        self.d_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().d_rs, self)
        self.e_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().e_rs, self)
        self.f_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().f_rs, self)
        self.alpha_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().alpha_rs, self)
        self.beta_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().beta_rs, self)
        self.K_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().K_rs, self)
        self.p = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().p, self)
        self.variables_of_interest = MultiSelectField(ModelsEnum.EPILEPTOR_RS.get_class().variables_of_interest,
                                                      self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['Iext', 'Iext2', 'r', 'x0', 'slope', 'tau_rs', 'a_rs', 'b_rs', 'I_rs', 'd_rs', 'e_rs', 'f_rs',
                'alpha_rs', 'beta_rs', 'gamma_rs']


class JansenRitModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(JansenRitModelForm, self).__init__(prefix)
        self.A = ArrayField(ModelsEnum.JANSEN_RIT.get_class().A, self)
        self.B = ArrayField(ModelsEnum.JANSEN_RIT.get_class().B, self)
        self.a = ArrayField(ModelsEnum.JANSEN_RIT.get_class().a, self)
        self.b = ArrayField(ModelsEnum.JANSEN_RIT.get_class().b, self)
        self.v0 = ArrayField(ModelsEnum.JANSEN_RIT.get_class().v0, self)
        self.nu_max = ArrayField(ModelsEnum.JANSEN_RIT.get_class().nu_max, self)
        self.r = ArrayField(ModelsEnum.JANSEN_RIT.get_class().r, self)
        self.J = ArrayField(ModelsEnum.JANSEN_RIT.get_class().J, self)
        self.a_1 = ArrayField(ModelsEnum.JANSEN_RIT.get_class().a_1, self)
        self.a_2 = ArrayField(ModelsEnum.JANSEN_RIT.get_class().a_2, self)
        self.a_3 = ArrayField(ModelsEnum.JANSEN_RIT.get_class().a_3, self)
        self.a_4 = ArrayField(ModelsEnum.JANSEN_RIT.get_class().a_4, self)
        self.p_min = ArrayField(ModelsEnum.JANSEN_RIT.get_class().p_min, self)
        self.p_max = ArrayField(ModelsEnum.JANSEN_RIT.get_class().p_max, self)
        self.mu = ArrayField(ModelsEnum.JANSEN_RIT.get_class().mu, self)
        self.variables_of_interest = ArrayField(ModelsEnum.JANSEN_RIT.get_class().variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['A', 'B', 'a', 'b', 'v0', 'nu_max', 'r', 'J', 'a_1', 'a_2', 'a_3', 'a_4', 'p_min', 'p_max', 'mu']


class ZetterbergJansenModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(ZetterbergJansenModelForm, self).__init__(prefix)
        self.He = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().He, self)
        self.Hi = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().Hi, self)
        self.ke = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().ke, self)
        self.ki = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().ki, self)
        self.e0 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().e0, self)
        self.rho_2 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().rho_2, self)
        self.rho_1 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().rho_1, self)
        self.gamma_1 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_1, self)
        self.gamma_2 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_2, self)
        self.gamma_3 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_3, self)
        self.gamma_4 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_4, self)
        self.gamma_5 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_5, self)
        self.gamma_1T = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_1T, self)
        self.gamma_2T = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_2T, self)
        self.gamma_3T = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_3T, self)
        self.P = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().P, self)
        self.U = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().U, self)
        self.Q = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().Q, self)
        self.variables_of_interest = MultiSelectField(ModelsEnum.ZETTERBERG_JANSEN.get_class().variables_of_interest,
                                                      self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['He', 'Hi', 'ke', 'ki', 'e0', 'rho_2', 'rho_1', 'gamma_1', 'gamma_2', 'gamma_3', 'gamma_4', 'gamma_5',
                'P', 'U', 'Q']


class ReducedWongWangModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(ReducedWongWangModelForm, self).__init__(prefix)
        self.a = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().a, self)
        self.b = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().b, self)
        self.d = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().d, self)
        self.gamma = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().gamma, self)
        self.tau_s = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().tau_s, self)
        self.w = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().w, self)
        self.J_N = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().J_N, self)
        self.I_o = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().I_o, self)
        self.sigma_noise = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().sigma_noise, self)
        self.variables_of_interest = MultiSelectField(ModelsEnum.REDUCED_WONG_WANG.get_class().variables_of_interest,
                                                      self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['a', 'b', 'd', 'gamma', 'tau_s', 'w', 'J_N', 'I_o']


class ReducedWongWangExcInhModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(ReducedWongWangExcInhModelForm, self).__init__(prefix)
        self.a_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().a_e, self)
        self.b_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().b_e, self)
        self.d_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().d_e, self)
        self.gamma_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().gamma_e, self)
        self.tau_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().tau_e, self)
        self.w_p = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().w_p, self)
        self.J_N = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().J_N, self)
        self.W_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().W_e, self)
        self.a_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().a_i, self)
        self.b_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().b_i, self)
        self.d_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().d_i, self)
        self.gamma_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().gamma_i, self)
        self.tau_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().tau_i, self)
        self.J_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().J_i, self)
        self.W_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().W_i, self)
        self.I_o = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().I_o, self)
        self.G = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().G, self)
        self.lamda = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().lamda, self)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().variables_of_interest,
            self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['a_e', 'b_e', 'd_e', 'gamma_e', 'tau_e', 'W_e', 'w_p', 'J_N', 'a_i', 'b_i', 'd_i', 'gamma_i', 'tau_i',
                'W_i', 'J_i', 'I_o', 'G', 'lamda']


class ReducedSetFitzHughNagumoModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(ReducedSetFitzHughNagumoModelForm, self).__init__(prefix)
        self.tau = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().tau, self)
        self.a = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().a, self)
        self.b = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().b, self)
        self.K11 = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().K11, self)
        self.K12 = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().K12, self)
        self.K21 = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().K21, self)
        self.sigma = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().sigma, self)
        self.mu = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().mu, self)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['tau', 'a', 'b', 'K11', 'K12', 'K21', 'sigma', 'mu']


class ReducedSetHindmarshRoseModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(ReducedSetHindmarshRoseModelForm, self).__init__(prefix)
        self.r = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().r, self)
        self.a = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().a, self)
        self.b = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().b, self)
        self.c = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().c, self)
        self.d = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().d, self)
        self.s = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().s, self)
        self.xo = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().xo, self)
        self.K11 = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().K11, self)
        self.K12 = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().K12, self)
        self.K21 = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().K21, self)
        self.sigma = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().sigma, self)
        self.mu = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().mu, self)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['r', 'a', 'b', 'c', 'd', 's', 'xo', 'K11', 'K12', 'K21', 'sigma', 'mu']


class ZerlautAdaptationFirstOrderModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(ZerlautAdaptationFirstOrderModelForm, self).__init__(prefix)
        self.g_L = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().g_L, self)
        self.E_L_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().E_L_e, self)
        self.E_L_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().E_L_i, self)
        self.C_m = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().C_m, self)
        self.b_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().b_e, self)
        self.b_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().b_i, self)
        self.a_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().a_e, self)
        self.a_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().a_i, self)
        self.tau_w_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().tau_w_e, self)
        self.tau_w_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().tau_w_i, self)
        self.E_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().E_e, self)
        self.E_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().E_i, self)
        self.Q_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().Q_e, self)
        self.Q_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().Q_i, self)
        self.tau_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().tau_e, self)
        self.tau_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().tau_i, self)
        self.N_tot = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().N_tot, self)
        self.p_connect = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().p_connect, self)
        self.g = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().g, self)
        self.K_ext_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().K_ext_e, self)
        self.K_ext_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().K_ext_i, self)
        self.T = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().T, self)
        self.P_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().P_e, self)
        self.P_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().P_i, self)
        self.external_input_ex_ex = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().external_input_ex_ex, self)
        self.external_input_ex_in = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().external_input_ex_in, self)
        self.external_input_in_ex = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().external_input_in_ex, self)
        self.external_input_in_in = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().external_input_in_in, self)
        self.variables_of_interest = MultiSelectField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().variables_of_interest,
                                                      self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['g_L', 'E_L_e', 'E_L_i', 'C_m', 'b_e', 'b_i', 'a_e', 'a_i', 'tau_w_e', 'tau_w_i', 'E_e', 'E_i', 'Q_e',
                'Q_i', 'tau_e', 'tau_i', 'N_tot', 'p_connect', 'g', 'K_ext_e', 'K_ext_i', 'T', 'external_input_ex_ex',
                'external_input_ex_in', 'external_input_in_ex', 'external_input_in_in']


class ZerlautAdaptationSecondOrderModelForm(ZerlautAdaptationFirstOrderModelForm):

    def __init__(self, prefix=''):
        super(ZerlautAdaptationSecondOrderModelForm, self).__init__(prefix)
        self.variables_of_interest = MultiSelectField(ModelsEnum.ZERLAUT_SECOND_ORDER.get_class().variables_of_interest,
                                                      self)


class LinearModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(LinearModelForm, self).__init__(prefix)
        self.gamma = ArrayField(ModelsEnum.LINEAR.get_class().gamma, self)
        self.variables_of_interest = MultiSelectField(ModelsEnum.LINEAR.get_class().variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['gamma']


class WilsonCowanModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(WilsonCowanModelForm, self).__init__(prefix)
        self.c_ee = ArrayField(ModelsEnum.WILSON_COWAN.get_class().c_ee, self)
        self.c_ie = ArrayField(ModelsEnum.WILSON_COWAN.get_class().c_ie, self)
        self.c_ei = ArrayField(ModelsEnum.WILSON_COWAN.get_class().c_ei, self)
        self.c_ii = ArrayField(ModelsEnum.WILSON_COWAN.get_class().c_ii, self)
        self.tau_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().tau_e, self)
        self.tau_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().tau_i, self)
        self.a_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().a_e, self)
        self.b_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().b_e, self)
        self.c_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().c_e, self)
        self.theta_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().theta_e, self)
        self.a_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().a_i, self)
        self.b_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().b_i, self)
        self.theta_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().theta_i, self)
        self.c_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().c_i, self)
        self.r_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().r_e, self)
        self.r_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().r_i, self)
        self.k_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().k_e, self)
        self.k_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().k_i, self)
        self.P = ArrayField(ModelsEnum.WILSON_COWAN.get_class().P, self)
        self.Q = ArrayField(ModelsEnum.WILSON_COWAN.get_class().Q, self)
        self.alpha_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().alpha_e, self)
        self.alpha_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().alpha_i, self)
        self.variables_of_interest = MultiSelectField(ModelsEnum.WILSON_COWAN.get_class().variables_of_interest, self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['c_ee', 'c_ei', 'c_ie', 'c_ii', 'tau_e', 'tau_i', 'a_e', 'b_e', 'c_e', 'a_i', 'b_i', 'c_i', 'r_e',
                'r_i', 'k_e', 'k_i', 'P', 'Q', 'theta_e', 'theta_i', 'alpha_e', 'alpha_i']


class LarterBreakspearModelForm(FormWithRanges):

    def __init__(self, prefix=''):
        super(LarterBreakspearModelForm, self).__init__(prefix)
        self.gCa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().gCa, self)
        self.gK = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().gK, self)
        self.gL = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().gL, self)
        self.phi = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().phi, self)
        self.gNa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().gNa, self)
        self.TK = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().TK, self)
        self.TCa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().TCa, self)
        self.TNa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().TNa, self)
        self.VCa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().VCa, self)
        self.VK = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().VK, self)
        self.VL = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().VL, self)
        self.VNa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().VNa, self)
        self.d_K = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().d_K, self)
        self.tau_K = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().tau_K, self)
        self.d_Na = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().d_Na, self)
        self.d_Ca = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().d_Ca, self)
        self.aei = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().aei, self)
        self.aie = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().aie, self)
        self.b = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().b, self)
        self.C = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().C, self)
        self.ane = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().ane, self)
        self.ani = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().ani, self)
        self.aee = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().aee, self)
        self.Iext = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().Iext, self)
        self.rNMDA = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().rNMDA, self)
        self.VT = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().VT, self)
        self.d_V = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().d_V, self)
        self.ZT = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().ZT, self)
        self.d_Z = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().d_Z, self)
        self.QV_max = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().QV_max, self)
        self.QZ_max = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().QZ_max, self)
        self.t_scale = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().t_scale, self)
        self.variables_of_interest = MultiSelectField(ModelsEnum.LARTER_BREAKSPEAR.get_class().variables_of_interest,
                                                      self)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['gCa', 'gK', 'gL', 'phi', 'gNa', 'TK', 'TCa', 'TNa', 'VCa', 'VK', 'VL', 'VNa', 'd_K', 'tau_K', 'd_Na',
                'd_Ca', 'aei', 'aie', 'b', 'C', 'ane', 'ani', 'aee', 'Iext', 'rNMDA', 'VT', 'd_V', 'ZT', 'd_Z',
                'QV_max', 'QZ_max', 't_scale']
