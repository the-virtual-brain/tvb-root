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
        ModelsEnum.MONTBRIO_PAZO_ROXIN.get_class(): MontbrioPazoRoxinModelForm,
        ModelsEnum.COOMBES_BYRNE.get_class(): CoombesByrneModelForm,
        ModelsEnum.COOMBES_BYRNE_2D.get_class(): CoombesByrne2DModelForm,
        ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.get_class(): GastSchmidtKnoscheSDModelForm,
        ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.get_class(): GastSchmidtKnoscheSFModelForm,
        ModelsEnum.DUMONT_GUTKIN.get_class(): DumontGutkinModelForm,
        ModelsEnum.LINEAR.get_class(): LinearModelForm,
        ModelsEnum.WILSON_COWAN.get_class(): WilsonCowanModelForm,
        ModelsEnum.LARTER_BREAKSPEAR.get_class(): LarterBreakspearModelForm
    }

    return model_class_to_form


def get_ui_name_to_model():
    ui_name_to_model = {
        'Generic 2d Oscillator': ModelsEnum.GENERIC_2D_OSCILLATOR.get_class(),
        'Kuramoto Oscillator': ModelsEnum.KURAMOTO.get_class(),
        'supHopf': ModelsEnum.SUP_HOPF.get_class(),
        'Hopfield': ModelsEnum.HOPFIELD.get_class(),
        'Epileptor': ModelsEnum.EPILEPTOR.get_class(),
        'Epileptor2D': ModelsEnum.EPILEPTOR_2D.get_class(),
        'Epileptor codim 3': ModelsEnum.EPILEPTOR_CODIM_3.get_class(),
        'Epileptor codim 3 ultra-slow modulations': ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class(),
        'Epileptor Resting State': ModelsEnum.EPILEPTOR_RS.get_class(),
        'Jansen-Rit': ModelsEnum.JANSEN_RIT.get_class(),
        'Zetterberg-Jansen': ModelsEnum.ZETTERBERG_JANSEN.get_class(),
        'Reduced Wong-Wang': ModelsEnum.REDUCED_WONG_WANG.get_class(),
        'Reduced Wong-Wang with Excitatory and Inhibitory Coupled Populations': ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class(),
        'Stefanescu-Jirsa 2D': ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class(),
        'Stefanescu-Jirsa 3D': ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class(),
        'Zerlaut adaptation first order': ModelsEnum.ZERLAUT_FIRST_ORDER.get_class(),
        'Zerlaut adaptation second order': ModelsEnum.ZERLAUT_SECOND_ORDER.get_class(),
        'Montbrio Pazo Roxin': ModelsEnum.MONTBRIO_PAZO_ROXIN.get_class(),
        'Coombes Byrne': ModelsEnum.COOMBES_BYRNE.get_class(),
        'Coombes Byrne 2D': ModelsEnum.COOMBES_BYRNE_2D.get_class(),
        'Gast Schmidt Knosche_SD': ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.get_class(),
        'Gast Schmidt Knosche_SF': ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.get_class(),
        'Dumont Gutkin': ModelsEnum.DUMONT_GUTKIN.get_class(),
        'Linear model': ModelsEnum.LINEAR.get_class(),
        'Wilson-Cowan': ModelsEnum.WILSON_COWAN.get_class(),
        'Larter-Breakspear': ModelsEnum.LARTER_BREAKSPEAR.get_class()
    }
    return ui_name_to_model


def get_form_for_model(model_class):
    return get_model_to_form_dict().get(model_class)


class StateVariableRangesForm(Form):

    def __init__(self):
        super(StateVariableRangesForm, self).__init__()


class Generic2dOscillatorModelForm(FormWithRanges):

    def __init__(self):
        super(Generic2dOscillatorModelForm, self).__init__()
        self.tau = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().tau)
        self.I = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().I)
        self.a = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().a)
        self.b = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().b)
        self.c = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().c)
        self.d = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().d)
        self.e = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().e)
        self.f = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().f)
        self.g = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().g)
        self.alpha = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().alpha)
        self.beta = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().beta)
        self.gamma = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().gamma)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['tau', 'a', 'b', 'c', 'I', 'd', 'e', 'f', 'g', 'alpha', 'beta', 'gamma']


class KuramotoModelForm(FormWithRanges):

    def __init__(self):
        super(KuramotoModelForm, self).__init__()
        self.omega = ArrayField(ModelsEnum.KURAMOTO.get_class().omega)
        self.variables_of_interest = MultiSelectField(ModelsEnum.KURAMOTO.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['omega']


class SupHopfModelForm(FormWithRanges):

    def __init__(self):
        super(SupHopfModelForm, self).__init__()
        self.a = ArrayField(ModelsEnum.SUP_HOPF.get_class().a)
        self.omega = ArrayField(ModelsEnum.SUP_HOPF.get_class().omega)
        self.variables_of_interest = MultiSelectField(ModelsEnum.SUP_HOPF.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['a', 'omega']


class HopfieldModelForm(FormWithRanges):

    def __init__(self):
        super(HopfieldModelForm, self).__init__()
        self.taux = ArrayField(ModelsEnum.HOPFIELD.get_class().taux)
        self.tauT = ArrayField(ModelsEnum.HOPFIELD.get_class().tauT)
        self.dynamic = ArrayField(ModelsEnum.HOPFIELD.get_class().dynamic)
        self.variables_of_interest = MultiSelectField(ModelsEnum.HOPFIELD.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['taux', 'tauT', 'dynamic']


class EpileptorModelForm(FormWithRanges):

    def __init__(self):
        super(EpileptorModelForm, self).__init__()
        self.a = ArrayField(ModelsEnum.EPILEPTOR.get_class().a)
        self.b = ArrayField(ModelsEnum.EPILEPTOR.get_class().b)
        self.c = ArrayField(ModelsEnum.EPILEPTOR.get_class().c)
        self.d = ArrayField(ModelsEnum.EPILEPTOR.get_class().d)
        self.r = ArrayField(ModelsEnum.EPILEPTOR.get_class().r)
        self.s = ArrayField(ModelsEnum.EPILEPTOR.get_class().s)
        self.x0 = ArrayField(ModelsEnum.EPILEPTOR.get_class().x0)
        self.Iext = ArrayField(ModelsEnum.EPILEPTOR.get_class().Iext)
        self.slope = ArrayField(ModelsEnum.EPILEPTOR.get_class().slope)
        self.Iext2 = ArrayField(ModelsEnum.EPILEPTOR.get_class().Iext2)
        self.tau = ArrayField(ModelsEnum.EPILEPTOR.get_class().tau)
        self.aa = ArrayField(ModelsEnum.EPILEPTOR.get_class().aa)
        self.bb = ArrayField(ModelsEnum.EPILEPTOR.get_class().bb)
        self.Kvf = ArrayField(ModelsEnum.EPILEPTOR.get_class().Kvf)
        self.Kf = ArrayField(ModelsEnum.EPILEPTOR.get_class().Kf)
        self.Ks = ArrayField(ModelsEnum.EPILEPTOR.get_class().Ks)
        self.tt = ArrayField(ModelsEnum.EPILEPTOR.get_class().tt)
        self.modification = ArrayField(ModelsEnum.EPILEPTOR.get_class().modification)
        self.variables_of_interest = MultiSelectField(ModelsEnum.EPILEPTOR.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ["Iext", "Iext2", "r", "x0", "slope"]


class Epileptor2DModelForm(FormWithRanges):

    def __init__(self):
        super(Epileptor2DModelForm, self).__init__()
        self.a = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().a)
        self.b = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().b)
        self.c = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().c)
        self.d = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().d)
        self.r = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().r)
        self.x0 = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().x0)
        self.Iext = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().Iext)
        self.slope = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().slope)
        self.Kvf = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().Kvf)
        self.Ks = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().Ks)
        self.tt = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().tt)
        self.modification = ArrayField(ModelsEnum.EPILEPTOR_2D.get_class().modification)
        self.variables_of_interest = MultiSelectField(ModelsEnum.EPILEPTOR_2D.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ["r", "Iext", "x0"]


class EpileptorCodim3ModelForm(FormWithRanges):

    def __init__(self):
        super(EpileptorCodim3ModelForm, self).__init__()
        self.mu1_start = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().mu1_start)
        self.mu2_start = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().mu2_start)
        self.nu_start = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().nu_start)
        self.mu1_stop = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().mu1_stop)
        self.mu2_stop = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().mu2_stop)
        self.nu_stop = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().nu_stop)
        self.b = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().b)
        self.R = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().R)
        self.c = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().c)
        self.dstar = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().dstar)
        self.Ks = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().Ks)
        self.N = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().N)
        self.modification = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().modification)
        self.variables_of_interest = MultiSelectField(ModelsEnum.EPILEPTOR_CODIM_3.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['mu1_start', 'mu2_start', 'nu_start', 'mu1_stop', 'mu2_stop', 'nu_stop', 'b', 'R', 'c', 'dstar', 'N',
                'Ks']


class EpileptorCodim3SlowModModelForm(FormWithRanges):

    def __init__(self):
        super(EpileptorCodim3SlowModModelForm, self).__init__()
        self.mu1_Ain = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu1_Ain)
        self.mu2_Ain = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu2_Ain)
        self.nu_Ain = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().nu_Ain)
        self.mu1_Bin = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu1_Bin)
        self.mu2_Bin = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu2_Bin)
        self.nu_Bin = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().nu_Bin)
        self.mu1_Aend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu1_Aend)
        self.mu2_Aend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu2_Aend)
        self.nu_Aend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().nu_Aend)
        self.mu1_Bend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu1_Bend)
        self.mu2_Bend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu2_Bend)
        self.nu_Bend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().nu_Bend)
        self.b = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().b)
        self.R = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().R)
        self.c = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().c)
        self.cA = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().cA)
        self.cB = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().cB)
        self.dstar = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().dstar)
        self.Ks = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().Ks)
        self.N = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().N)
        self.modification = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().modification)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['mu1_Ain', 'mu2_Ain', 'nu_Ain', 'mu1_Bin', 'mu2_Bin', 'nu_Bin', 'mu1_Aend', 'mu2_Aend', 'nu_Aend',
                'mu1_Bend', 'mu2_Bend', 'nu_Bend', 'b', 'R', 'c', 'dstar', 'N']


class EpileptorRestingStateModelForm(FormWithRanges):

    def __init__(self):
        super(EpileptorRestingStateModelForm, self).__init__()
        self.a = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().a)
        self.b = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().b)
        self.c = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().c)
        self.d = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().d)
        self.r = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().r)
        self.s = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().s)
        self.x0 = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().x0)
        self.Iext = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().Iext)
        self.slope = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().slope)
        self.Iext2 = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().Iext2)
        self.tau = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().tau)
        self.aa = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().aa)
        self.bb = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().bb)
        self.Kvf = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().Kvf)
        self.Kf = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().Kf)
        self.Ks = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().Ks)
        self.tt = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().tt)
        self.tau_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().tau_rs)
        self.I_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().I_rs)
        self.a_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().a_rs)
        self.b_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().b_rs)
        self.d_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().d_rs)
        self.e_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().e_rs)
        self.f_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().f_rs)
        self.alpha_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().alpha_rs)
        self.beta_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().beta_rs)
        self.K_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().K_rs)
        self.p = ArrayField(ModelsEnum.EPILEPTOR_RS.get_class().p)
        self.variables_of_interest = MultiSelectField(ModelsEnum.EPILEPTOR_RS.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['Iext', 'Iext2', 'r', 'x0', 'slope', 'tau_rs', 'a_rs', 'b_rs', 'I_rs', 'd_rs', 'e_rs', 'f_rs',
                'alpha_rs', 'beta_rs', 'gamma_rs']


class JansenRitModelForm(FormWithRanges):

    def __init__(self):
        super(JansenRitModelForm, self).__init__()
        self.A = ArrayField(ModelsEnum.JANSEN_RIT.get_class().A)
        self.B = ArrayField(ModelsEnum.JANSEN_RIT.get_class().B)
        self.a = ArrayField(ModelsEnum.JANSEN_RIT.get_class().a)
        self.b = ArrayField(ModelsEnum.JANSEN_RIT.get_class().b)
        self.v0 = ArrayField(ModelsEnum.JANSEN_RIT.get_class().v0)
        self.nu_max = ArrayField(ModelsEnum.JANSEN_RIT.get_class().nu_max)
        self.r = ArrayField(ModelsEnum.JANSEN_RIT.get_class().r)
        self.J = ArrayField(ModelsEnum.JANSEN_RIT.get_class().J)
        self.a_1 = ArrayField(ModelsEnum.JANSEN_RIT.get_class().a_1)
        self.a_2 = ArrayField(ModelsEnum.JANSEN_RIT.get_class().a_2)
        self.a_3 = ArrayField(ModelsEnum.JANSEN_RIT.get_class().a_3)
        self.a_4 = ArrayField(ModelsEnum.JANSEN_RIT.get_class().a_4)
        self.p_min = ArrayField(ModelsEnum.JANSEN_RIT.get_class().p_min)
        self.p_max = ArrayField(ModelsEnum.JANSEN_RIT.get_class().p_max)
        self.mu = ArrayField(ModelsEnum.JANSEN_RIT.get_class().mu)
        self.variables_of_interest = MultiSelectField(ModelsEnum.JANSEN_RIT.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['A', 'B', 'a', 'b', 'v0', 'nu_max', 'r', 'J', 'a_1', 'a_2', 'a_3', 'a_4', 'p_min', 'p_max', 'mu']


class ZetterbergJansenModelForm(FormWithRanges):

    def __init__(self):
        super(ZetterbergJansenModelForm, self).__init__()
        self.He = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().He)
        self.Hi = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().Hi)
        self.ke = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().ke)
        self.ki = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().ki)
        self.e0 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().e0)
        self.rho_2 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().rho_2)
        self.rho_1 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().rho_1)
        self.gamma_1 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_1)
        self.gamma_2 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_2)
        self.gamma_3 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_3)
        self.gamma_4 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_4)
        self.gamma_5 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_5)
        self.gamma_1T = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_1T)
        self.gamma_2T = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_2T)
        self.gamma_3T = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_3T)
        self.P = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().P)
        self.U = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().U)
        self.Q = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.get_class().Q)
        self.variables_of_interest = MultiSelectField(ModelsEnum.ZETTERBERG_JANSEN.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['He', 'Hi', 'ke', 'ki', 'e0', 'rho_2', 'rho_1', 'gamma_1', 'gamma_2', 'gamma_3', 'gamma_4', 'gamma_5',
                'P', 'U', 'Q']


class ReducedWongWangModelForm(FormWithRanges):

    def __init__(self):
        super(ReducedWongWangModelForm, self).__init__()
        self.a = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().a)
        self.b = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().b)
        self.d = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().d)
        self.gamma = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().gamma)
        self.tau_s = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().tau_s)
        self.w = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().w)
        self.J_N = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().J_N)
        self.I_o = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().I_o)
        self.sigma_noise = ArrayField(ModelsEnum.REDUCED_WONG_WANG.get_class().sigma_noise)
        self.variables_of_interest = MultiSelectField(ModelsEnum.REDUCED_WONG_WANG.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['a', 'b', 'd', 'gamma', 'tau_s', 'w', 'J_N', 'I_o']


class ReducedWongWangExcInhModelForm(FormWithRanges):

    def __init__(self):
        super(ReducedWongWangExcInhModelForm, self).__init__()
        self.a_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().a_e)
        self.b_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().b_e)
        self.d_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().d_e)
        self.gamma_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().gamma_e)
        self.tau_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().tau_e)
        self.w_p = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().w_p)
        self.J_N = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().J_N)
        self.W_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().W_e)
        self.a_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().a_i)
        self.b_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().b_i)
        self.d_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().d_i)
        self.gamma_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().gamma_i)
        self.tau_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().tau_i)
        self.J_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().J_i)
        self.W_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().W_i)
        self.I_o = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().I_o)
        self.G = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().G)
        self.lamda = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().lamda)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['a_e', 'b_e', 'd_e', 'gamma_e', 'tau_e', 'W_e', 'w_p', 'J_N', 'a_i', 'b_i', 'd_i', 'gamma_i', 'tau_i',
                'W_i', 'J_i', 'I_o', 'G', 'lamda']


class ReducedSetFitzHughNagumoModelForm(FormWithRanges):

    def __init__(self):
        super(ReducedSetFitzHughNagumoModelForm, self).__init__()
        self.tau = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().tau)
        self.a = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().a)
        self.b = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().b)
        self.K11 = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().K11)
        self.K12 = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().K12)
        self.K21 = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().K21)
        self.sigma = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().sigma)
        self.mu = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().mu)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['tau', 'a', 'b', 'K11', 'K12', 'K21', 'sigma', 'mu']


class ReducedSetHindmarshRoseModelForm(FormWithRanges):

    def __init__(self):
        super(ReducedSetHindmarshRoseModelForm, self).__init__()
        self.r = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().r)
        self.a = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().a)
        self.b = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().b)
        self.c = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().c)
        self.d = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().d)
        self.s = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().s)
        self.xo = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().xo)
        self.K11 = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().K11)
        self.K12 = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().K12)
        self.K21 = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().K21)
        self.sigma = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().sigma)
        self.mu = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().mu)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['r', 'a', 'b', 'c', 'd', 's', 'xo', 'K11', 'K12', 'K21', 'sigma', 'mu']


class ZerlautAdaptationFirstOrderModelForm(FormWithRanges):

    def __init__(self):
        super(ZerlautAdaptationFirstOrderModelForm, self).__init__()
        self.g_L = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().g_L)
        self.E_L_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().E_L_e)
        self.E_L_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().E_L_i)
        self.C_m = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().C_m)
        self.b_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().b_e)
        self.b_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().b_i)
        self.a_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().a_e)
        self.a_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().a_i)
        self.tau_w_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().tau_w_e)
        self.tau_w_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().tau_w_i)
        self.E_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().E_e)
        self.E_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().E_i)
        self.Q_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().Q_e)
        self.Q_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().Q_i)
        self.tau_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().tau_e)
        self.tau_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().tau_i)
        self.N_tot = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().N_tot)
        self.p_connect = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().p_connect)
        self.g = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().g)
        self.K_ext_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().K_ext_e)
        self.K_ext_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().K_ext_i)
        self.T = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().T)
        self.P_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().P_e)
        self.P_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().P_i)
        self.external_input_ex_ex = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().external_input_ex_ex)
        self.external_input_ex_in = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().external_input_ex_in)
        self.external_input_in_ex = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().external_input_in_ex)
        self.external_input_in_in = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().external_input_in_in)
        self.variables_of_interest = MultiSelectField(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['g_L', 'E_L_e', 'E_L_i', 'C_m', 'b_e', 'b_i', 'a_e', 'a_i', 'tau_w_e', 'tau_w_i', 'E_e', 'E_i', 'Q_e',
                'Q_i', 'tau_e', 'tau_i', 'N_tot', 'p_connect', 'g', 'K_ext_e', 'K_ext_i', 'T', 'external_input_ex_ex',
                'external_input_ex_in', 'external_input_in_ex', 'external_input_in_in']


class ZerlautAdaptationSecondOrderModelForm(ZerlautAdaptationFirstOrderModelForm):

    def __init__(self):
        super(ZerlautAdaptationSecondOrderModelForm, self).__init__()
        self.variables_of_interest = MultiSelectField(ModelsEnum.ZERLAUT_SECOND_ORDER.get_class().variables_of_interest)

class MontbrioPazoRoxinModelForm(FormWithRanges):

    def __init__(self):
        super(MontbrioPazoRoxinModelForm, self).__init__()
        self.tau = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.get_class().tau)
        self.I = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.get_class().I)
        self.Delta = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.get_class().Delta)
        self.J = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.get_class().J)
        self.eta = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.get_class().eta)
        self.Gamma = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.get_class().Gamma)
        self.cr = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.get_class().cr)
        self.cv = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.get_class().cv)
        self.variables_of_interest = MultiSelectField(ModelsEnum.MONTBRIO_PAZO_ROXIN.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['tau', 'I', 'Delta', 'J', 'eta', 'Gamma', 'cr', 'cv']

class CoombesByrneModelForm(FormWithRanges):

    def __init__(self):
        super(CoombesByrneModelForm, self).__init__()
        self.Delta = ArrayField(ModelsEnum.COOMBES_BYRNE.get_class().Delta)
        self.alpha = ArrayField(ModelsEnum.COOMBES_BYRNE.get_class().alpha)
        self.v_syn = ArrayField(ModelsEnum.COOMBES_BYRNE.get_class().v_syn)
        self.k = ArrayField(ModelsEnum.COOMBES_BYRNE.get_class().k)
        self.eta = ArrayField(ModelsEnum.COOMBES_BYRNE.get_class().eta)
        self.variables_of_interest = MultiSelectField(ModelsEnum.COOMBES_BYRNE.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['Delta', 'alpha', 'v_syn', 'k', 'eta']

class CoombesByrne2DModelForm(FormWithRanges):

    def __init__(self):
        super(CoombesByrne2DModelForm, self).__init__()
        self.Delta = ArrayField(ModelsEnum.COOMBES_BYRNE_2D.get_class().Delta)
        self.v_syn = ArrayField(ModelsEnum.COOMBES_BYRNE_2D.get_class().v_syn)
        self.k = ArrayField(ModelsEnum.COOMBES_BYRNE_2D.get_class().k)
        self.eta = ArrayField(ModelsEnum.COOMBES_BYRNE_2D.get_class().eta)
        self.variables_of_interest = MultiSelectField(ModelsEnum.COOMBES_BYRNE_2D.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['Delta', 'v_syn', 'k', 'eta']

class GastSchmidtKnoscheSDModelForm(FormWithRanges):

    def __init__(self):
        super(GastSchmidtKnoscheSDModelForm, self).__init__()
        self.tau = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.get_class().tau)
        self.tau_A = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.get_class().tau_A)
        self.alpha = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.get_class().alpha)
        self.I = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.get_class().I)
        self.Delta = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.get_class().Delta)
        self.J = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.get_class().J)
        self.eta = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.get_class().eta)
        self.cr = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.get_class().cr)
        self.cv = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.get_class().cv)
        self.variables_of_interest = MultiSelectField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['tau', 'tau_A', 'alpha', 'I', 'Delta', 'J', 'eta', 'cr', 'cv']

class GastSchmidtKnoscheSFModelForm(FormWithRanges):

    def __init__(self):
        super(GastSchmidtKnoscheSFModelForm, self).__init__()
        self.tau = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.get_class().tau)
        self.tau_A = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.get_class().tau_A)
        self.alpha = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.get_class().alpha)
        self.I = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.get_class().I)
        self.Delta = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.get_class().Delta)
        self.J = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.get_class().J)
        self.eta = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.get_class().eta)
        self.cr = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.get_class().cr)
        self.cv = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.get_class().cv)
        self.variables_of_interest = MultiSelectField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['tau', 'tau_A', 'alpha', 'I', 'Delta', 'J', 'eta', 'cr', 'cv']

class DumontGutkinModelForm(FormWithRanges):

    def __init__(self):
        super(DumontGutkinModelForm, self).__init__()
        self.I_e = ArrayField(ModelsEnum.DUMONT_GUTKIN.get_class().I_e)
        self.Delta_e = ArrayField(ModelsEnum.DUMONT_GUTKIN.get_class().Delta_e)
        self.eta_e = ArrayField(ModelsEnum.DUMONT_GUTKIN.get_class().eta_e)
        self.tau_e = ArrayField(ModelsEnum.DUMONT_GUTKIN.get_class().tau_e)
        self.I_i = ArrayField(ModelsEnum.DUMONT_GUTKIN.get_class().I_i)
        self.Delta_i = ArrayField(ModelsEnum.DUMONT_GUTKIN.get_class().Delta_i)
        self.eta_i = ArrayField(ModelsEnum.DUMONT_GUTKIN.get_class().eta_i)
        self.tau_i = ArrayField(ModelsEnum.DUMONT_GUTKIN.get_class().tau_i)
        self.tau_s = ArrayField(ModelsEnum.DUMONT_GUTKIN.get_class().tau_s)
        self.J_ee = ArrayField(ModelsEnum.DUMONT_GUTKIN.get_class().J_ee)
        self.J_ei = ArrayField(ModelsEnum.DUMONT_GUTKIN.get_class().J_ei)
        self.J_ie = ArrayField(ModelsEnum.DUMONT_GUTKIN.get_class().J_ie)
        self.J_ii = ArrayField(ModelsEnum.DUMONT_GUTKIN.get_class().J_ii)
        self.Gamma = ArrayField(ModelsEnum.DUMONT_GUTKIN.get_class().Gamma)
        self.variables_of_interest = MultiSelectField(ModelsEnum.DUMONT_GUTKIN.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['I_e', 'Delta_e', 'eta_e', 'tau_e', 'I_i', 'Delta_i', 'eta_i', 'tau_i', 'tau_s',
                'J_ee', 'J_ei', 'J_ie', 'J_ii', 'Gamma']


class LinearModelForm(FormWithRanges):

    def __init__(self):
        super(LinearModelForm, self).__init__()
        self.gamma = ArrayField(ModelsEnum.LINEAR.get_class().gamma)
        self.variables_of_interest = MultiSelectField(ModelsEnum.LINEAR.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['gamma']


class WilsonCowanModelForm(FormWithRanges):

    def __init__(self):
        super(WilsonCowanModelForm, self).__init__()
        self.c_ee = ArrayField(ModelsEnum.WILSON_COWAN.get_class().c_ee)
        self.c_ie = ArrayField(ModelsEnum.WILSON_COWAN.get_class().c_ie)
        self.c_ei = ArrayField(ModelsEnum.WILSON_COWAN.get_class().c_ei)
        self.c_ii = ArrayField(ModelsEnum.WILSON_COWAN.get_class().c_ii)
        self.tau_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().tau_e)
        self.tau_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().tau_i)
        self.a_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().a_e)
        self.b_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().b_e)
        self.c_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().c_e)
        self.theta_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().theta_e)
        self.a_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().a_i)
        self.b_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().b_i)
        self.theta_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().theta_i)
        self.c_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().c_i)
        self.r_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().r_e)
        self.r_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().r_i)
        self.k_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().k_e)
        self.k_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().k_i)
        self.P = ArrayField(ModelsEnum.WILSON_COWAN.get_class().P)
        self.Q = ArrayField(ModelsEnum.WILSON_COWAN.get_class().Q)
        self.alpha_e = ArrayField(ModelsEnum.WILSON_COWAN.get_class().alpha_e)
        self.alpha_i = ArrayField(ModelsEnum.WILSON_COWAN.get_class().alpha_i)
        self.variables_of_interest = MultiSelectField(ModelsEnum.WILSON_COWAN.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['c_ee', 'c_ei', 'c_ie', 'c_ii', 'tau_e', 'tau_i', 'a_e', 'b_e', 'c_e', 'a_i', 'b_i', 'c_i', 'r_e',
                'r_i', 'k_e', 'k_i', 'P', 'Q', 'theta_e', 'theta_i', 'alpha_e', 'alpha_i']


class LarterBreakspearModelForm(FormWithRanges):

    def __init__(self):
        super(LarterBreakspearModelForm, self).__init__()
        self.gCa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().gCa)
        self.gK = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().gK)
        self.gL = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().gL)
        self.phi = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().phi)
        self.gNa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().gNa)
        self.TK = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().TK)
        self.TCa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().TCa)
        self.TNa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().TNa)
        self.VCa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().VCa)
        self.VK = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().VK)
        self.VL = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().VL)
        self.VNa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().VNa)
        self.d_K = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().d_K)
        self.tau_K = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().tau_K)
        self.d_Na = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().d_Na)
        self.d_Ca = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().d_Ca)
        self.aei = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().aei)
        self.aie = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().aie)
        self.b = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().b)
        self.C = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().C)
        self.ane = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().ane)
        self.ani = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().ani)
        self.aee = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().aee)
        self.Iext = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().Iext)
        self.rNMDA = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().rNMDA)
        self.VT = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().VT)
        self.d_V = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().d_V)
        self.ZT = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().ZT)
        self.d_Z = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().d_Z)
        self.QV_max = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().QV_max)
        self.QZ_max = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().QZ_max)
        self.t_scale = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.get_class().t_scale)
        self.variables_of_interest = MultiSelectField(ModelsEnum.LARTER_BREAKSPEAR.get_class().variables_of_interest)

    @staticmethod
    def get_params_configurable_in_phase_plane():
        return ['gCa', 'gK', 'gL', 'phi', 'gNa', 'TK', 'TCa', 'TNa', 'VCa', 'VK', 'VL', 'VNa', 'd_K', 'tau_K', 'd_Na',
                'd_Ca', 'aei', 'aie', 'b', 'C', 'ane', 'ani', 'aee', 'Iext', 'rNMDA', 'VT', 'd_V', 'ZT', 'd_Z',
                'QV_max', 'QZ_max', 't_scale']
