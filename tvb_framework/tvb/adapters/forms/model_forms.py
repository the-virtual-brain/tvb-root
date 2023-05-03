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

from tvb.simulator import models
from tvb.adapters.forms.form_with_ranges import FormWithRanges
from tvb.basic.neotraits.api import TupleEnum
from tvb.core.neotraits.forms import Form, ArrayField, MultiSelectField


class ModelsEnum(TupleEnum):
    GENERIC_2D_OSCILLATOR = (models.ModelsEnum.GENERIC_2D_OSCILLATOR.get_class(), "Generic 2D Oscillator")
    KURAMOTO = (models.ModelsEnum.KURAMOTO.get_class(), "Kuramoto Oscillator")
    SUP_HOPF = (models.ModelsEnum.SUP_HOPF.get_class(), "Suphopf")
    HOPFIELD = (models.ModelsEnum.HOPFIELD.get_class(), "Hopfield")
    EPILEPTOR = (models.ModelsEnum.EPILEPTOR.get_class(), "Epileptor")
    EPILEPTOR_2D = (models.ModelsEnum.EPILEPTOR_2D.get_class(), "Epileptor2D")
    EPILEPTOR_CODIM_3 = (models.ModelsEnum.EPILEPTOR_CODIM_3.get_class(), "Epileptor Codim 3")
    EPILEPTOR_CODIM_3_SLOW = (
        models.ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class(), "Epileptor Codim 3 Ultra-Slow Modulations")
    EPILEPTOR_RS = (models.ModelsEnum.EPILEPTOR_RS.get_class(), "Epileptor Resting State")
    JANSEN_RIT = (models.ModelsEnum.JANSEN_RIT.get_class(), "Jansen-Rit")
    ZETTERBERG_JANSEN = (models.ModelsEnum.ZETTERBERG_JANSEN.get_class(), "Zetterberg-Jansen")
    REDUCED_WONG_WANG = (models.ModelsEnum.REDUCED_WONG_WANG.get_class(), "Reduced Wong-Wang")
    REDUCED_WONG_WANG_EXC_INH = (models.ModelsEnum.REDUCED_WONG_WANG_EXC_INH.get_class(),
                                  "Reduced Wong-Wang With Excitatory And Inhibitory Coupled Populations")
    REDUCED_SET_FITZ_HUGH_NAGUMO = (models.ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class(), "Stefanescu-Jirsa 2D")
    REDUCED_SET_HINDMARSH_ROSE = (models.ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class(), "Stefanescu-Jirsa 3D")
    ZERLAUT_FIRST_ORDER = (models.ModelsEnum.ZERLAUT_FIRST_ORDER.get_class(), "Zerlaut Adaptation First Order")
    ZERLAUT_SECOND_ORDER = (models.ModelsEnum.ZERLAUT_SECOND_ORDER.get_class(), "Zerlaut Adaptation Second Order")
    MONTBRIO_PAZO_ROXIN = (models.ModelsEnum.MONTBRIO_PAZO_ROXIN.get_class(), "Montbrio Pazo Roxin")
    COOMBES_BYRNE = (models.ModelsEnum.COOMBES_BYRNE.get_class(), "Coombes Byrne")
    COOMBES_BYRNE_2D = (models.ModelsEnum.COOMBES_BYRNE_2D.get_class(), "Coombes Byrne 2D")
    GAST_SCHMIDT_KNOSCHE_SD = (models.ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.get_class(), "Gast Schmidt Knosche_Sd")
    GAST_SCHMIDT_KNOSCHE_SF = (models.ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.get_class(), "Gast Schmidt Knosche_Sf")
    DUMONT_GUTKIN = (models.ModelsEnum.DUMONT_GUTKIN.get_class(), "Dumont Gutkin")
    LINEAR = (models.ModelsEnum.LINEAR.get_class(), "Linear Model")
    WILSON_COWAN = (models.ModelsEnum.WILSON_COWAN.get_class(), "Wilson-Cowan")
    LARTER_BREAKSPEAR = (models.ModelsEnum.LARTER_BREAKSPEAR.get_class(), "Larter-Breakspear")


def get_model_to_form_dict():
    model_class_to_form = {
        ModelsEnum.GENERIC_2D_OSCILLATOR.value: Generic2dOscillatorModelForm,
        ModelsEnum.KURAMOTO.value: KuramotoModelForm,
        ModelsEnum.SUP_HOPF.value: SupHopfModelForm,
        ModelsEnum.HOPFIELD.value: HopfieldModelForm,
        ModelsEnum.EPILEPTOR.value: EpileptorModelForm,
        ModelsEnum.EPILEPTOR_2D.value: Epileptor2DModelForm,
        ModelsEnum.EPILEPTOR_CODIM_3.value: EpileptorCodim3ModelForm,
        ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value: EpileptorCodim3SlowModModelForm,
        ModelsEnum.EPILEPTOR_RS.value: EpileptorRestingStateModelForm,
        ModelsEnum.JANSEN_RIT.value: JansenRitModelForm,
        ModelsEnum.ZETTERBERG_JANSEN.value: ZetterbergJansenModelForm,
        ModelsEnum.REDUCED_WONG_WANG.value: ReducedWongWangModelForm,
        ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value: ReducedWongWangExcInhModelForm,
        ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.value: ReducedSetFitzHughNagumoModelForm,
        ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.value: ReducedSetHindmarshRoseModelForm,
        ModelsEnum.ZERLAUT_FIRST_ORDER.value: ZerlautAdaptationFirstOrderModelForm,
        ModelsEnum.ZERLAUT_SECOND_ORDER.value: ZerlautAdaptationSecondOrderModelForm,
        ModelsEnum.MONTBRIO_PAZO_ROXIN.value: MontbrioPazoRoxinModelForm,
        ModelsEnum.COOMBES_BYRNE.value: CoombesByrneModelForm,
        ModelsEnum.COOMBES_BYRNE_2D.value: CoombesByrne2DModelForm,
        ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.value: GastSchmidtKnoscheSDModelForm,
        ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.value: GastSchmidtKnoscheSFModelForm,
        ModelsEnum.DUMONT_GUTKIN.value: DumontGutkinModelForm,
        ModelsEnum.LINEAR.value: LinearModelForm,
        ModelsEnum.WILSON_COWAN.value: WilsonCowanModelForm,
        ModelsEnum.LARTER_BREAKSPEAR.value: LarterBreakspearModelForm
    }

    return model_class_to_form


def get_form_for_model(model_class):
    return get_model_to_form_dict().get(model_class)


class StateVariableRangesForm(Form):

    def __init__(self):
        super(StateVariableRangesForm, self).__init__()


class ModelForm(FormWithRanges):

    def __init__(self, are_voi_disabled=False):
        super(ModelForm, self).__init__()
        self.are_voi_disabled = are_voi_disabled

    def fill_from_trait(self, trait):
        # type: (Model) -> None
        super(ModelForm, self).fill_from_trait(trait)

        if self.are_voi_disabled:
            self.variables_of_interest.disabled = True

    @staticmethod
    def get_params_configurable_in_phase_plane():
        """
        Returns the configurable parameters in phase plane.
        """
        raise NotImplementedError


class Generic2dOscillatorModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(Generic2dOscillatorModelForm, self).__init__(are_voi_disabled)
        self.tau = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.value.tau)
        self.I = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.value.I)
        self.a = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.value.a)
        self.b = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.value.b)
        self.c = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.value.c)
        self.d = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.value.d)
        self.e = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.value.e)
        self.f = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.value.f)
        self.g = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.value.g)
        self.alpha = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.value.alpha)
        self.beta = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.value.beta)
        self.gamma = ArrayField(ModelsEnum.GENERIC_2D_OSCILLATOR.value.gamma)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.GENERIC_2D_OSCILLATOR.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.tau, self.a, self.b, self.c, self.I, self.d, self.e, self.f, self.g, self.alpha, self.beta,
                self.gamma]


class KuramotoModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(KuramotoModelForm, self).__init__(are_voi_disabled)
        self.omega = ArrayField(ModelsEnum.KURAMOTO.value.omega)
        self.variables_of_interest = MultiSelectField(ModelsEnum.KURAMOTO.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.omega]


class SupHopfModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(SupHopfModelForm, self).__init__(are_voi_disabled)
        self.a = ArrayField(ModelsEnum.SUP_HOPF.value.a)
        self.omega = ArrayField(ModelsEnum.SUP_HOPF.value.omega)
        self.variables_of_interest = MultiSelectField(ModelsEnum.SUP_HOPF.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.a, self.omega]


class HopfieldModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(HopfieldModelForm, self).__init__(are_voi_disabled)
        self.taux = ArrayField(ModelsEnum.HOPFIELD.value.taux)
        self.tauT = ArrayField(ModelsEnum.HOPFIELD.value.tauT)
        self.dynamic = ArrayField(ModelsEnum.HOPFIELD.value.dynamic)
        self.variables_of_interest = MultiSelectField(ModelsEnum.HOPFIELD.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.taux, self.tauT, self.dynamic]


class EpileptorModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(EpileptorModelForm, self).__init__(are_voi_disabled)
        self.a = ArrayField(ModelsEnum.EPILEPTOR.value.a)
        self.b = ArrayField(ModelsEnum.EPILEPTOR.value.b)
        self.c = ArrayField(ModelsEnum.EPILEPTOR.value.c)
        self.d = ArrayField(ModelsEnum.EPILEPTOR.value.d)
        self.r = ArrayField(ModelsEnum.EPILEPTOR.value.r)
        self.s = ArrayField(ModelsEnum.EPILEPTOR.value.s)
        self.x0 = ArrayField(ModelsEnum.EPILEPTOR.value.x0)
        self.Iext = ArrayField(ModelsEnum.EPILEPTOR.value.Iext)
        self.slope = ArrayField(ModelsEnum.EPILEPTOR.value.slope)
        self.Iext2 = ArrayField(ModelsEnum.EPILEPTOR.value.Iext2)
        self.tau = ArrayField(ModelsEnum.EPILEPTOR.value.tau)
        self.aa = ArrayField(ModelsEnum.EPILEPTOR.value.aa)
        self.bb = ArrayField(ModelsEnum.EPILEPTOR.value.bb)
        self.Kvf = ArrayField(ModelsEnum.EPILEPTOR.value.Kvf)
        self.Kf = ArrayField(ModelsEnum.EPILEPTOR.value.Kf)
        self.Ks = ArrayField(ModelsEnum.EPILEPTOR.value.Ks)
        self.tt = ArrayField(ModelsEnum.EPILEPTOR.value.tt)
        self.modification = ArrayField(ModelsEnum.EPILEPTOR.value.modification)
        self.variables_of_interest = MultiSelectField(ModelsEnum.EPILEPTOR.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.Iext, self.Iext2, self.r, self.x0, self.slope]


class Epileptor2DModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(Epileptor2DModelForm, self).__init__(are_voi_disabled)
        self.a = ArrayField(ModelsEnum.EPILEPTOR_2D.value.a)
        self.b = ArrayField(ModelsEnum.EPILEPTOR_2D.value.b)
        self.c = ArrayField(ModelsEnum.EPILEPTOR_2D.value.c)
        self.d = ArrayField(ModelsEnum.EPILEPTOR_2D.value.d)
        self.r = ArrayField(ModelsEnum.EPILEPTOR_2D.value.r)
        self.x0 = ArrayField(ModelsEnum.EPILEPTOR_2D.value.x0)
        self.Iext = ArrayField(ModelsEnum.EPILEPTOR_2D.value.Iext)
        self.slope = ArrayField(ModelsEnum.EPILEPTOR_2D.value.slope)
        self.Kvf = ArrayField(ModelsEnum.EPILEPTOR_2D.value.Kvf)
        self.Ks = ArrayField(ModelsEnum.EPILEPTOR_2D.value.Ks)
        self.tt = ArrayField(ModelsEnum.EPILEPTOR_2D.value.tt)
        self.modification = ArrayField(ModelsEnum.EPILEPTOR_2D.value.modification)
        self.variables_of_interest = MultiSelectField(ModelsEnum.EPILEPTOR_2D.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.r, self.Iext, self.x0]


class EpileptorCodim3ModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(EpileptorCodim3ModelForm, self).__init__(are_voi_disabled)
        self.mu1_start = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.value.mu1_start)
        self.mu2_start = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.value.mu2_start)
        self.nu_start = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.value.nu_start)
        self.mu1_stop = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.value.mu1_stop)
        self.mu2_stop = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.value.mu2_stop)
        self.nu_stop = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.value.nu_stop)
        self.b = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.value.b)
        self.R = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.value.R)
        self.c = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.value.c)
        self.dstar = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.value.dstar)
        self.Ks = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.value.Ks)
        self.N = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.value.N)
        self.modification = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3.value.modification)
        self.variables_of_interest = MultiSelectField(ModelsEnum.EPILEPTOR_CODIM_3.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.mu1_start, self.mu2_start, self.nu_start, self.mu1_stop, self.mu2_stop, self.nu_stop, self.b,
                self.R, self.c, self.dstar, self.N, self.Ks]


class EpileptorCodim3SlowModModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(EpileptorCodim3SlowModModelForm, self).__init__(are_voi_disabled)
        self.mu1_Ain = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.mu1_Ain)
        self.mu2_Ain = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.mu2_Ain)
        self.nu_Ain = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.nu_Ain)
        self.mu1_Bin = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.mu1_Bin)
        self.mu2_Bin = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.mu2_Bin)
        self.nu_Bin = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.nu_Bin)
        self.mu1_Aend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.mu1_Aend)
        self.mu2_Aend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.mu2_Aend)
        self.nu_Aend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.nu_Aend)
        self.mu1_Bend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.mu1_Bend)
        self.mu2_Bend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.mu2_Bend)
        self.nu_Bend = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.nu_Bend)
        self.b = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.b)
        self.R = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.R)
        self.c = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.c)
        self.cA = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.cA)
        self.cB = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.cB)
        self.dstar = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.dstar)
        self.Ks = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.Ks)
        self.N = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.N)
        self.modification = ArrayField(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.modification)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.EPILEPTOR_CODIM_3_SLOW.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.mu1_Ain, self.mu2_Ain, self.nu_Ain, self.mu1_Bin, self.mu2_Bin, self.nu_Bin, self.mu1_Aend,
                self.mu2_Aend, self.nu_Aend, self.mu1_Bend, self.mu2_Bend, self.nu_Bend, self.b, self.R, self.c,
                self.dstar, self.N]


class EpileptorRestingStateModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(EpileptorRestingStateModelForm, self).__init__(are_voi_disabled)
        self.a = ArrayField(ModelsEnum.EPILEPTOR_RS.value.a)
        self.b = ArrayField(ModelsEnum.EPILEPTOR_RS.value.b)
        self.c = ArrayField(ModelsEnum.EPILEPTOR_RS.value.c)
        self.d = ArrayField(ModelsEnum.EPILEPTOR_RS.value.d)
        self.r = ArrayField(ModelsEnum.EPILEPTOR_RS.value.r)
        self.s = ArrayField(ModelsEnum.EPILEPTOR_RS.value.s)
        self.x0 = ArrayField(ModelsEnum.EPILEPTOR_RS.value.x0)
        self.Iext = ArrayField(ModelsEnum.EPILEPTOR_RS.value.Iext)
        self.slope = ArrayField(ModelsEnum.EPILEPTOR_RS.value.slope)
        self.Iext2 = ArrayField(ModelsEnum.EPILEPTOR_RS.value.Iext2)
        self.tau = ArrayField(ModelsEnum.EPILEPTOR_RS.value.tau)
        self.aa = ArrayField(ModelsEnum.EPILEPTOR_RS.value.aa)
        self.bb = ArrayField(ModelsEnum.EPILEPTOR_RS.value.bb)
        self.Kvf = ArrayField(ModelsEnum.EPILEPTOR_RS.value.Kvf)
        self.Kf = ArrayField(ModelsEnum.EPILEPTOR_RS.value.Kf)
        self.Ks = ArrayField(ModelsEnum.EPILEPTOR_RS.value.Ks)
        self.tt = ArrayField(ModelsEnum.EPILEPTOR_RS.value.tt)
        self.tau_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.value.tau_rs)
        self.I_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.value.I_rs)
        self.a_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.value.a_rs)
        self.b_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.value.b_rs)
        self.d_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.value.d_rs)
        self.e_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.value.e_rs)
        self.f_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.value.f_rs)
        self.alpha_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.value.alpha_rs)
        self.beta_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.value.beta_rs)
        self.K_rs = ArrayField(ModelsEnum.EPILEPTOR_RS.value.K_rs)
        self.p = ArrayField(ModelsEnum.EPILEPTOR_RS.value.p)
        self.variables_of_interest = MultiSelectField(ModelsEnum.EPILEPTOR_RS.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.Iext, self.Iext2, self.r, self.x0, self.slope, self.tau_rs, self.a_rs, self.b_rs, self.I_rs,
                self.d_rs, self.e_rs, self.f_rs, self.alpha_rs, self.beta_rs]


class JansenRitModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(JansenRitModelForm, self).__init__(are_voi_disabled)
        self.A = ArrayField(ModelsEnum.JANSEN_RIT.value.A)
        self.B = ArrayField(ModelsEnum.JANSEN_RIT.value.B)
        self.a = ArrayField(ModelsEnum.JANSEN_RIT.value.a)
        self.b = ArrayField(ModelsEnum.JANSEN_RIT.value.b)
        self.v0 = ArrayField(ModelsEnum.JANSEN_RIT.value.v0)
        self.nu_max = ArrayField(ModelsEnum.JANSEN_RIT.value.nu_max)
        self.r = ArrayField(ModelsEnum.JANSEN_RIT.value.r)
        self.J = ArrayField(ModelsEnum.JANSEN_RIT.value.J)
        self.a_1 = ArrayField(ModelsEnum.JANSEN_RIT.value.a_1)
        self.a_2 = ArrayField(ModelsEnum.JANSEN_RIT.value.a_2)
        self.a_3 = ArrayField(ModelsEnum.JANSEN_RIT.value.a_3)
        self.a_4 = ArrayField(ModelsEnum.JANSEN_RIT.value.a_4)
        self.p_min = ArrayField(ModelsEnum.JANSEN_RIT.value.p_min)
        self.p_max = ArrayField(ModelsEnum.JANSEN_RIT.value.p_max)
        self.mu = ArrayField(ModelsEnum.JANSEN_RIT.value.mu)
        self.variables_of_interest = MultiSelectField(ModelsEnum.JANSEN_RIT.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.A, self.B, self.a, self.b, self.v0, self.nu_max, self.r, self.J, self.a_1, self.a_2, self.a_3,
                self.a_4, self.p_min, self.p_max, self.mu]


class ZetterbergJansenModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(ZetterbergJansenModelForm, self).__init__(are_voi_disabled)
        self.He = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.He)
        self.Hi = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.Hi)
        self.ke = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.ke)
        self.ki = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.ki)
        self.e0 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.e0)
        self.rho_2 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.rho_2)
        self.rho_1 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.rho_1)
        self.gamma_1 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.gamma_1)
        self.gamma_2 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.gamma_2)
        self.gamma_3 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.gamma_3)
        self.gamma_4 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.gamma_4)
        self.gamma_5 = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.gamma_5)
        self.gamma_1T = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.gamma_1T)
        self.gamma_2T = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.gamma_2T)
        self.gamma_3T = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.gamma_3T)
        self.P = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.P)
        self.U = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.U)
        self.Q = ArrayField(ModelsEnum.ZETTERBERG_JANSEN.value.Q)
        self.variables_of_interest = MultiSelectField(ModelsEnum.ZETTERBERG_JANSEN.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.He, self.Hi, self.ke, self.ki, self.e0, self.rho_2, self.rho_1, self.gamma_1, self.gamma_2,
                self.gamma_3, self.gamma_4, self.gamma_5, self.P, self.U, self.Q]


class ReducedWongWangModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(ReducedWongWangModelForm, self).__init__(are_voi_disabled)
        self.a = ArrayField(ModelsEnum.REDUCED_WONG_WANG.value.a)
        self.b = ArrayField(ModelsEnum.REDUCED_WONG_WANG.value.b)
        self.d = ArrayField(ModelsEnum.REDUCED_WONG_WANG.value.d)
        self.gamma = ArrayField(ModelsEnum.REDUCED_WONG_WANG.value.gamma)
        self.tau_s = ArrayField(ModelsEnum.REDUCED_WONG_WANG.value.tau_s)
        self.w = ArrayField(ModelsEnum.REDUCED_WONG_WANG.value.w)
        self.J_N = ArrayField(ModelsEnum.REDUCED_WONG_WANG.value.J_N)
        self.I_o = ArrayField(ModelsEnum.REDUCED_WONG_WANG.value.I_o)
        self.sigma_noise = ArrayField(ModelsEnum.REDUCED_WONG_WANG.value.sigma_noise)
        self.variables_of_interest = MultiSelectField(ModelsEnum.REDUCED_WONG_WANG.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.a, self.b, self.d, self.gamma, self.tau_s, self.w, self.J_N, self.I_o]


class ReducedWongWangExcInhModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(ReducedWongWangExcInhModelForm, self).__init__(are_voi_disabled)
        self.a_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.a_e)
        self.b_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.b_e)
        self.d_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.d_e)
        self.gamma_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.gamma_e)
        self.tau_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.tau_e)
        self.w_p = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.w_p)
        self.J_N = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.J_N)
        self.W_e = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.W_e)
        self.a_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.a_i)
        self.b_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.b_i)
        self.d_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.d_i)
        self.gamma_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.gamma_i)
        self.tau_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.tau_i)
        self.J_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.J_i)
        self.W_i = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.W_i)
        self.I_o = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.I_o)
        self.G = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.G)
        self.lamda = ArrayField(ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.lamda)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.REDUCED_WONG_WANG_EXC_INH.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.a_e, self.b_e, self.d_e, self.gamma_e, self.tau_e, self.W_e, self.w_p, self.J_N, self.a_i,
                self.b_i, self.d_i, self.gamma_i, self.tau_i, self.W_i, self.J_i, self.I_o, self.G, self.lamda]


class ReducedSetFitzHughNagumoModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(ReducedSetFitzHughNagumoModelForm, self).__init__(are_voi_disabled)
        self.tau = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.value.tau)
        self.a = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.value.a)
        self.b = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.value.b)
        self.K11 = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.value.K11)
        self.K12 = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.value.K12)
        self.K21 = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.value.K21)
        self.sigma = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.value.sigma)
        self.mu = ArrayField(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.value.mu)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.tau, self.a, self.b, self.K11, self.K12, self.K21, self.sigma, self.mu]


class ReducedSetHindmarshRoseModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(ReducedSetHindmarshRoseModelForm, self).__init__(are_voi_disabled)
        self.r = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.value.r)
        self.a = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.value.a)
        self.b = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.value.b)
        self.c = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.value.c)
        self.d = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.value.d)
        self.s = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.value.s)
        self.xo = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.value.xo)
        self.K11 = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.value.K11)
        self.K12 = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.value.K12)
        self.K21 = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.value.K21)
        self.sigma = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.value.sigma)
        self.mu = ArrayField(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.value.mu)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.r, self.a, self.b, self.c, self.d, self.s, self.xo, self.K11, self.K12, self.K21, self.sigma,
                self.mu]


class ZerlautAdaptationFirstOrderModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(ZerlautAdaptationFirstOrderModelForm, self).__init__(are_voi_disabled)
        self.g_L = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.g_L)
        self.E_L_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.E_L_e)
        self.E_L_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.E_L_i)
        self.C_m = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.C_m)
        self.b_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.b_e)
        self.b_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.b_i)
        self.a_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.a_e)
        self.a_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.a_i)
        self.tau_w_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.tau_w_e)
        self.tau_w_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.tau_w_i)
        self.E_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.E_e)
        self.E_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.E_i)
        self.Q_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.Q_e)
        self.Q_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.Q_i)
        self.tau_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.tau_e)
        self.tau_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.tau_i)
        self.N_tot = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.N_tot)
        self.p_connect_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.p_connect_e)
        self.p_connect_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.p_connect_i)
        self.g = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.g)
        self.K_ext_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.K_ext_e)
        self.K_ext_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.K_ext_i)
        self.T = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.T)
        self.P_e = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.P_e)
        self.P_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.P_i)
        self.external_input_ex_ex = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.external_input_ex_ex)
        self.external_input_ex_in = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.external_input_ex_in)
        self.external_input_in_ex = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.external_input_in_ex)
        self.external_input_in_in = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.external_input_in_in)
        self.tau_OU = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.tau_OU)
        self.weight_noise = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.weight_noise)
        self.S_i = ArrayField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.S_i)
        self.variables_of_interest = MultiSelectField(ModelsEnum.ZERLAUT_FIRST_ORDER.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.g_L, self.E_L_e, self.E_L_i, self.C_m, self.b_e, self.b_i, self.a_e, self.a_i, self.tau_w_e,
                self.tau_w_i, self.E_e, self.E_i, self.Q_e, self.Q_i, self.tau_e, self.tau_i, self.N_tot,
                self.p_connect_e, self.p_connect_i, self.g, self.K_ext_e, self.K_ext_i, self.T,
                self.external_input_ex_ex, self.external_input_ex_in, self.external_input_in_ex,
                self.external_input_in_in, self.tau_OU, self.weight_noise, self.S_i]


class ZerlautAdaptationSecondOrderModelForm(ZerlautAdaptationFirstOrderModelForm):

    def __init__(self, are_voi_disabled=False):
        super(ZerlautAdaptationSecondOrderModelForm, self).__init__(are_voi_disabled)


class MontbrioPazoRoxinModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(MontbrioPazoRoxinModelForm, self).__init__(are_voi_disabled)
        self.tau = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.value.tau)
        self.I = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.value.I)
        self.Delta = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.value.Delta)
        self.J = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.value.J)
        self.eta = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.value.eta)
        self.Gamma = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.value.Gamma)
        self.cr = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.value.cr)
        self.cv = ArrayField(ModelsEnum.MONTBRIO_PAZO_ROXIN.value.cv)
        self.variables_of_interest = MultiSelectField(ModelsEnum.MONTBRIO_PAZO_ROXIN.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.tau, self.I, self.Delta, self.J, self.eta, self.Gamma, self.cr, self.cv]


class CoombesByrneModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(CoombesByrneModelForm, self).__init__(are_voi_disabled)
        self.Delta = ArrayField(ModelsEnum.COOMBES_BYRNE.value.Delta)
        self.alpha = ArrayField(ModelsEnum.COOMBES_BYRNE.value.alpha)
        self.v_syn = ArrayField(ModelsEnum.COOMBES_BYRNE.value.v_syn)
        self.k = ArrayField(ModelsEnum.COOMBES_BYRNE.value.k)
        self.eta = ArrayField(ModelsEnum.COOMBES_BYRNE.value.eta)
        self.variables_of_interest = MultiSelectField(ModelsEnum.COOMBES_BYRNE.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.Delta, self.alpha, self.v_syn, self.k, self.eta]


class CoombesByrne2DModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(CoombesByrne2DModelForm, self).__init__(are_voi_disabled)
        self.Delta = ArrayField(ModelsEnum.COOMBES_BYRNE_2D.value.Delta)
        self.v_syn = ArrayField(ModelsEnum.COOMBES_BYRNE_2D.value.v_syn)
        self.k = ArrayField(ModelsEnum.COOMBES_BYRNE_2D.value.k)
        self.eta = ArrayField(ModelsEnum.COOMBES_BYRNE_2D.value.eta)
        self.variables_of_interest = MultiSelectField(ModelsEnum.COOMBES_BYRNE_2D.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.Delta, self.v_syn, self.k, self.eta]


class GastSchmidtKnoscheSDModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(GastSchmidtKnoscheSDModelForm, self).__init__(are_voi_disabled)
        self.tau = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.value.tau)
        self.tau_A = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.value.tau_A)
        self.alpha = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.value.alpha)
        self.I = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.value.I)
        self.Delta = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.value.Delta)
        self.J = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.value.J)
        self.eta = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.value.eta)
        self.cr = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.value.cr)
        self.cv = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.value.cv)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.tau, self.tau_A, self.alpha, self.I, self.Delta, self.J, self.eta, self.cr, self.cv]


class GastSchmidtKnoscheSFModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(GastSchmidtKnoscheSFModelForm, self).__init__(are_voi_disabled)
        self.tau = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.value.tau)
        self.tau_A = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.value.tau_A)
        self.alpha = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.value.alpha)
        self.I = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.value.I)
        self.Delta = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.value.Delta)
        self.J = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.value.J)
        self.eta = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.value.eta)
        self.cr = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.value.cr)
        self.cv = ArrayField(ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.value.cv)
        self.variables_of_interest = MultiSelectField(
            ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.tau, self.tau_A, self.alpha, self.I, self.Delta, self.J, self.eta, self.cr, self.cv]


class DumontGutkinModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(DumontGutkinModelForm, self).__init__(are_voi_disabled)
        self.I_e = ArrayField(ModelsEnum.DUMONT_GUTKIN.value.I_e)
        self.Delta_e = ArrayField(ModelsEnum.DUMONT_GUTKIN.value.Delta_e)
        self.eta_e = ArrayField(ModelsEnum.DUMONT_GUTKIN.value.eta_e)
        self.tau_e = ArrayField(ModelsEnum.DUMONT_GUTKIN.value.tau_e)
        self.I_i = ArrayField(ModelsEnum.DUMONT_GUTKIN.value.I_i)
        self.Delta_i = ArrayField(ModelsEnum.DUMONT_GUTKIN.value.Delta_i)
        self.eta_i = ArrayField(ModelsEnum.DUMONT_GUTKIN.value.eta_i)
        self.tau_i = ArrayField(ModelsEnum.DUMONT_GUTKIN.value.tau_i)
        self.tau_s = ArrayField(ModelsEnum.DUMONT_GUTKIN.value.tau_s)
        self.J_ee = ArrayField(ModelsEnum.DUMONT_GUTKIN.value.J_ee)
        self.J_ei = ArrayField(ModelsEnum.DUMONT_GUTKIN.value.J_ei)
        self.J_ie = ArrayField(ModelsEnum.DUMONT_GUTKIN.value.J_ie)
        self.J_ii = ArrayField(ModelsEnum.DUMONT_GUTKIN.value.J_ii)
        self.Gamma = ArrayField(ModelsEnum.DUMONT_GUTKIN.value.Gamma)
        self.variables_of_interest = MultiSelectField(ModelsEnum.DUMONT_GUTKIN.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.I_e, self.Delta_e, self.eta_e, self.tau_e, self.I_i, self.Delta_i, self.eta_i, self.tau_i,
                self.tau_s, self.J_ee, self.J_ei, self.J_ie, self.J_ii, self.Gamma]


class LinearModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(LinearModelForm, self).__init__(are_voi_disabled)
        self.gamma = ArrayField(ModelsEnum.LINEAR.value.gamma)
        self.variables_of_interest = MultiSelectField(ModelsEnum.LINEAR.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.gamma]


class WilsonCowanModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(WilsonCowanModelForm, self).__init__(are_voi_disabled)
        self.c_ee = ArrayField(ModelsEnum.WILSON_COWAN.value.c_ee)
        self.c_ie = ArrayField(ModelsEnum.WILSON_COWAN.value.c_ie)
        self.c_ei = ArrayField(ModelsEnum.WILSON_COWAN.value.c_ei)
        self.c_ii = ArrayField(ModelsEnum.WILSON_COWAN.value.c_ii)
        self.tau_e = ArrayField(ModelsEnum.WILSON_COWAN.value.tau_e)
        self.tau_i = ArrayField(ModelsEnum.WILSON_COWAN.value.tau_i)
        self.a_e = ArrayField(ModelsEnum.WILSON_COWAN.value.a_e)
        self.b_e = ArrayField(ModelsEnum.WILSON_COWAN.value.b_e)
        self.c_e = ArrayField(ModelsEnum.WILSON_COWAN.value.c_e)
        self.theta_e = ArrayField(ModelsEnum.WILSON_COWAN.value.theta_e)
        self.a_i = ArrayField(ModelsEnum.WILSON_COWAN.value.a_i)
        self.b_i = ArrayField(ModelsEnum.WILSON_COWAN.value.b_i)
        self.theta_i = ArrayField(ModelsEnum.WILSON_COWAN.value.theta_i)
        self.c_i = ArrayField(ModelsEnum.WILSON_COWAN.value.c_i)
        self.r_e = ArrayField(ModelsEnum.WILSON_COWAN.value.r_e)
        self.r_i = ArrayField(ModelsEnum.WILSON_COWAN.value.r_i)
        self.k_e = ArrayField(ModelsEnum.WILSON_COWAN.value.k_e)
        self.k_i = ArrayField(ModelsEnum.WILSON_COWAN.value.k_i)
        self.P = ArrayField(ModelsEnum.WILSON_COWAN.value.P)
        self.Q = ArrayField(ModelsEnum.WILSON_COWAN.value.Q)
        self.alpha_e = ArrayField(ModelsEnum.WILSON_COWAN.value.alpha_e)
        self.alpha_i = ArrayField(ModelsEnum.WILSON_COWAN.value.alpha_i)
        self.variables_of_interest = MultiSelectField(ModelsEnum.WILSON_COWAN.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.c_ee, self.c_ei, self.c_ie, self.c_ii, self.tau_e, self.tau_i, self.a_e, self.b_e, self.c_e,
                self.a_i, self.b_i, self.c_i, self.r_e, self.r_i, self.k_e, self.k_i, self.P, self.Q, self.theta_e,
                self.theta_i, self.alpha_e, self.alpha_i]


class LarterBreakspearModelForm(ModelForm):

    def __init__(self, are_voi_disabled=False):
        super(LarterBreakspearModelForm, self).__init__(are_voi_disabled)
        self.gCa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.gCa)
        self.gK = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.gK)
        self.gL = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.gL)
        self.phi = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.phi)
        self.gNa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.gNa)
        self.TK = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.TK)
        self.TCa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.TCa)
        self.TNa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.TNa)
        self.VCa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.VCa)
        self.VK = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.VK)
        self.VL = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.VL)
        self.VNa = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.VNa)
        self.d_K = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.d_K)
        self.tau_K = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.tau_K)
        self.d_Na = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.d_Na)
        self.d_Ca = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.d_Ca)
        self.aei = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.aei)
        self.aie = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.aie)
        self.b = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.b)
        self.C = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.C)
        self.ane = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.ane)
        self.ani = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.ani)
        self.aee = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.aee)
        self.Iext = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.Iext)
        self.rNMDA = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.rNMDA)
        self.VT = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.VT)
        self.d_V = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.d_V)
        self.ZT = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.ZT)
        self.d_Z = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.d_Z)
        self.QV_max = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.QV_max)
        self.QZ_max = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.QZ_max)
        self.t_scale = ArrayField(ModelsEnum.LARTER_BREAKSPEAR.value.t_scale)
        self.variables_of_interest = MultiSelectField(ModelsEnum.LARTER_BREAKSPEAR.value.variables_of_interest)

    def get_params_configurable_in_phase_plane(self):
        return [self.gCa, self.gK, self.gL, self.phi, self.gNa, self.TK, self.TCa, self.TNa, self.VCa, self.VK, self.VL,
                self.VNa, self.d_K, self.tau_K, self.d_Na, self.d_Ca, self.aei, self.aie, self.b, self.C, self.ane,
                self.ani, self.aee, self.Iext, self.rNMDA, self.VT, self.d_V, self.ZT, self.d_Z, self.QV_max,
                self.QZ_max, self.t_scale]
