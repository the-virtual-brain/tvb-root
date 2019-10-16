from tvb.simulator.models import Generic2dOscillator, Kuramoto, Hopfield, Epileptor, Epileptor2D, \
    EpileptorCodim3, EpileptorCodim3SlowMod, JC_Epileptor, JansenRit, ZetterbergJansen, ReducedWongWang, \
    ReducedWongWangExcIOInhI, ReducedSetFitzHughNagumo, ReducedSetHindmarshRose, Zerlaut_adaptation_first_order, \
    Zerlaut_adaptation_second_order, Linear, WilsonCowan, LarterBreakspear
from tvb.simulator.models.oscillator import supHopf

from tvb.core.neotraits._forms import Form, ArrayField, MultiSelectField


def get_model_to_form_dict():
    model_class_to_form = {
        Generic2dOscillator: Generic2dOscillatorModelForm,
        Kuramoto: KuramotoModelForm,
        supHopf: SupHopfModelForm,
        Hopfield: HopfieldModelForm,
        Epileptor: EpileptorModelForm,
        Epileptor2D: Epileptor2DModelForm,
        EpileptorCodim3: EpileptorCodim3ModelForm,
        EpileptorCodim3SlowMod: EpileptorCodim3SlowModModelForm,
        JC_Epileptor: JC_EpileptorModelForm,
        JansenRit: JansenRitModelForm,
        ZetterbergJansen: ZetterbergJansenModelForm,
        ReducedWongWang: ReducedWongWangModelForm,
        ReducedWongWangExcIOInhI: ReducedWongWangExcIOInhIModelForm,
        ReducedSetFitzHughNagumo: ReducedSetFitzHughNagumoModelForm,
        ReducedSetHindmarshRose: ReducedSetHindmarshRoseModelForm,
        Zerlaut_adaptation_first_order: Zerlaut_adaptation_first_orderModelForm,
        Zerlaut_adaptation_second_order: Zerlaut_adaptation_second_orderModelForm,
        Linear: LinearModelForm,
        WilsonCowan: WilsonCowanModelForm,
        LarterBreakspear: LarterBreakspearModelForm
    }

    return model_class_to_form


def get_ui_name_to_model():
    ui_name_to_model = {
        'Generic 2d Oscillator' : Generic2dOscillator,
        'Kuramoto Oscillator' : Kuramoto,
        'supHopf' : supHopf,
        'Hopfield' : Hopfield,
        'Epileptor' : Epileptor,
        'Epileptor2D' : Epileptor2D,
        'Epileptor codim 3' : EpileptorCodim3,
        'Epileptor codim 3 ultra-slow modulations' : EpileptorCodim3SlowMod,
        'JC_Epileptor' : JC_Epileptor,
        'Jansen-Rit' : JansenRit,
        'Zetterberg-Jansen' : ZetterbergJansen,
        'Reduced Wong-Wang' : ReducedWongWang,
        'Reduced Wong-Wang with Excitatory and Inhibitory Coupled Populations' : ReducedWongWangExcIOInhI,
        'Stefanescu-Jirsa 2D' : ReducedSetFitzHughNagumo,
        'Stefanescu-Jirsa 3D' : ReducedSetHindmarshRose,
        'Zerlaut adaptation first order' : Zerlaut_adaptation_first_order,
        'Zerlaut adaptation second order': Zerlaut_adaptation_second_order,
        'Linear model' : Linear,
        'Wilson-Cowan' : WilsonCowan,
        'Larter-Breakspear' : LarterBreakspear
    }
    return ui_name_to_model


def get_form_for_model(model_class):
    return get_model_to_form_dict().get(model_class)


class StateVariableRangesForm(Form):

    def __init__(self, prefix=''):
        super(StateVariableRangesForm, self).__init__(prefix)


class Generic2dOscillatorModelForm(Form):

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


class KuramotoModelForm(Form):

    def __init__(self, prefix=''):
        super(KuramotoModelForm, self).__init__(prefix)
        self.omega = ArrayField(Kuramoto.omega, self)
        self.variables_of_interest = MultiSelectField(Kuramoto.variables_of_interest, self)


class SupHopfModelForm(Form):

    def __init__(self, prefix=''):
        super(SupHopfModelForm, self).__init__(prefix)
        self.a = ArrayField(supHopf.a, self)
        self.omega = ArrayField(supHopf.omega, self)
        self.variables_of_interest = MultiSelectField(supHopf.variables_of_interest, self)


class HopfieldModelForm(Form):

    def __init__(self, prefix=''):
        super(HopfieldModelForm, self).__init__(prefix)
        self.taux = ArrayField(Hopfield.taux, self)
        self.tauT = ArrayField(Hopfield.tauT, self)
        self.dynamic = ArrayField(Hopfield.dynamic, self)
        self.variables_of_interest = MultiSelectField(Hopfield.variables_of_interest, self)


class EpileptorModelForm(Form):

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


class Epileptor2DModelForm(Form):

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


class EpileptorCodim3ModelForm(Form):

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


class EpileptorCodim3SlowModModelForm(Form):

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


class JC_EpileptorModelForm(Form):

    def __init__(self, prefix=''):
        super(JC_EpileptorModelForm, self).__init__(prefix)
        self.a = ArrayField(JC_Epileptor.a, self)
        self.b = ArrayField(JC_Epileptor.b, self)
        self.c = ArrayField(JC_Epileptor.c, self)
        self.d = ArrayField(JC_Epileptor.d, self)
        self.r = ArrayField(JC_Epileptor.r, self)
        self.s = ArrayField(JC_Epileptor.s, self)
        self.x0 = ArrayField(JC_Epileptor.x0, self)
        self.Iext = ArrayField(JC_Epileptor.Iext, self)
        self.slope = ArrayField(JC_Epileptor.slope, self)
        self.Iext2 = ArrayField(JC_Epileptor.Iext2, self)
        self.tau = ArrayField(JC_Epileptor.tau, self)
        self.aa = ArrayField(JC_Epileptor.aa, self)
        self.bb = ArrayField(JC_Epileptor.bb, self)
        self.Kvf = ArrayField(JC_Epileptor.Kvf, self)
        self.Kf = ArrayField(JC_Epileptor.Kf, self)
        self.Ks = ArrayField(JC_Epileptor.Ks, self)
        self.tt = ArrayField(JC_Epileptor.tt, self)
        self.tau_rs = ArrayField(JC_Epileptor.tau_rs, self)
        self.I_rs = ArrayField(JC_Epileptor.I_rs, self)
        self.a_rs = ArrayField(JC_Epileptor.a_rs, self)
        self.b_rs = ArrayField(JC_Epileptor.b_rs, self)
        self.d_rs = ArrayField(JC_Epileptor.d_rs, self)
        self.e_rs = ArrayField(JC_Epileptor.e_rs, self)
        self.f_rs = ArrayField(JC_Epileptor.f_rs, self)
        self.alpha_rs = ArrayField(JC_Epileptor.alpha_rs, self)
        self.beta_rs = ArrayField(JC_Epileptor.beta_rs, self)
        self.K_rs = ArrayField(JC_Epileptor.K_rs, self)
        self.p = ArrayField(JC_Epileptor.p, self)
        self.variables_of_interest = MultiSelectField(JC_Epileptor.variables_of_interest, self)


class JansenRitModelForm(Form):

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


class ZetterbergJansenModelForm(Form):

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


class ReducedWongWangModelForm(Form):

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


class ReducedWongWangExcIOInhIModelForm(Form):

    def __init__(self, prefix=''):
        super(ReducedWongWangExcIOInhIModelForm, self).__init__(prefix)
        self.a_e = ArrayField(ReducedWongWangExcIOInhI.a_e, self)
        self.b_e = ArrayField(ReducedWongWangExcIOInhI.b_e, self)
        self.d_e = ArrayField(ReducedWongWangExcIOInhI.d_e, self)
        self.gamma_e = ArrayField(ReducedWongWangExcIOInhI.gamma_e, self)
        self.tau_e = ArrayField(ReducedWongWangExcIOInhI.tau_e, self)
        self.w_p = ArrayField(ReducedWongWangExcIOInhI.w_p, self)
        self.J_N = ArrayField(ReducedWongWangExcIOInhI.J_N, self)
        self.W_e = ArrayField(ReducedWongWangExcIOInhI.W_e, self)
        self.a_i = ArrayField(ReducedWongWangExcIOInhI.a_i, self)
        self.b_i = ArrayField(ReducedWongWangExcIOInhI.b_i, self)
        self.d_i = ArrayField(ReducedWongWangExcIOInhI.d_i, self)
        self.gamma_i = ArrayField(ReducedWongWangExcIOInhI.gamma_i, self)
        self.tau_i = ArrayField(ReducedWongWangExcIOInhI.tau_i, self)
        self.J_i = ArrayField(ReducedWongWangExcIOInhI.J_i, self)
        self.W_i = ArrayField(ReducedWongWangExcIOInhI.W_i, self)
        self.I_o = ArrayField(ReducedWongWangExcIOInhI.I_o, self)
        self.G = ArrayField(ReducedWongWangExcIOInhI.G, self)
        self.lamda = ArrayField(ReducedWongWangExcIOInhI.lamda, self)
        self.variables_of_interest = MultiSelectField(ReducedWongWangExcIOInhI.variables_of_interest, self)


class ReducedSetFitzHughNagumoModelForm(Form):

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


class ReducedSetHindmarshRoseModelForm(Form):

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


class Zerlaut_adaptation_first_orderModelForm(Form):

    def __init__(self, prefix=''):
        super(Zerlaut_adaptation_first_orderModelForm, self).__init__(prefix)
        self.g_L = ArrayField(Zerlaut_adaptation_first_order.g_L, self)
        self.E_L_e = ArrayField(Zerlaut_adaptation_first_order.E_L_e, self)
        self.E_L_i = ArrayField(Zerlaut_adaptation_first_order.E_L_i, self)
        self.C_m = ArrayField(Zerlaut_adaptation_first_order.C_m, self)
        self.b = ArrayField(Zerlaut_adaptation_first_order.b, self)
        self.tau_w = ArrayField(Zerlaut_adaptation_first_order.tau_w, self)
        self.E_e = ArrayField(Zerlaut_adaptation_first_order.E_e, self)
        self.E_i = ArrayField(Zerlaut_adaptation_first_order.E_i, self)
        self.Q_e = ArrayField(Zerlaut_adaptation_first_order.Q_e, self)
        self.Q_i = ArrayField(Zerlaut_adaptation_first_order.Q_i, self)
        self.tau_e = ArrayField(Zerlaut_adaptation_first_order.tau_e, self)
        self.tau_i = ArrayField(Zerlaut_adaptation_first_order.tau_i, self)
        self.N_tot = ArrayField(Zerlaut_adaptation_first_order.N_tot, self)
        self.p_connect = ArrayField(Zerlaut_adaptation_first_order.p_connect, self)
        self.g = ArrayField(Zerlaut_adaptation_first_order.g, self)
        self.T = ArrayField(Zerlaut_adaptation_first_order.T, self)
        self.P_e = ArrayField(Zerlaut_adaptation_first_order.P_e, self)
        self.P_i = ArrayField(Zerlaut_adaptation_first_order.P_i, self)
        self.external_input = ArrayField(Zerlaut_adaptation_first_order.external_input, self)
        self.variables_of_interest = MultiSelectField(Zerlaut_adaptation_first_order.variables_of_interest, self)


class Zerlaut_adaptation_second_orderModelForm(Zerlaut_adaptation_first_orderModelForm):

    def __init__(self, prefix=''):
        super(Zerlaut_adaptation_second_orderModelForm, self).__init__(prefix)
        self.variables_of_interest = MultiSelectField(Zerlaut_adaptation_second_order.variables_of_interest, self)


class LinearModelForm(Form):

    def __init__(self, prefix=''):
        super(LinearModelForm, self).__init__(prefix)
        self.gamma = ArrayField(Linear.gamma, self)
        self.variables_of_interest = MultiSelectField(Linear.variables_of_interest, self)


class WilsonCowanModelForm(Form):

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


class LarterBreakspearModelForm(Form):

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
