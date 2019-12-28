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
import json
import numpy
from tvb.simulator.models import *
from tvb.core.entities.file.simulator.configurations_h5 import SimulatorConfigurationH5
from tvb.core.neotraits.h5 import DataSet, Json, JsonFinal


class StateVariablesEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, numpy.ndarray):
            o = o.tolist()
        return o


class StateVariablesDecoder(json.JSONDecoder):
    def __init__(self):
        json.JSONDecoder.__init__(self, object_hook=self.dict_array)

    def dict_array(self, dictionary):
        dict_array = {}
        for k, v in dictionary.items():
            dict_array.update({k: numpy.array(v)})
        return dict_array


class EpileptorH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(EpileptorH5, self).__init__(path)

        self.a = DataSet(Epileptor.a, self)
        self.b = DataSet(Epileptor.b, self)
        self.c = DataSet(Epileptor.c, self)
        self.d = DataSet(Epileptor.d, self)
        self.r = DataSet(Epileptor.r, self)
        self.s = DataSet(Epileptor.s, self)
        self.x0 = DataSet(Epileptor.x0, self)
        self.Iext = DataSet(Epileptor.Iext, self)
        self.slope = DataSet(Epileptor.slope, self)
        self.Iext2 = DataSet(Epileptor.Iext2, self)
        self.tau = DataSet(Epileptor.tau, self)
        self.aa = DataSet(Epileptor.aa, self)
        self.bb = DataSet(Epileptor.bb, self)
        self.Kvf = DataSet(Epileptor.Kvf, self)
        self.Kf = DataSet(Epileptor.Kf, self)
        self.Ks = DataSet(Epileptor.Ks, self)
        self.tt = DataSet(Epileptor.tt, self)
        self.modification = DataSet(Epileptor.modification, self)
        self.state_variable_range = JsonFinal(Epileptor.state_variable_range, self, json_encoder=StateVariablesEncoder,
                                              json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(Epileptor.variables_of_interest, self)


class Epileptor2DH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(Epileptor2DH5, self).__init__(path)

        self.a = DataSet(Epileptor2D.a, self)
        self.b = DataSet(Epileptor2D.b, self)
        self.c = DataSet(Epileptor2D.c, self)
        self.d = DataSet(Epileptor2D.d, self)
        self.r = DataSet(Epileptor2D.r, self)
        self.x0 = DataSet(Epileptor2D.x0, self)
        self.Iext = DataSet(Epileptor2D.Iext, self)
        self.slope = DataSet(Epileptor2D.slope, self)
        self.Kvf = DataSet(Epileptor2D.Kvf, self)
        self.Ks = DataSet(Epileptor2D.Ks, self)
        self.tt = DataSet(Epileptor2D.tt, self)
        self.modification = DataSet(Epileptor2D.modification, self)
        self.state_variable_range = JsonFinal(Epileptor2D.state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(Epileptor2D.variables_of_interest, self)


class EpileptorCodim3H5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(EpileptorCodim3H5, self).__init__(path)
        self.mu1_start = DataSet(EpileptorCodim3.mu1_start, self)
        self.mu2_start = DataSet(EpileptorCodim3.mu2_start, self)
        self.nu_start = DataSet(EpileptorCodim3.nu_start, self)
        self.mu1_stop = DataSet(EpileptorCodim3.mu1_stop, self)
        self.mu2_stop = DataSet(EpileptorCodim3.mu2_stop, self)
        self.nu_stop = DataSet(EpileptorCodim3.nu_stop, self)
        self.b = DataSet(EpileptorCodim3.b, self)
        self.R = DataSet(EpileptorCodim3.R, self)
        self.c = DataSet(EpileptorCodim3.c, self)
        self.dstar = DataSet(EpileptorCodim3.dstar, self)
        self.Ks = DataSet(EpileptorCodim3.Ks, self)
        self.N = DataSet(EpileptorCodim3.N, self)
        self.modification = DataSet(EpileptorCodim3.modification, self)
        self.state_variable_range = JsonFinal(EpileptorCodim3.state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(EpileptorCodim3.variables_of_interest, self)


class EpileptorCodim3SlowModH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(EpileptorCodim3SlowModH5, self).__init__(path)
        self.mu1_Ain = DataSet(EpileptorCodim3SlowMod.mu1_Ain, self)
        self.mu2_Ain = DataSet(EpileptorCodim3SlowMod.mu2_Ain, self)
        self.nu_Ain = DataSet(EpileptorCodim3SlowMod.nu_Ain, self)
        self.mu1_Bin = DataSet(EpileptorCodim3SlowMod.mu1_Bin, self)
        self.mu2_Bin = DataSet(EpileptorCodim3SlowMod.mu2_Bin, self)
        self.nu_Bin = DataSet(EpileptorCodim3SlowMod.nu_Bin, self)
        self.mu1_Aend = DataSet(EpileptorCodim3SlowMod.mu1_Aend, self)
        self.mu2_Aend = DataSet(EpileptorCodim3SlowMod.mu2_Aend, self)
        self.nu_Aend = DataSet(EpileptorCodim3SlowMod.nu_Aend, self)
        self.mu1_Bend = DataSet(EpileptorCodim3SlowMod.mu1_Bend, self)
        self.mu2_Bend = DataSet(EpileptorCodim3SlowMod.mu2_Bend, self)
        self.nu_Bend = DataSet(EpileptorCodim3SlowMod.nu_Bend, self)
        self.b = DataSet(EpileptorCodim3SlowMod.b, self)
        self.R = DataSet(EpileptorCodim3SlowMod.R, self)
        self.c = DataSet(EpileptorCodim3SlowMod.c, self)
        self.cA = DataSet(EpileptorCodim3SlowMod.cA, self)
        self.cB = DataSet(EpileptorCodim3SlowMod.cB, self)
        self.dstar = DataSet(EpileptorCodim3SlowMod.dstar, self)
        self.Ks = DataSet(EpileptorCodim3SlowMod.Ks, self)
        self.N = DataSet(EpileptorCodim3SlowMod.N, self)
        self.modification = DataSet(EpileptorCodim3SlowMod.modification, self)
        self.state_variable_range = JsonFinal(EpileptorCodim3SlowMod.state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(EpileptorCodim3SlowMod.variables_of_interest, self)


class HopfieldH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(HopfieldH5, self).__init__(path)
        self.taux = DataSet(Hopfield.taux, self)
        self.tauT = DataSet(Hopfield.tauT, self)
        self.dynamic = DataSet(Hopfield.dynamic, self)
        self.state_variable_range = JsonFinal(Hopfield.state_variable_range, self, json_encoder=StateVariablesEncoder,
                                              json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(Hopfield.variables_of_interest, self)


class JansenRitH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(JansenRitH5, self).__init__(path)
        self.A = DataSet(JansenRit.A, self)
        self.B = DataSet(JansenRit.B, self)
        self.a = DataSet(JansenRit.a, self)
        self.b = DataSet(JansenRit.b, self)
        self.v0 = DataSet(JansenRit.v0, self)
        self.nu_max = DataSet(JansenRit.nu_max, self)
        self.r = DataSet(JansenRit.r, self)
        self.J = DataSet(JansenRit.J, self)
        self.a_1 = DataSet(JansenRit.a_1, self)
        self.a_2 = DataSet(JansenRit.a_2, self)
        self.a_3 = DataSet(JansenRit.a_3, self)
        self.a_4 = DataSet(JansenRit.a_4, self)
        self.p_min = DataSet(JansenRit.p_min, self)
        self.p_max = DataSet(JansenRit.p_max, self)
        self.mu = DataSet(JansenRit.mu, self)
        self.state_variable_range = JsonFinal(JansenRit.state_variable_range, self, json_encoder=StateVariablesEncoder,
                                              json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(JansenRit.variables_of_interest, self)


class ZetterbergJansenH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(ZetterbergJansenH5, self).__init__(path)
        self.He = DataSet(ZetterbergJansen.He, self)
        self.Hi = DataSet(ZetterbergJansen.Hi, self)
        self.ke = DataSet(ZetterbergJansen.ke, self)
        self.ki = DataSet(ZetterbergJansen.ki, self)
        self.e0 = DataSet(ZetterbergJansen.e0, self)
        self.rho_2 = DataSet(ZetterbergJansen.rho_2, self)
        self.rho_1 = DataSet(ZetterbergJansen.rho_1, self)
        self.gamma_1 = DataSet(ZetterbergJansen.gamma_1, self)
        self.gamma_2 = DataSet(ZetterbergJansen.gamma_2, self)
        self.gamma_3 = DataSet(ZetterbergJansen.gamma_3, self)
        self.gamma_4 = DataSet(ZetterbergJansen.gamma_4, self)
        self.gamma_5 = DataSet(ZetterbergJansen.gamma_5, self)
        self.gamma_1T = DataSet(ZetterbergJansen.gamma_1T, self)
        self.gamma_2T = DataSet(ZetterbergJansen.gamma_2T, self)
        self.gamma_3T = DataSet(ZetterbergJansen.gamma_3T, self)
        self.P = DataSet(ZetterbergJansen.P, self)
        self.U = DataSet(ZetterbergJansen.U, self)
        self.Q = DataSet(ZetterbergJansen.Q, self)
        self.state_variable_range = JsonFinal(ZetterbergJansen.state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ZetterbergJansen.variables_of_interest, self)


class EpileptorRestingStateH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(EpileptorRestingStateH5, self).__init__(path)
        self.a = DataSet(EpileptorRestingState.a, self)
        self.b = DataSet(EpileptorRestingState.b, self)
        self.c = DataSet(EpileptorRestingState.c, self)
        self.d = DataSet(EpileptorRestingState.d, self)
        self.r = DataSet(EpileptorRestingState.r, self)
        self.s = DataSet(EpileptorRestingState.s, self)
        self.x0 = DataSet(EpileptorRestingState.x0, self)
        self.Iext = DataSet(EpileptorRestingState.Iext, self)
        self.slope = DataSet(EpileptorRestingState.slope, self)
        self.Iext2 = DataSet(EpileptorRestingState.Iext2, self)
        self.tau = DataSet(EpileptorRestingState.tau, self)
        self.aa = DataSet(EpileptorRestingState.aa, self)
        self.bb = DataSet(EpileptorRestingState.bb, self)
        self.Kvf = DataSet(EpileptorRestingState.Kvf, self)
        self.Kf = DataSet(EpileptorRestingState.Kf, self)
        self.Ks = DataSet(EpileptorRestingState.Ks, self)
        self.tt = DataSet(EpileptorRestingState.tt, self)
        self.tau_rs = DataSet(EpileptorRestingState.tau_rs, self)
        self.I_rs = DataSet(EpileptorRestingState.I_rs, self)
        self.a_rs = DataSet(EpileptorRestingState.a_rs, self)
        self.b_rs = DataSet(EpileptorRestingState.b_rs, self)
        self.d_rs = DataSet(EpileptorRestingState.d_rs, self)
        self.e_rs = DataSet(EpileptorRestingState.e_rs, self)
        self.f_rs = DataSet(EpileptorRestingState.f_rs, self)
        self.alpha_rs = DataSet(EpileptorRestingState.alpha_rs, self)
        self.beta_rs = DataSet(EpileptorRestingState.beta_rs, self)
        self.gamma_rs = DataSet(EpileptorRestingState.gamma_rs, self)
        self.K_rs = DataSet(EpileptorRestingState.K_rs, self)
        self.p = DataSet(EpileptorRestingState.p, self)
        self.state_variable_range = JsonFinal(EpileptorRestingState.state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(EpileptorRestingState.variables_of_interest, self)


class LarterBreakspearH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(LarterBreakspearH5, self).__init__(path)
        self.gCa = DataSet(LarterBreakspear.gCa, self)
        self.gK = DataSet(LarterBreakspear.gK, self)
        self.gL = DataSet(LarterBreakspear.gL, self)
        self.phi = DataSet(LarterBreakspear.phi, self)
        self.gNa = DataSet(LarterBreakspear.gNa, self)
        self.TK = DataSet(LarterBreakspear.TK, self)
        self.TCa = DataSet(LarterBreakspear.TCa, self)
        self.TNa = DataSet(LarterBreakspear.TNa, self)
        self.VCa = DataSet(LarterBreakspear.VCa, self)
        self.VK = DataSet(LarterBreakspear.VK, self)
        self.VL = DataSet(LarterBreakspear.VL, self)
        self.VNa = DataSet(LarterBreakspear.VNa, self)
        self.d_K = DataSet(LarterBreakspear.d_K, self)
        self.tau_K = DataSet(LarterBreakspear.tau_K, self)
        self.d_Na = DataSet(LarterBreakspear.d_Na, self)
        self.d_Ca = DataSet(LarterBreakspear.d_Ca, self)
        self.aei = DataSet(LarterBreakspear.aei, self)
        self.aie = DataSet(LarterBreakspear.aie, self)
        self.b = DataSet(LarterBreakspear.b, self)
        self.C = DataSet(LarterBreakspear.C, self)
        self.ane = DataSet(LarterBreakspear.ane, self)
        self.ani = DataSet(LarterBreakspear.ani, self)
        self.aee = DataSet(LarterBreakspear.aee, self)
        self.Iext = DataSet(LarterBreakspear.Iext, self)
        self.rNMDA = DataSet(LarterBreakspear.rNMDA, self)
        self.VT = DataSet(LarterBreakspear.VT, self)
        self.d_V = DataSet(LarterBreakspear.d_V, self)
        self.ZT = DataSet(LarterBreakspear.ZT, self)
        self.d_Z = DataSet(LarterBreakspear.d_Z, self)
        self.QV_max = DataSet(LarterBreakspear.QV_max, self)
        self.QZ_max = DataSet(LarterBreakspear.QZ_max, self)
        self.t_scale = DataSet(LarterBreakspear.t_scale, self)
        self.variables_of_interest = Json(LarterBreakspear.variables_of_interest, self)
        self.state_variable_range = JsonFinal(LarterBreakspear.state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)


class LinearH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(LinearH5, self).__init__(path)
        self.gamma = DataSet(Linear.gamma, self)
        self.state_variable_range = JsonFinal(Linear.state_variable_range, self, json_encoder=StateVariablesEncoder,
                                              json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(Linear.variables_of_interest, self)


class Generic2dOscillatorH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(Generic2dOscillatorH5, self).__init__(path)
        self.tau = DataSet(Generic2dOscillator.tau, self)
        self.I = DataSet(Generic2dOscillator.I, self)
        self.a = DataSet(Generic2dOscillator.a, self)
        self.b = DataSet(Generic2dOscillator.b, self)
        self.c = DataSet(Generic2dOscillator.c, self)
        self.d = DataSet(Generic2dOscillator.d, self)
        self.e = DataSet(Generic2dOscillator.e, self)
        self.f = DataSet(Generic2dOscillator.f, self)
        self.g = DataSet(Generic2dOscillator.g, self)
        self.alpha = DataSet(Generic2dOscillator.alpha, self)
        self.beta = DataSet(Generic2dOscillator.beta, self)
        self.gamma = DataSet(Generic2dOscillator.gamma, self)
        self.state_variable_range = JsonFinal(Generic2dOscillator.state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(Generic2dOscillator.variables_of_interest, self)


class KuramotoH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(KuramotoH5, self).__init__(path)
        self.omega = DataSet(Kuramoto.omega, self)
        self.state_variable_range = JsonFinal(Kuramoto.state_variable_range, self, json_encoder=StateVariablesEncoder,
                                              json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(Kuramoto.variables_of_interest, self)


class SupHopfH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(SupHopfH5, self).__init__(path)
        self.a = DataSet(SupHopf.a, self)
        self.omega = DataSet(SupHopf.omega, self)
        self.state_variable_range = JsonFinal(SupHopf.state_variable_range, self, json_encoder=StateVariablesEncoder,
                                              json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(SupHopf.variables_of_interest, self)


class ReducedSetFitzHughNagumoH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(ReducedSetFitzHughNagumoH5, self).__init__(path)
        self.tau = DataSet(ReducedSetFitzHughNagumo.tau, self)
        self.a = DataSet(ReducedSetFitzHughNagumo.a, self)
        self.b = DataSet(ReducedSetFitzHughNagumo.b, self)
        self.K11 = DataSet(ReducedSetFitzHughNagumo.K11, self)
        self.K12 = DataSet(ReducedSetFitzHughNagumo.K12, self)
        self.K21 = DataSet(ReducedSetFitzHughNagumo.K21, self)
        self.sigma = DataSet(ReducedSetFitzHughNagumo.sigma, self)
        self.mu = DataSet(ReducedSetFitzHughNagumo.mu, self)
        self.state_variable_range = JsonFinal(ReducedSetFitzHughNagumo.state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ReducedSetFitzHughNagumo.variables_of_interest, self)


class ReducedSetHindmarshRoseH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(ReducedSetHindmarshRoseH5, self).__init__(path)
        self.r = DataSet(ReducedSetHindmarshRose.r, self)
        self.a = DataSet(ReducedSetHindmarshRose.a, self)
        self.b = DataSet(ReducedSetHindmarshRose.b, self)
        self.c = DataSet(ReducedSetHindmarshRose.c, self)
        self.d = DataSet(ReducedSetHindmarshRose.d, self)
        self.s = DataSet(ReducedSetHindmarshRose.s, self)
        self.xo = DataSet(ReducedSetHindmarshRose.xo, self)
        self.K11 = DataSet(ReducedSetHindmarshRose.K11, self)
        self.K12 = DataSet(ReducedSetHindmarshRose.K12, self)
        self.K21 = DataSet(ReducedSetHindmarshRose.K21, self)
        self.sigma = DataSet(ReducedSetHindmarshRose.sigma, self)
        self.mu = DataSet(ReducedSetHindmarshRose.mu, self)
        self.state_variable_range = JsonFinal(ReducedSetHindmarshRose.state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ReducedSetHindmarshRose.variables_of_interest, self)


class WilsonCowanH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(WilsonCowanH5, self).__init__(path)
        self.c_ee = DataSet(WilsonCowan.c_ee, self)
        self.c_ie = DataSet(WilsonCowan.c_ie, self)
        self.c_ei = DataSet(WilsonCowan.c_ei, self)
        self.c_ii = DataSet(WilsonCowan.c_ii, self)
        self.tau_e = DataSet(WilsonCowan.tau_e, self)
        self.tau_i = DataSet(WilsonCowan.tau_i, self)
        self.a_e = DataSet(WilsonCowan.a_e, self)
        self.b_e = DataSet(WilsonCowan.b_e, self)
        self.c_e = DataSet(WilsonCowan.c_e, self)
        self.theta_e = DataSet(WilsonCowan.theta_e, self)
        self.a_i = DataSet(WilsonCowan.a_i, self)
        self.b_i = DataSet(WilsonCowan.b_i, self)
        self.theta_i = DataSet(WilsonCowan.theta_i, self)
        self.c_i = DataSet(WilsonCowan.c_i, self)
        self.r_e = DataSet(WilsonCowan.r_e, self)
        self.r_i = DataSet(WilsonCowan.r_i, self)
        self.k_e = DataSet(WilsonCowan.k_e, self)
        self.k_i = DataSet(WilsonCowan.k_i, self)
        self.P = DataSet(WilsonCowan.P, self)
        self.Q = DataSet(WilsonCowan.Q, self)
        self.alpha_e = DataSet(WilsonCowan.alpha_e, self)
        self.alpha_i = DataSet(WilsonCowan.alpha_i, self)
        self.state_variable_range = JsonFinal(WilsonCowan.state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(WilsonCowan.variables_of_interest, self)


class ReducedWongWangH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(ReducedWongWangH5, self).__init__(path)
        self.a = DataSet(ReducedWongWang.a, self)
        self.b = DataSet(ReducedWongWang.b, self)
        self.d = DataSet(ReducedWongWang.d, self)
        self.gamma = DataSet(ReducedWongWang.gamma, self)
        self.tau_s = DataSet(ReducedWongWang.tau_s, self)
        self.w = DataSet(ReducedWongWang.w, self)
        self.J_N = DataSet(ReducedWongWang.J_N, self)
        self.I_o = DataSet(ReducedWongWang.I_o, self)
        self.sigma_noise = DataSet(ReducedWongWang.sigma_noise, self)
        self.state_variable_range = JsonFinal(ReducedWongWang.state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ReducedWongWang.variables_of_interest, self)


class ReducedWongWangExcInhH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(ReducedWongWangExcInhH5, self).__init__(path)
        self.a_e = DataSet(ReducedWongWangExcInh.a_e, self)
        self.b_e = DataSet(ReducedWongWangExcInh.b_e, self)
        self.d_e = DataSet(ReducedWongWangExcInh.d_e, self)
        self.gamma_e = DataSet(ReducedWongWangExcInh.gamma_e, self)
        self.tau_e = DataSet(ReducedWongWangExcInh.tau_e, self)
        self.w_p = DataSet(ReducedWongWangExcInh.w_p, self)
        self.J_N = DataSet(ReducedWongWangExcInh.J_N, self)
        self.W_e = DataSet(ReducedWongWangExcInh.W_e, self)
        self.a_i = DataSet(ReducedWongWangExcInh.a_i, self)
        self.b_i = DataSet(ReducedWongWangExcInh.b_i, self)
        self.d_i = DataSet(ReducedWongWangExcInh.d_i, self)
        self.gamma_i = DataSet(ReducedWongWangExcInh.gamma_i, self)
        self.tau_i = DataSet(ReducedWongWangExcInh.tau_i, self)
        self.J_i = DataSet(ReducedWongWangExcInh.J_i, self)
        self.W_i = DataSet(ReducedWongWangExcInh.W_i, self)
        self.I_o = DataSet(ReducedWongWangExcInh.I_o, self)
        self.G = DataSet(ReducedWongWangExcInh.G, self)
        self.lamda = DataSet(ReducedWongWangExcInh.lamda, self)
        self.state_variable_range = JsonFinal(ReducedWongWangExcInh.state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ReducedWongWangExcInh.variables_of_interest, self)


class ZerlautFirstOrderH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(ZerlautFirstOrderH5, self).__init__(path)
        self.g_L = DataSet(ZerlautFirstOrder.g_L, self)
        self.E_L_e = DataSet(ZerlautFirstOrder.E_L_e, self)
        self.E_L_i = DataSet(ZerlautFirstOrder.E_L_i, self)
        self.C_m = DataSet(ZerlautFirstOrder.C_m, self)
        self.b = DataSet(ZerlautFirstOrder.b, self)
        self.tau_w = DataSet(ZerlautFirstOrder.tau_w, self)
        self.E_e = DataSet(ZerlautFirstOrder.E_e, self)
        self.E_i = DataSet(ZerlautFirstOrder.E_i, self)
        self.Q_e = DataSet(ZerlautFirstOrder.Q_e, self)
        self.Q_i = DataSet(ZerlautFirstOrder.Q_i, self)
        self.tau_e = DataSet(ZerlautFirstOrder.tau_e, self)
        self.tau_i = DataSet(ZerlautFirstOrder.tau_i, self)
        self.N_tot = DataSet(ZerlautFirstOrder.N_tot, self)
        self.p_connect = DataSet(ZerlautFirstOrder.p_connect, self)
        self.g = DataSet(ZerlautFirstOrder.g, self)
        self.T = DataSet(ZerlautFirstOrder.T, self)
        self.P_e = DataSet(ZerlautFirstOrder.P_e, self)
        self.P_i = DataSet(ZerlautFirstOrder.P_i, self)
        self.external_input = DataSet(ZerlautFirstOrder.external_input, self)
        self.state_variable_range = JsonFinal(ZerlautFirstOrder.state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ZerlautFirstOrder.variables_of_interest, self)


class ZerlautSecondOrderH5(ZerlautFirstOrderH5):

    def __init__(self, path):
        super(ZerlautSecondOrderH5, self).__init__(path)
        self.state_variable_range = JsonFinal(ZerlautSecondOrder.state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ZerlautSecondOrder.variables_of_interest, self)
