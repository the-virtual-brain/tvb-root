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
from tvb.simulator.models import ModelsEnum
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

        self.a = DataSet(ModelsEnum.EPILEPTOR.get_class().a, self)
        self.b = DataSet(ModelsEnum.EPILEPTOR.get_class().b, self)
        self.c = DataSet(ModelsEnum.EPILEPTOR.get_class().c, self)
        self.d = DataSet(ModelsEnum.EPILEPTOR.get_class().d, self)
        self.r = DataSet(ModelsEnum.EPILEPTOR.get_class().r, self)
        self.s = DataSet(ModelsEnum.EPILEPTOR.get_class().s, self)
        self.x0 = DataSet(ModelsEnum.EPILEPTOR.get_class().x0, self)
        self.Iext = DataSet(ModelsEnum.EPILEPTOR.get_class().Iext, self)
        self.slope = DataSet(ModelsEnum.EPILEPTOR.get_class().slope, self)
        self.Iext2 = DataSet(ModelsEnum.EPILEPTOR.get_class().Iext2, self)
        self.tau = DataSet(ModelsEnum.EPILEPTOR.get_class().tau, self)
        self.aa = DataSet(ModelsEnum.EPILEPTOR.get_class().aa, self)
        self.bb = DataSet(ModelsEnum.EPILEPTOR.get_class().bb, self)
        self.Kvf = DataSet(ModelsEnum.EPILEPTOR.get_class().Kvf, self)
        self.Kf = DataSet(ModelsEnum.EPILEPTOR.get_class().Kf, self)
        self.Ks = DataSet(ModelsEnum.EPILEPTOR.get_class().Ks, self)
        self.tt = DataSet(ModelsEnum.EPILEPTOR.get_class().tt, self)
        self.modification = DataSet(ModelsEnum.EPILEPTOR.get_class().modification, self)
        self.state_variable_range = JsonFinal(ModelsEnum.EPILEPTOR.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder,
                                              json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.EPILEPTOR.get_class().variables_of_interest, self)


class Epileptor2DH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(Epileptor2DH5, self).__init__(path)

        self.a = DataSet(ModelsEnum.EPILEPTOR_2D.get_class().a, self)
        self.b = DataSet(ModelsEnum.EPILEPTOR_2D.get_class().b, self)
        self.c = DataSet(ModelsEnum.EPILEPTOR_2D.get_class().c, self)
        self.d = DataSet(ModelsEnum.EPILEPTOR_2D.get_class().d, self)
        self.r = DataSet(ModelsEnum.EPILEPTOR_2D.get_class().r, self)
        self.x0 = DataSet(ModelsEnum.EPILEPTOR_2D.get_class().x0, self)
        self.Iext = DataSet(ModelsEnum.EPILEPTOR_2D.get_class().Iext, self)
        self.slope = DataSet(ModelsEnum.EPILEPTOR_2D.get_class().slope, self)
        self.Kvf = DataSet(ModelsEnum.EPILEPTOR_2D.get_class().Kvf, self)
        self.Ks = DataSet(ModelsEnum.EPILEPTOR_2D.get_class().Ks, self)
        self.tt = DataSet(ModelsEnum.EPILEPTOR_2D.get_class().tt, self)
        self.modification = DataSet(ModelsEnum.EPILEPTOR_2D.get_class().modification, self)
        self.state_variable_range = JsonFinal(ModelsEnum.EPILEPTOR_2D.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.EPILEPTOR_2D.get_class().variables_of_interest, self)


class EpileptorCodim3H5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(EpileptorCodim3H5, self).__init__(path)
        self.mu1_start = DataSet(ModelsEnum.EPILEPTOR_CODIM_3.get_class().mu1_start, self)
        self.mu2_start = DataSet(ModelsEnum.EPILEPTOR_CODIM_3.get_class().mu2_start, self)
        self.nu_start = DataSet(ModelsEnum.EPILEPTOR_CODIM_3.get_class().nu_start, self)
        self.mu1_stop = DataSet(ModelsEnum.EPILEPTOR_CODIM_3.get_class().mu1_stop, self)
        self.mu2_stop = DataSet(ModelsEnum.EPILEPTOR_CODIM_3.get_class().mu2_stop, self)
        self.nu_stop = DataSet(ModelsEnum.EPILEPTOR_CODIM_3.get_class().nu_stop, self)
        self.b = DataSet(ModelsEnum.EPILEPTOR_CODIM_3.get_class().b, self)
        self.R = DataSet(ModelsEnum.EPILEPTOR_CODIM_3.get_class().R, self)
        self.c = DataSet(ModelsEnum.EPILEPTOR_CODIM_3.get_class().c, self)
        self.dstar = DataSet(ModelsEnum.EPILEPTOR_CODIM_3.get_class().dstar, self)
        self.Ks = DataSet(ModelsEnum.EPILEPTOR_CODIM_3.get_class().Ks, self)
        self.N = DataSet(ModelsEnum.EPILEPTOR_CODIM_3.get_class().N, self)
        self.modification = DataSet(ModelsEnum.EPILEPTOR_CODIM_3.get_class().modification, self)
        self.state_variable_range = JsonFinal(ModelsEnum.EPILEPTOR_CODIM_3.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.EPILEPTOR_CODIM_3.get_class().variables_of_interest, self)


class EpileptorCodim3SlowModH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(EpileptorCodim3SlowModH5, self).__init__(path)
        self.mu1_Ain = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu1_Ain, self)
        self.mu2_Ain = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu2_Ain, self)
        self.nu_Ain = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().nu_Ain, self)
        self.mu1_Bin = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu1_Bin, self)
        self.mu2_Bin = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu2_Bin, self)
        self.nu_Bin = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().nu_Bin, self)
        self.mu1_Aend = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu1_Aend, self)
        self.mu2_Aend = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu2_Aend, self)
        self.nu_Aend = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().nu_Aend, self)
        self.mu1_Bend = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu1_Bend, self)
        self.mu2_Bend = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().mu2_Bend, self)
        self.nu_Bend = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().nu_Bend, self)
        self.b = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().b, self)
        self.R = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().R, self)
        self.c = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().c, self)
        self.cA = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().cA, self)
        self.cB = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().cB, self)
        self.dstar = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().dstar, self)
        self.Ks = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().Ks, self)
        self.N = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().N, self)
        self.modification = DataSet(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().modification, self)
        self.state_variable_range = JsonFinal(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class().variables_of_interest, self)


class HopfieldH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(HopfieldH5, self).__init__(path)
        self.taux = DataSet(ModelsEnum.HOPFIELD.get_class().taux, self)
        self.tauT = DataSet(ModelsEnum.HOPFIELD.get_class().tauT, self)
        self.dynamic = DataSet(ModelsEnum.HOPFIELD.get_class().dynamic, self)
        self.state_variable_range = JsonFinal(ModelsEnum.HOPFIELD.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder,
                                              json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.HOPFIELD.get_class().variables_of_interest, self)


class JansenRitH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(JansenRitH5, self).__init__(path)
        self.A = DataSet(ModelsEnum.JANSEN_RIT.get_class().A, self)
        self.B = DataSet(ModelsEnum.JANSEN_RIT.get_class().B, self)
        self.a = DataSet(ModelsEnum.JANSEN_RIT.get_class().a, self)
        self.b = DataSet(ModelsEnum.JANSEN_RIT.get_class().b, self)
        self.v0 = DataSet(ModelsEnum.JANSEN_RIT.get_class().v0, self)
        self.nu_max = DataSet(ModelsEnum.JANSEN_RIT.get_class().nu_max, self)
        self.r = DataSet(ModelsEnum.JANSEN_RIT.get_class().r, self)
        self.J = DataSet(ModelsEnum.JANSEN_RIT.get_class().J, self)
        self.a_1 = DataSet(ModelsEnum.JANSEN_RIT.get_class().a_1, self)
        self.a_2 = DataSet(ModelsEnum.JANSEN_RIT.get_class().a_2, self)
        self.a_3 = DataSet(ModelsEnum.JANSEN_RIT.get_class().a_3, self)
        self.a_4 = DataSet(ModelsEnum.JANSEN_RIT.get_class().a_4, self)
        self.p_min = DataSet(ModelsEnum.JANSEN_RIT.get_class().p_min, self)
        self.p_max = DataSet(ModelsEnum.JANSEN_RIT.get_class().p_max, self)
        self.mu = DataSet(ModelsEnum.JANSEN_RIT.get_class().mu, self)
        self.state_variable_range = JsonFinal(ModelsEnum.JANSEN_RIT.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder,
                                              json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.JANSEN_RIT.get_class().variables_of_interest, self)


class ZetterbergJansenH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(ZetterbergJansenH5, self).__init__(path)
        self.He = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().He, self)
        self.Hi = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().Hi, self)
        self.ke = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().ke, self)
        self.ki = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().ki, self)
        self.e0 = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().e0, self)
        self.rho_2 = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().rho_2, self)
        self.rho_1 = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().rho_1, self)
        self.gamma_1 = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_1, self)
        self.gamma_2 = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_2, self)
        self.gamma_3 = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_3, self)
        self.gamma_4 = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_4, self)
        self.gamma_5 = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_5, self)
        self.gamma_1T = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_1T, self)
        self.gamma_2T = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_2T, self)
        self.gamma_3T = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().gamma_3T, self)
        self.P = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().P, self)
        self.U = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().U, self)
        self.Q = DataSet(ModelsEnum.ZETTERBERG_JANSEN.get_class().Q, self)
        self.state_variable_range = JsonFinal(ModelsEnum.ZETTERBERG_JANSEN.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.ZETTERBERG_JANSEN.get_class().variables_of_interest, self)


class EpileptorRestingStateH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(EpileptorRestingStateH5, self).__init__(path)
        self.a = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().a, self)
        self.b = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().b, self)
        self.c = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().c, self)
        self.d = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().d, self)
        self.r = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().r, self)
        self.s = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().s, self)
        self.x0 = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().x0, self)
        self.Iext = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().Iext, self)
        self.slope = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().slope, self)
        self.Iext2 = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().Iext2, self)
        self.tau = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().tau, self)
        self.aa = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().aa, self)
        self.bb = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().bb, self)
        self.Kvf = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().Kvf, self)
        self.Kf = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().Kf, self)
        self.Ks = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().Ks, self)
        self.tt = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().tt, self)
        self.tau_rs = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().tau_rs, self)
        self.I_rs = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().I_rs, self)
        self.a_rs = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().a_rs, self)
        self.b_rs = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().b_rs, self)
        self.d_rs = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().d_rs, self)
        self.e_rs = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().e_rs, self)
        self.f_rs = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().f_rs, self)
        self.alpha_rs = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().alpha_rs, self)
        self.beta_rs = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().beta_rs, self)
        self.gamma_rs = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().gamma_rs, self)
        self.K_rs = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().K_rs, self)
        self.p = DataSet(ModelsEnum.EPILEPTOR_RS.get_class().p, self)
        self.state_variable_range = JsonFinal(ModelsEnum.EPILEPTOR_RS.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.EPILEPTOR_RS.get_class().variables_of_interest, self)


class LarterBreakspearH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(LarterBreakspearH5, self).__init__(path)
        self.gCa = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().gCa, self)
        self.gK = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().gK, self)
        self.gL = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().gL, self)
        self.phi = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().phi, self)
        self.gNa = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().gNa, self)
        self.TK = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().TK, self)
        self.TCa = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().TCa, self)
        self.TNa = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().TNa, self)
        self.VCa = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().VCa, self)
        self.VK = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().VK, self)
        self.VL = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().VL, self)
        self.VNa = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().VNa, self)
        self.d_K = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().d_K, self)
        self.tau_K = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().tau_K, self)
        self.d_Na = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().d_Na, self)
        self.d_Ca = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().d_Ca, self)
        self.aei = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().aei, self)
        self.aie = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().aie, self)
        self.b = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().b, self)
        self.C = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().C, self)
        self.ane = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().ane, self)
        self.ani = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().ani, self)
        self.aee = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().aee, self)
        self.Iext = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().Iext, self)
        self.rNMDA = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().rNMDA, self)
        self.VT = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().VT, self)
        self.d_V = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().d_V, self)
        self.ZT = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().ZT, self)
        self.d_Z = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().d_Z, self)
        self.QV_max = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().QV_max, self)
        self.QZ_max = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().QZ_max, self)
        self.t_scale = DataSet(ModelsEnum.LARTER_BREAKSPEAR.get_class().t_scale, self)
        self.variables_of_interest = Json(ModelsEnum.LARTER_BREAKSPEAR.get_class().variables_of_interest, self)
        self.state_variable_range = JsonFinal(ModelsEnum.LARTER_BREAKSPEAR.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)


class LinearH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(LinearH5, self).__init__(path)
        self.gamma = DataSet(ModelsEnum.LINEAR.get_class().gamma, self)
        self.state_variable_range = JsonFinal(ModelsEnum.LINEAR.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder,
                                              json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.LINEAR.get_class().variables_of_interest, self)


class Generic2dOscillatorH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(Generic2dOscillatorH5, self).__init__(path)
        self.tau = DataSet(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().tau, self)
        self.I = DataSet(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().I, self)
        self.a = DataSet(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().a, self)
        self.b = DataSet(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().b, self)
        self.c = DataSet(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().c, self)
        self.d = DataSet(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().d, self)
        self.e = DataSet(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().e, self)
        self.f = DataSet(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().f, self)
        self.g = DataSet(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().g, self)
        self.alpha = DataSet(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().alpha, self)
        self.beta = DataSet(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().beta, self)
        self.gamma = DataSet(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().gamma, self)
        self.state_variable_range = JsonFinal(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.GENERIC_2D_OSCILLATOR.get_class().variables_of_interest, self)


class KuramotoH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(KuramotoH5, self).__init__(path)
        self.omega = DataSet(ModelsEnum.KURAMOTO.get_class().omega, self)
        self.state_variable_range = JsonFinal(ModelsEnum.KURAMOTO.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder,
                                              json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.KURAMOTO.get_class().variables_of_interest, self)


class SupHopfH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(SupHopfH5, self).__init__(path)
        self.a = DataSet(ModelsEnum.SUP_HOPF.get_class().a, self)
        self.omega = DataSet(ModelsEnum.SUP_HOPF.get_class().omega, self)
        self.state_variable_range = JsonFinal(ModelsEnum.SUP_HOPF.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder,
                                              json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.SUP_HOPF.get_class().variables_of_interest, self)


class ReducedSetFitzHughNagumoH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(ReducedSetFitzHughNagumoH5, self).__init__(path)
        self.tau = DataSet(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().tau, self)
        self.a = DataSet(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().a, self)
        self.b = DataSet(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().b, self)
        self.K11 = DataSet(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().K11, self)
        self.K12 = DataSet(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().K12, self)
        self.K21 = DataSet(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().K21, self)
        self.sigma = DataSet(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().sigma, self)
        self.mu = DataSet(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().mu, self)
        self.state_variable_range = JsonFinal(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().state_variable_range,
                                              self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class().variables_of_interest,
                                          self)


class ReducedSetHindmarshRoseH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(ReducedSetHindmarshRoseH5, self).__init__(path)
        self.r = DataSet(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().r, self)
        self.a = DataSet(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().a, self)
        self.b = DataSet(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().b, self)
        self.c = DataSet(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().c, self)
        self.d = DataSet(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().d, self)
        self.s = DataSet(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().s, self)
        self.xo = DataSet(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().xo, self)
        self.K11 = DataSet(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().K11, self)
        self.K12 = DataSet(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().K12, self)
        self.K21 = DataSet(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().K21, self)
        self.sigma = DataSet(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().sigma, self)
        self.mu = DataSet(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().mu, self)
        self.state_variable_range = JsonFinal(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().state_variable_range,
                                              self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class().variables_of_interest, self)


class WilsonCowanH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(WilsonCowanH5, self).__init__(path)
        self.c_ee = DataSet(ModelsEnum.WILSON_COWAN.get_class().c_ee, self)
        self.c_ie = DataSet(ModelsEnum.WILSON_COWAN.get_class().c_ie, self)
        self.c_ei = DataSet(ModelsEnum.WILSON_COWAN.get_class().c_ei, self)
        self.c_ii = DataSet(ModelsEnum.WILSON_COWAN.get_class().c_ii, self)
        self.tau_e = DataSet(ModelsEnum.WILSON_COWAN.get_class().tau_e, self)
        self.tau_i = DataSet(ModelsEnum.WILSON_COWAN.get_class().tau_i, self)
        self.a_e = DataSet(ModelsEnum.WILSON_COWAN.get_class().a_e, self)
        self.b_e = DataSet(ModelsEnum.WILSON_COWAN.get_class().b_e, self)
        self.c_e = DataSet(ModelsEnum.WILSON_COWAN.get_class().c_e, self)
        self.theta_e = DataSet(ModelsEnum.WILSON_COWAN.get_class().theta_e, self)
        self.a_i = DataSet(ModelsEnum.WILSON_COWAN.get_class().a_i, self)
        self.b_i = DataSet(ModelsEnum.WILSON_COWAN.get_class().b_i, self)
        self.theta_i = DataSet(ModelsEnum.WILSON_COWAN.get_class().theta_i, self)
        self.c_i = DataSet(ModelsEnum.WILSON_COWAN.get_class().c_i, self)
        self.r_e = DataSet(ModelsEnum.WILSON_COWAN.get_class().r_e, self)
        self.r_i = DataSet(ModelsEnum.WILSON_COWAN.get_class().r_i, self)
        self.k_e = DataSet(ModelsEnum.WILSON_COWAN.get_class().k_e, self)
        self.k_i = DataSet(ModelsEnum.WILSON_COWAN.get_class().k_i, self)
        self.P = DataSet(ModelsEnum.WILSON_COWAN.get_class().P, self)
        self.Q = DataSet(ModelsEnum.WILSON_COWAN.get_class().Q, self)
        self.alpha_e = DataSet(ModelsEnum.WILSON_COWAN.get_class().alpha_e, self)
        self.alpha_i = DataSet(ModelsEnum.WILSON_COWAN.get_class().alpha_i, self)
        self.state_variable_range = JsonFinal(ModelsEnum.WILSON_COWAN.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.WILSON_COWAN.get_class().variables_of_interest, self)


class ReducedWongWangH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(ReducedWongWangH5, self).__init__(path)
        self.a = DataSet(ModelsEnum.REDUCED_WONG_WANG.get_class().a, self)
        self.b = DataSet(ModelsEnum.REDUCED_WONG_WANG.get_class().b, self)
        self.d = DataSet(ModelsEnum.REDUCED_WONG_WANG.get_class().d, self)
        self.gamma = DataSet(ModelsEnum.REDUCED_WONG_WANG.get_class().gamma, self)
        self.tau_s = DataSet(ModelsEnum.REDUCED_WONG_WANG.get_class().tau_s, self)
        self.w = DataSet(ModelsEnum.REDUCED_WONG_WANG.get_class().w, self)
        self.J_N = DataSet(ModelsEnum.REDUCED_WONG_WANG.get_class().J_N, self)
        self.I_o = DataSet(ModelsEnum.REDUCED_WONG_WANG.get_class().I_o, self)
        self.sigma_noise = DataSet(ModelsEnum.REDUCED_WONG_WANG.get_class().sigma_noise, self)
        self.state_variable_range = JsonFinal(ModelsEnum.REDUCED_WONG_WANG.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.REDUCED_WONG_WANG.get_class().variables_of_interest, self)


class ReducedWongWangExcInhH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(ReducedWongWangExcInhH5, self).__init__(path)
        self.a_e = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().a_e, self)
        self.b_e = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().b_e, self)
        self.d_e = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().d_e, self)
        self.gamma_e = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().gamma_e, self)
        self.tau_e = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().tau_e, self)
        self.w_p = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().w_p, self)
        self.J_N = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().J_N, self)
        self.W_e = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().W_e, self)
        self.a_i = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().a_i, self)
        self.b_i = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().b_i, self)
        self.d_i = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().d_i, self)
        self.gamma_i = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().gamma_i, self)
        self.tau_i = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().tau_i, self)
        self.J_i = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().J_i, self)
        self.W_i = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().W_i, self)
        self.I_o = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().I_o, self)
        self.G = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().G, self)
        self.lamda = DataSet(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().lamda, self)
        self.state_variable_range = JsonFinal(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().state_variable_range,
                                              self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class().variables_of_interest, self)


class ZerlautAdaptationFirstOrderH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(ZerlautAdaptationFirstOrderH5, self).__init__(path)
        self.g_L = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().g_L, self)
        self.E_L_e = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().E_L_e, self)
        self.E_L_i = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().E_L_i, self)
        self.C_m = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().C_m, self)
        self.a_e = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().a_e, self)
        self.a_i = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().a_i, self)
        self.b_e = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().b_e, self)
        self.b_i = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().b_i, self)
        self.tau_w_e = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().tau_w_e, self)
        self.tau_w_i = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().tau_w_i, self)
        self.E_e = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().E_e, self)
        self.E_i = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().E_i, self)
        self.Q_e = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().Q_e, self)
        self.Q_i = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().Q_i, self)
        self.tau_e = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().tau_e, self)
        self.tau_i = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().tau_i, self)
        self.N_tot = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().N_tot, self)
        self.p_connect = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().p_connect, self)
        self.g = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().g, self)
        self.K_ext_e = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().K_ext_e, self)
        self.K_ext_i = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().K_ext_i, self)
        self.T = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().T, self)
        self.P_e = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().P_e, self)
        self.P_i = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().P_i, self)
        self.external_input_ex_ex = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().external_input_ex_ex, self)
        self.external_input_ex_in = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().external_input_ex_in, self)
        self.external_input_in_ex = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().external_input_in_ex, self)
        self.external_input_in_in = DataSet(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().external_input_in_in, self)
        self.state_variable_range = JsonFinal(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.ZERLAUT_FIRST_ORDER.get_class().variables_of_interest, self)


class ZerlautAdaptationSecondOrderH5(ZerlautAdaptationFirstOrderH5):

    def __init__(self, path):
        super(ZerlautAdaptationSecondOrderH5, self).__init__(path)
        self.state_variable_range = JsonFinal(ModelsEnum.ZERLAUT_SECOND_ORDER.get_class().state_variable_range, self,
                                              json_encoder=StateVariablesEncoder, json_decoder=StateVariablesDecoder)
        self.variables_of_interest = Json(ModelsEnum.ZERLAUT_SECOND_ORDER.get_class().variables_of_interest, self)
