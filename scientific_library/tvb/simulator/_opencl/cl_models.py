import pyopencl
import pyopencl.array
import numpy
from models import CLComponent,CLModel
from ..models.oscillator import Generic2dOscillator,Kuramoto
from ..models.jansen_rit import JansenRit, ZetterbergJansen
from ..models.linear import Linear
from ..models.larter_breakspear import LarterBreakspear
from ..models.epileptor import Epileptor
from ..models.hopfield import Hopfield
from ..models.stefanescu_jirsa import ReducedSetFitzHughNagumo, ReducedSetHindmarshRose
from ..models.wilson_cowan import WilsonCowan


class CL_Kuramoto(Kuramoto, CLModel):
    _opencl_ordered_params = "omega".split()
    _opencl_program_source_file = "Kuramoto.cl"
    with open(_opencl_program_source_file, "r") as fd:
        _opencl_program_source = fd.read()


class CL_Generic2D(Generic2dOscillator, CLModel):

    n_states = 2
    _opencl_ordered_params = "tau, a, b, c, I, d, e, f, g, alpha, beta, gamma".split()
    _opencl_program_source_file = "Generic2D.cl"
    with open(_opencl_program_source_file, "r") as fd:
         _opencl_program_source = fd.read()

class CL_Jansen_rit(JansenRit, CLModel):
    n_states = 6
    _opencl_ordered_params = "A B a b v0 nu_max r J a_1 a_2 a_3 a_4 p_min p_max mu".split()
    _opencl_program_source_file = "Jansen_rit.cl"


class CL_Zetterberg_Jasen(ZetterbergJansen, CLModel):
    n_states = 12
    _opencl_ordered_params = " He Hi ke ki e0 rho_2 rho_1 gamma_1 gamma_2 gamma_3 gamma_4 gamma_5 P U Q Heke Hiki ke_2 ki_2 keke kiki".split()
    _opencl_program_source_file = "Zetterberg_Jasen.cl"


class CL_Linear( Linear, CLModel ):
    n_states = 1
    _opencl_ordered_params = "gamma"
    _opencl_program_source_file = "Linear.cl"

class CL_Hopfield( Hopfield, CLModel ):
    n_states = 2
    _opencl_ordered_params = "taux tauT dynamic".split()
    _opencl_program_source_file = "Hopfield.cl"

#TODO: Verify input parameters
class CL_ReducedSetFitzHughNagumo( ReducedSetFitzHughNagumo, CLModel):
    n_states = 4
    _opencl_ordered_params = "tau a b K11 K12 K21 sigma mu Aik Bik Cik e_i f_i IE_i II_i m_i n_i".split()
    _opencl_program_source_file = "ReducedSetFitzHughNagumo.cl"

class CL_ReducedSetHindmarshRose( ReducedSetHindmarshRose , CLModel):
    n_states = 6
    _opencl_ordered_params = "r a b c d s xo K11 K12 K21 sigma mu A_iK B_iK C_iK a_i b_i c_i d_i e_i f_i h_i p_i IE_i II_i m_i n_i gamma_1T gamma_2T gamma_3T".split()
    _opencl_program_source_file = "ReducedSetHindmarshRose.cl"


class CL_Epileptor( Epileptor, CLModel):
    n_states = 6
    _opencl_ordered_params = "x0 Iext Iext2 a b slope tt Kvf c d r Ks Kf aa tau ydot".split()
    _opencl_program_source_file = "Epileptor.cl"

class CL_Later_Breakspear( LarterBreakspear, CLModel):
    n_states = 3
    _opencl_ordered_params = " gCa gK gL phi gNa TK TCa TNa VCa VK VL VNa d_K tau_K d_Na d_Ca aei aie b C ane ani aee Iext rNMDA VT d_V ZT d_Z QV_max QZ_max t_scale".split()
    _opencl_program_source_file = "Larter_breakspear.cl"

class CL_WilsonCowan ( WilsonCowan, CLModel ):
    n_states = 2
    _opencl_ordered_params = " c_ee c_ei c_ie c_ii tau_e tau_i a_e b_e c_e a_i b_i c_i r_e r_i k_e k_i P Q theta_e theta_i alpha_e alpha_i".split()
    _opencl_program_source_file = "Wilson_Cowan.cl"



def test():

    m = CL_Linear()
test()