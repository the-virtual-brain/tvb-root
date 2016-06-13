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
from ..models.stefanescu_jirsa import ReducedSetFitzHughNagumo
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
    _opencl_ordered_params = "He Hi ke ki e0 rho_2 rho_1 gamma_1 gamma_2 gamma_3 gamma_4 gamma_5 P U Q".split()
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
    _opencl_ordered_params = "tau a b K11 K12 K21 sigma mu".split()
    _opencl_program_source_file = "ReducedSetFitzHughNagumo.cl"

class CL_Epileptor( Epileptor,CLModel ):
    n_states = 6
    _opencl_ordered_params = "x0 Iext Iext2 a b slope tt Kvf c d r Ks Kf aa tau ydot".split()
    _opencl_program_source_file = "Epileptor.cl"

def test():
    m = CL_Linear()
test()