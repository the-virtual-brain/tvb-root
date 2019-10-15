# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

import pyopencl
import pyopencl.array
import numpy
import os
from .models import CLComponent,CLModel
from ..models.oscillator import Generic2dOscillator,Kuramoto
from ..models.jansen_rit import JansenRit, ZetterbergJansen
from ..models.linear import Linear
from ..models.larter_breakspear import LarterBreakspear
from ..models.epileptor import Epileptor
from ..models.hopfield import Hopfield
from ..models.stefanescu_jirsa import ReducedSetFitzHughNagumo, ReducedSetHindmarshRose
from ..models.wilson_cowan import WilsonCowan


class CL_Kuramoto(Kuramoto, CLModel):
    n_states = 1
    _opencl_ordered_params = "omega".split()
    _opencl_program_source_file = "Kuramoto.cl"
    _opencl_program_source = """
    __kernel void dfun(__global float *state, __global float *coupling,
                   __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        // this is boilerplate and could be generated
        float S=state[i], omega=param[i];
        float I = coupling[i];
        deriv[i] = omega + I;
    }
    """
    # with open(_opencl_program_source_file, "r") as fd:
    #     _opencl_program_source = fd.read()


class CL_Generic2D(Generic2dOscillator, CLModel):

    n_states = 2
    _opencl_ordered_params = "tau a b c I d e f g alpha beta gamma".split()
    _opencl_program_source_file = "Generic2D.cl"
    _opencl_program_source = """
    //Generic2dOscillator
    #define indexNum(cur,totalN) cur*n+i
    __kernel void dfun(__global float *state, __global float *coupling,
                       __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        // this is boilerplate and could be generated
        float V = state[i*2], W = state[i*2+1];
        //tau, a, b, c, I, d, e, f, g, alpha, beta, gamma
        float c_0 = coupling[i];
        float tau = param[n+i], a = param[2*n+i],
        b = param[3*n+i], c = param[4*n+i], I= param[5*n+i], d = param[6*n+i],
        e = param[7*n+i], f = param[8*n+i], g= param[9*n+i], alpha= param[10*n+i],
        beta = param[11*n+i], gamma = param[12*n+i];

        deriv[indexNum(0,2)] = d * tau * (alpha * W - f * V*V*V + e * V*V + g * V + gamma * I + gamma *c_0);
        deriv[indexNum(1,2)] = d * (a + b * V + c * V*V - beta * W) / tau;
    }"""
    # with open(os.path.join('/',_opencl_program_source_file), "r") as fd:
    #       _opencl_program_source = fd.read()

class CL_Jansen_rit(JansenRit, CLModel):
    n_states = 6
    _opencl_ordered_params = "A B a b v0 nu_max r J a_1 a_2 a_3 a_4 p_min p_max mu".split()
    _opencl_program_source_file = "Jansen_rit.cl"
    _opencl_program_source = """
    #define indexNum(cur,totalN) cur*n+i // consider i*totalN+cur
    __kernel void dfun(__global float *state, __global float *coupling,
                       __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        // this is boilerplate and could be generated
        float y0 = state[i*6], y1 = state[i*6+1],y2 = state[i*6+2],
        y3 = state[i*6+3], y4 = state[i*6+4], y5 = state[i*6+5];
        float c = coupling[i]; ;
        // nu_max  r  v0  a  a_1  a_2  a_3  a_4  A  b  B  J  mu //total 13 parameters
        float nu_max = param[indexNum(0,13)];
        float  r = param[indexNum(1,13)];
        float  v0 = param[indexNum(2,13)];
        float  a = param[indexNum(3,13)];
        float  a_1 = param[indexNum(4,13)];
        float  a_2 = param[indexNum(5,13)];
        float  a_3 = param[indexNum(6,13)];
        float  a_4 = param[indexNum(7,13)];
        float  A = param[indexNum(8,13)];
        float  b = param[indexNum(9,13)];
        float  B = param[indexNum(10,13)];
        float  J = param[indexNum(11,13)];
        float  mu = param[indexNum(12,13)];

        float src = y1 - y2;

        float sigm_y1_y2 = 2.0 * nu_max / (1.0 + exp(r * (v0 - (y1 - y2))));
        float sigm_y0_1 = 2.0 * nu_max / (1.0 + exp(r * (v0 - (a_1 * J * y0))));
        float sigm_y0_3 = 2.0 * nu_max / (1.0 + exp(r * (v0 - (a_3 * J * y0))));
        deriv[indexNum(0,6)] = y3;
        deriv[indexNum(1,6)] = y4;
        deriv[indexNum(2,6)] = y5;
        deriv[indexNum(3,6)] = A * a * sigm_y1_y2 - 2.0 * a * y3 - a * a * y0;
        deriv[indexNum(4,6)] = A * a * (mu + a_2 * J * sigm_y0_1 + c + src) - 2.0 * a * y4 - a * a * y1;
        deriv[indexNum(5,6)] = B * b * (a_4 * J * sigm_y0_3) - 2.0 * b * y5 - b *b * y2;
    }

    """


class CL_Zetterberg_Jasen(ZetterbergJansen, CLModel):
    n_states = 12
    _opencl_ordered_params = " He Hi ke ki e0 rho_2 rho_1 gamma_1 gamma_2 gamma_3 gamma_4 gamma_5 P U Q Heke Hiki ke_2 ki_2 keke kiki".split()
    _opencl_program_source_file = "Zetterberg_Jasen.cl"
    _opencl_program_source = """
    #define indexNum(cur,totalN) cur*n+i
    #define sigma_fun(sv) ( rho_1 *(rho_2 - sv) > 709 ? 0 : 2*e0 / (1+rho_1 *(rho_2 - sv)))


    __kernel void dfun(__global float *state, __global float *coupling,
                       __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);
        int n_param = 12;
        // this is boilerplate and could be generated
        float v1 = state[indexNum(0,12)], y1 = state[indexNum(1,12)],v2 = state[indexNum(2,12)],
        y2 = state[indexNum(3,12)],v3 = state[indexNum(4,12)], y3 = state[indexNum(5,12)],
        v4 = state[indexNum(6,12)], y4 = state[indexNum(7,12)], v5 = state[indexNum(8,12)],
        y5 = state[indexNum(9,12)], v6 = state[indexNum(10,12)], v7 = state[indexNum(11,12)],
        c = coupling[i];
        //He Hi ke ki e0 rho_2 rho_1 gamma_1 gamma_2 gamma_3 gamma_4 gamma_5 P U Q Heke Hiki ke_2 ki_2 keke kiki gamma_1T gamma_2T gamma_3T
        float He = param[indexNum(0,24)];
        float Hi = param[indexNum(1,24)];
        float ke = param[indexNum(2,24)];
        float ki = param[indexNum(3,24)];
        float e0 = param[indexNum(4,24)];
        float rho_2 = param[indexNum(5,24)];
        float rho_1 = param[indexNum(6,24)];
        float gamma_1 = param[indexNum(7,24)];
        float gamma_2 = param[indexNum(8,24)];
        float gamma_3 = param[indexNum(9,24)];
        float gamma_4 = param[indexNum(10,24)];
        float gamma_5 = param[indexNum(11,24)];
        float P = param[indexNum(12,24)];
        float U = param[indexNum(13,24)];
        float Q = param[indexNum(14,24)];
        float Heke = param[indexNum(15,24)];
        float Hiki = param[indexNum(16,24)];
        float ke_2 = param[indexNum(17,24)];
        float ki_2 = param[indexNum(18,24)];
        float keke = param[indexNum(19,24)];
        float kiki = param[indexNum(20,24)];
        float gamma_1T = param[indexNum(21,24)];
        float gamma_2T = param[indexNum(22,24)];
        float gamma_3T = param[indexNum(23,24)];
        // TODO: implement local coupling
        float locol_coupling = 1;

        float coupled_input = c + 6*locol_coupling;

        deriv[indexNum(0,12)] = y1;
        deriv[indexNum(1,12)] = Heke * (gamma_1 * sigma_fun(v2 - v3) + gamma_1T * (U + coupled_input )) - ke_2 * y1 - keke * v1;
        // exc input to the pyramidal cells
        deriv[indexNum(2,12)] = y2;
        deriv[indexNum(3,12)] = Heke * (gamma_2 * sigma_fun(v1)      + gamma_2T * (P + coupled_input )) - ke_2 * y2 - keke * v2;
        // inh input to the pyramidal cells
        deriv[indexNum(4,12)] = y3;
        deriv[indexNum(5,12)] = Hiki * (gamma_4 * sigma_fun(v4 - v5)) - ki_2 * y3 - kiki * v3;
        deriv[indexNum(6,12)] = y4;
        // exc input to the inhibitory interneurons
        deriv[indexNum(7,12)] = Heke * (gamma_3 * sigma_fun(v2 - v3) + gamma_3T * (Q + coupled_input)) - ke_2 * y4 - keke * v4;
        deriv[indexNum(8,12)] = y5;
        // inh input to the inhibitory interneurons
        deriv[indexNum(9,12)] = Hiki * (gamma_5 * sigma_fun(v4 - v5)) - ki_2 * y5 - keke * v5;
        // aux variables (the sum gathering the postsynaptic inh & exc potentials)
        // pyramidal cells
        deriv[indexNum(10,12)] = y2 - y3;
        // inhibitory cells
        deriv[indexNum(11,12)] = y4 - y5;

    }

    """


class CL_Linear( Linear, CLModel ):
    n_states = 1
    _opencl_ordered_params = "gamma".split()
    _opencl_program_source_file = "Linear.cl"
    _opencl_program_source = """
    __kernel void dfun(__global float *state, __global float *coupling,
                       __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        // this is boilerplate and could be generated
        float x = state[i];
        //gamma
        float c = coupling[i];
        float gamma = param[i];
        deriv[i] = gamma*x + x;
    }
    """

class CL_Hopfield( Hopfield, CLModel ):
    n_states = 2
    _opencl_ordered_params = "taux tauT dynamic".split()
    _opencl_program_source_file = "Hopfield.cl"
    _opencl_program_source = """
    #define indexNum(cur,totalN) cur*n+i
    __kernel void dfun(__global float *state, __global float *coupling,
                   __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        // this is boilerplate and could be generated
        float x = state[indexNum(0,2)], theta=param[indexNum(1,2)];
        float c = coupling[i];
        // taux tauT dynamic
        float taux = param[indexNum(0,3)],tauT = param[indexNum(1,3)], dynamic = param[indexNum(2,3)];


        deriv[indexNum(0,2)] = (-x + c) / taux;
        deriv[indexNum(1,2)] = (-theta + c) /tauT;
    }"""

class CL_ReducedSetFitzHughNagumo( ReducedSetFitzHughNagumo, CLModel):
    n_states = 4
    """tau b K11 K12 K21 are nomral variables, f_i IE_i II_i m_i n_i are (1,3) vector, Aik Bik Cik are (3,3) matrices """
    _opencl_ordered_params = "tau b K11 K12 K21 e_i f_i IE_i II_i m_i n_i  Aik Bik Cik ".split()
    _opencl_program_source_file = "ReducedSetFitzHughNagumo.cl"
    _opencl_program_source = """//ReducedSetFitzHugnNagumo
    #define indexNum(cur,totalN) cur*n+i
    #define getFloat3Vector(ptr) (float3)(*ptr,*(ptr+1),*(ptr+2))
    __kernel void dfun(__global float *state, __global float *coupling,
                   __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        float3 xi = vload3(i,state), eta=vload3(i,state+3*n),
        alpha = vload3(i,state+6*n), beta = vload3(i,state+9*n);
        // tau b K11 K12 K21    e_i f_i IE_i II_i m_i n_i (1*3 vector)   Aik Bik Cik (3*3 matrix) params length = 50


        float c_0 = coupling[i];

        float tau = param[i];
        float b = param[n+i];
        float K11 = param[2*n+i];
        float K12 = param[3*n+i];
        float K21 = param[4*n+i];

        float3 e_i = vload3(i,param + 5*n);
        float3 f_i = vload3(i,param + 8*n);
        float3 IE_i = vload3(i,param + 11*n);
        float3 II_i = vload3(i,param +14*n);
        float3 m_i = vload3(i,param+ 17*n);
        float3 n_i = vload3(i,param + 20*n);

        float3 Aik_0 = vload3(i,param + 23*n);
        float3 Aik_1 = vload3(i,param + 23*n+3);
        float3 Aik_2 = vload3(i,param + 23*n+6);

        float3 Bik_0 = vload3(i,param + 32*n);
        float3 Bik_1 = vload3(i,param + 32*n+3);
        float3 Bik_2 = vload3(i,param + 32*n+6);

        float3 Cik_0 = vload3(i,param + 41*n);
        float3 Cik_1 = vload3(i,param + 41*n+3);
        float3 Cik_2 = vload3(i,param + 41*n+6);


        float local_coupling = 0;
        float3 deriv1 = tau * (xi - e_i * pow(xi,3)-eta)+
                                 K11 * ( (float3)(dot(xi,Aik_0),dot(xi,Aik_1),dot(xi,Aik_2)) - xi)-
                                 K12*( (float)(dot(alpha,Bik_0),dot(alpha,Bik_1),dot(alpha,Bik_2) )-xi)+
                                 tau * (IE_i+c_0+local_coupling*xi);

        float3 deriv2 = (xi - b*eta + m_i)/tau;
        float3 deriv3 = tau * (alpha-f_i*pow(alpha,3)/3 - beta)+
                                K21 * ((float3)(dot(xi,Cik_0),dot(xi,Cik_1),dot(xi,Cik_2)) -alpha)+
                                tau * (II_i+c_0+local_coupling*xi);
        float3 deriv4 = (alpha-b*beta+n_i)/tau;

        vstore3(deriv1,i,deriv);
        vstore3(deriv2,i,deriv+3*n);
        vstore3(deriv3,i,deriv+6*n);
        vstore3(deriv4,i,deriv+9*n);

    }

    """




class CL_ReducedSetHindmarshRose( ReducedSetHindmarshRose , CLModel):
    n_states = 6
    _opencl_ordered_params = "r s K11 K12 K21 a_i b_i c_i d_i e_i f_i h_i p_i IE_i II_i m_i n_i   A_ik B_ik C_ik ".split()
    _opencl_program_source_file = "ReducedSetHindmarshRose.cl"
    _opencl_program_source = """

    __kernel void dfun(__global float *state, __global float *coupling,
                   __global float *params, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        float3 xi = vload3(i,state), eta=vload3(i,state+3*n),
        tau = vload3(i,state+6*n), alpha = vload3(i,state+9*n),
        beta = vload3(i,state+12*n), gamma = vload3(i,state+15*n);

        // "r s K11 K12 K21   a_i b_i c_i d_i e_i f_i h_i p_i IE_i II_i m_i n_i   A_ik B_ik C_ik params length = 68

        float c_0 = coupling[i];

        float r = params[i];
        float s = params[n+i];
        float K11 = params[2*n+i];
        float K12 = params[3*n+i];
        float K21 = params[4*n+i];

        float3 a_i = vload3(i,params + 5*n);
        float3 b_i = vload3(i,params + 8*n);
        float3 c_i = vload3(i,params + 11*n);
        float3 d_i = vload3(i,params + 14*n);
        float3 e_i = vload3(i,params + 17*n);
        float3 f_i = vload3(i,params + 20*n);
        float3 h_i = vload3(i,params + 23*n);
        float3 p_i = vload3(i,params + 26*n);
        float3 IE_i = vload3(i,params + 29*n);
        float3 II_i = vload3(i,params + 32*n);
        float3 m_i = vload3(i,params + 35*n);
        float3 n_i = vload3(i,params + 38*n);



        float3 Aik_0 = vload3(i,params + 41*n);
        float3 Aik_1 = vload3(i,params + 41*n+3);
        float3 Aik_2 = vload3(i,params + 41*n+6);

        float3 Bik_0 = vload3(i,params + 50*n);
        float3 Bik_1 = vload3(i,params + 50*n+3);
        float3 Bik_2 = vload3(i,params + 50*n+6);

        float3 Cik_0 = vload3(i,params + 59*n);
        float3 Cik_1 = vload3(i,params + 59*n+3);
        float3 Cik_2 = vload3(i,params + 59*n+6);

        float local_coupling = 0;
        //TODO Dot Product
        float3 deriv1 = (eta - a_i * pow(xi , 3) + b_i * pow(xi, 2) - tau +
                                K11 * ( (float3)(dot(xi , Aik_0),dot(xi , Aik_1),dot(xi , Aik_2)) - xi) -
                                K12 * ( (float3)(dot(alpha , Bik_0),dot(alpha , Bik_1),dot(alpha , Bik_2) )- xi) +
                                IE_i + c_0 + local_coupling * xi);

        float3 deriv2 = c_i - d_i * pow(xi, 2) - eta;

        float3 deriv3 = r * s * xi - r * tau - m_i;

        float3 deriv4 = beta - e_i * pow(alpha, 3) + f_i * pow(alpha , 2) - gamma +
                                K21 * ( (float3)(dot(xi , Cik_0),dot(xi , Cik_1), dot(xi , Cik_2)) - alpha) +
                                II_i + c_0 + local_coupling * xi;

        float3 deriv5 = h_i - p_i * pow(alpha, 2) - beta;

        float3 deriv6 = r * s * alpha - r * gamma - n_i;


        vstore3(deriv1,i,deriv);
        vstore3(deriv2,i,deriv+3*n);
        vstore3(deriv3,i,deriv+6*n);
        vstore3(deriv4,i,deriv+9*n);
        vstore3(deriv5,i,deriv+12*n);
        vstore3(deriv6,i,deriv+15*n);



    }
    """


class CL_Epileptor( Epileptor, CLModel):
    n_states = 6
    _opencl_ordered_params = "x0 Iext Iext2 a b slope tt Kvf c d r Ks Kf aa tau".split()
    _opencl_program_source_file = "Epileptor.cl"
    _opencl_program_source = """
   #define indexNum(cur,totalN) cur*n+i
__kernel void dfun(__global float *state, __global float *coupling,
                   __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        float y0 = state[indexNum(0,6)], y1=param[indexNum(1,6)],
        y2 = state[indexNum(2,6)], y3 = param[indexNum(4,6)],
        y4 = state[indexNum(4,6)], y5 = param[indexNum(5,6)];

        // x0 Iext Iext2 a b slope tt Kvf c d r Ks Kf aa tau ydot

        float c_pop1 = coupling[indexNum(0,2)];
        float c_pop2 = coupling[indexNum(1,2)];

        float x0 = param[indexNum(0,15)];
        float Iext = param[indexNum(1,15)];
        float Iext2 = param[indexNum(2,15)];
        float a = param[indexNum(3,15)];
        float b = param[indexNum(4,15)];
        float slope = param[indexNum(5,15)];
        float tt = param[indexNum(6,15)];
        float Kvf = param[indexNum(7,15)];
        float c = param[indexNum(8,15)];
        float d = param[indexNum(9,15)];
        float r = param[indexNum(10,15)];
        float Ks = param[indexNum(11,15)];
        float Kf = param[indexNum(12,15)];
        float aa = param[indexNum(13,15)];
        float tau = param[indexNum(14,15)];


        float temp_ydot0,temp_ydot2,temp_ydot4;
        if(y0 < 0.0){
            temp_ydot0 = -a*y0*y0+b*y0;
        }else{
            temp_ydot0 = slope-y3+0.6*(y2-4.0)*(y2-4.0);
        }

        deriv[0] = tt * (y1-y2+Iext + Kvf + c_pop1+temp_ydot0*y0);
        deriv[1] = tt * (c - d*y0*y0 - y1);

        if( y2 < 0.0){
            temp_ydot2 = -0.1*pow(y2,7);
        }else{
            temp_ydot2 = 0.0;
        }
        deriv[2] = tt* (r * (4*(y0-x0)-y2+temp_ydot2+Ks*c_pop1));

        deriv[3] = tt * (-y4 + y3 - pow(y3,3) + Iext2 + 2 * y5 - 0.3 * (y2 - 3.5) + Kf * c_pop2);

        if(y3<-0.25){
            temp_ydot4 = 0.0;
        }else{
            temp_ydot4 = aa*(y3+0.25);
        }
        deriv[4] = tt * ((-y4 + temp_ydot4) / tau);
        deriv[5] = tt * (-0.01 * (y5 - 0.1 * y0));


    }
    """

class CL_Later_Breakspear( LarterBreakspear, CLModel):
    n_states = 3
    _opencl_ordered_params = " gCa gK gL phi gNa TK TCa TNa VCa VK VL VNa d_K tau_K d_Na d_Ca aei aie b C ane ani aee Iext rNMDA VT d_V ZT d_Z QV_max QZ_max t_scale".split()
    _opencl_program_source_file = "Larter_breakspear.cl"
    _opencl_program_source = """
    #define indexNum(cur,totalN) cur*n+i
__kernel void dfun(__global float *state, __global float *coupling,
                   __global float *param, __global float *deriv)
    {
    int i = get_global_id(0), n = get_global_size(0);
    float V = state[indexNum(0,3)], W = param[indexNum(1,3)],Z = param[indexNum(2,3)];

    float c_0 = coupling[i];

    float gCa = param[indexNum(0,32)];
    float gK = param[indexNum(1,32)];
    float gL = param[indexNum(2,32)];
    float phi = param[indexNum(3,32)];
    float gNa = param[indexNum(4,32)];
    float TK = param[indexNum(5,32)];
    float TCa = param[indexNum(6,32)];
    float TNa = param[indexNum(7,32)];
    float VCa = param[indexNum(8,32)];
    float VK = param[indexNum(9,32)];
    float VL = param[indexNum(10,32)];
    float VNa = param[indexNum(11,32)];
    float d_K = param[indexNum(12,32)];
    float tau_K = param[indexNum(13,32)];
    float d_Na = param[indexNum(14,32)];
    float d_Ca = param[indexNum(15,32)];
    float aei = param[indexNum(16,32)];
    float aie = param[indexNum(17,32)];
    float b = param[indexNum(18,32)];
    float C = param[indexNum(19,32)];
    float ane = param[indexNum(20,32)];
    float ani = param[indexNum(21,32)];
    float aee = param[indexNum(22,32)];
    float Iext = param[indexNum(23,32)];
    float rNMDA = param[indexNum(24,32)];
    float VT = param[indexNum(25,32)];
    float d_V = param[indexNum(26,32)];
    float ZT = param[indexNum(27,32)];
    float d_Z = param[indexNum(28,32)];
    float QV_max = param[indexNum(29,32)];
    float QZ_max = param[indexNum(30,32)];
    float t_scale = param[indexNum(31,32)];

    float local_coupling = 1;
    float m_Ca = 0.5 * (1 + tan((V - TCa) / d_Ca));
    float m_Na = 0.5 * (1 + tan((V - TNa) / d_Na));
    float m_K  = 0.5 * (1 + tan((V - TK )  / d_K));
    // voltage to firing rate
    float QV    = 0.5 * QV_max * (1 + tan((V - VT) / d_V));
    float QZ    = 0.5 * QZ_max * (1 + tan((Z - ZT) / d_Z));
    float lc_0  = local_coupling * QV;
    deriv[0] = t_scale * (- (gCa + (1.0 - C) * (rNMDA * aee) * (QV + lc_0)+ C * rNMDA * aee * c_0) * m_Ca * (V - VCa)
                     - gK * W * (V - VK)
                     - gL * (V - VL)
                     - (gNa * m_Na + (1.0 - C) * aee * (QV  + lc_0) + C * aee * c_0) * (V - VNa)
                     - aie * Z * QZ
                     + ane * Iext);
    deriv[1] = t_scale * phi * (m_K - W) / tau_K;
    deriv[2] = t_scale * b * (ani * Iext + aei * V * QV);

    }
   """

class CL_WilsonCowan ( WilsonCowan, CLModel ):
    n_states = 2
    _opencl_ordered_params = " c_ee c_ei c_ie c_ii tau_e tau_i a_e b_e c_e a_i b_i c_i r_e r_i k_e k_i P Q theta_e theta_i alpha_e alpha_i".split()
    _opencl_program_source_file = "Wilson_Cowan.cl"
    _opencl_program_source = """
    #define indexNum(cur,totalN) cur*n+i
    __kernel void dfun(__global float *state, __global float *coupling,
                   __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        float E = state[indexNum(0,2)], I = state[indexNum(1,2)];
        //c_ee c_ei c_ie c_ii tau_e tau_i a_e b_e c_e a_i b_i c_i r_e r_i k_e k_i P Q theta_e theta_i alpha_e alpha_i

        float c_0 = coupling[i];

        float c_ee = param[indexNum(0,22)];
        float c_ei = param[indexNum(1,22)];
        float c_ie = param[indexNum(2,22)];
        float c_ii = param[indexNum(3,22)];
        float tau_e = param[indexNum(4,22)];
        float tau_i = param[indexNum(5,22)];
        float a_e = param[indexNum(6,22)];
        float b_e = param[indexNum(7,22)];
        float c_e = param[indexNum(8,22)];
        float a_i = param[indexNum(9,22)];
        float b_i = param[indexNum(10,22)];
        float c_i = param[indexNum(11,22)];
        float r_e = param[indexNum(12,22)];
        float r_i = param[indexNum(13,22)];
        float k_e = param[indexNum(14,22)];
        float k_i = param[indexNum(15,22)];
        float P = param[indexNum(16,22)];
        float Q = param[indexNum(17,22)];
        float theta_e = param[indexNum(18,22)];
        float theta_i = param[indexNum(19,22)];
        float alpha_e = param[indexNum(20,22)];
        float alpha_i = param[indexNum(21,22)];

        //TODO: dummy local_coupling
        float local_coupling = 1;
        float lc_0 = local_coupling * E;
        float lc_1 = local_coupling * I;

        float x_e = alpha_e * (c_ee * E - c_ei * I + P  - theta_e +  c_0 + lc_0 + lc_1);
        float x_i = alpha_i * (c_ie * E - c_ii * I + Q  - theta_i + lc_0 + lc_1);

        float s_e = c_e / (1.0 + exp(-a_e * (x_e - b_e)));
        float s_i = c_i / (1.0 + exp(-a_i * (x_i - b_i)));

        deriv[indexNum(0,2)] = (-E + (k_e - r_e * E) * s_e) / tau_e;
        deriv[indexNum(1,2)] = (-I + (k_i - r_i * I) * s_i) / tau_i;



    }"""
