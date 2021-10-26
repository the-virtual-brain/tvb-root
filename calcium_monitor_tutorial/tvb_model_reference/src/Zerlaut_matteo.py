#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:32:01 2021

@author: nuuria
"""

from tvb_model_reference.src import Zerlaut
from tvb.simulator.models.base import numpy
from numba import jit

class Zerlaut_adaptation_first_order(Zerlaut.Zerlaut_adaptation_first_order):

    @staticmethod
    @jit(nopython=True,cache=True)
    def get_fluct_regime_vars(Fe, Fi, Fe_ext, Fi_ext, W, Q_e, tau_e, E_e, Q_i, tau_i, E_i, g_L, C_m, E_L, N_tot, p_connect_e,p_connect_i, g, K_ext_e, K_ext_i):
        """
        Compute the mean characteristic of neurons.
        Inspired from the next repository :
        https://github.com/yzerlaut/notebook_papers/tree/master/modeling_mesoscopic_dynamics
        :param Fe: firing rate of excitatory population
        :param Fi: firing rate of inhibitory population
        :param Fe_ext: external excitatory input
        :param Fi_ext: external inhibitory input
        :param W: level of adaptation
        :param Q_e: excitatory quantal conductance
        :param tau_e: excitatory decay
        :param E_e: excitatory reversal potential
        :param Q_i: inhibitory quantal conductance
        :param tau_i: inhibitory decay
        :param E_i: inhibitory reversal potential
        :param E_L: leakage reversal voltage of neurons
        :param g_L: leak conductance
        :param C_m: membrane capacitance
        :param E_L: leak reversal potential
        :param N_tot: cell number
        :param p_connect_e: connectivity probability of excitatory neurons
        :param p_connect_i: connectivity probability of inhibitory neurons
        :param g: fraction of inhibitory cells
        :return: mean and variance of membrane voltage of neurons and autocorrelation time constant
        """
        # firing rate
        fe = Fe*(1.-g)*p_connect_e*N_tot + Fe_ext*K_ext_e
        fi = Fi*g*p_connect_i*N_tot + Fi_ext*K_ext_i

        # conductance fluctuation and effective membrane time constant
        mu_Ge, mu_Gi = Q_e*tau_e*fe, Q_i*tau_i*fi  # Eqns 5 from [MV_2018]
        mu_G = g_L+mu_Ge+mu_Gi  # Eqns 6 from [MV_2018]
        T_m = C_m/mu_G # Eqns 6 from [MV_2018]

        # membrane potential
        mu_V = (mu_Ge*E_e+mu_Gi*E_i+g_L*E_L-W)/mu_G  # Eqns 7 from [MV_2018]
        # post-synaptic membrane potential event s around muV
        U_e, U_i = Q_e/mu_G*(E_e-mu_V), Q_i/mu_G*(E_i-mu_V)
        # Standard deviation of the fluctuations
        # Eqns 8 from [MV_2018]
        sigma_V = numpy.sqrt(fe*(U_e*tau_e)**2/(2.*(tau_e+T_m))+fi*(U_i*tau_i)**2/(2.*(tau_i+T_m)))
        fe, fi = fe+1e-9, fi+1e-9
        # Autocorrelation-time of the fluctuations Eqns 9 from [MV_2018]
        T_V_numerator = (fe*(U_e*tau_e)**2 + fi*(U_i*tau_i)**2)
        T_V_denominator = (fe*(U_e*tau_e)**2/(tau_e+T_m) + fi*(U_i*tau_i)**2/(tau_i+T_m))
        T_V = T_V_numerator/T_V_denominator
        return mu_V, sigma_V+1e-12, T_V

class Zerlaut_adaptation_second_order(Zerlaut.Zerlaut_adaptation_second_order):
    # same then previous function
    @staticmethod
    @jit(nopython=True,cache=True)
    def get_fluct_regime_vars(Fe, Fi, Fe_ext, Fi_ext, W, Q_e, tau_e, E_e, Q_i, tau_i, E_i, g_L, C_m, E_L, N_tot, p_connect_e,p_connect_i, g, K_ext_e, K_ext_i):
        """
        Compute the mean characteristic of neurons.
        Inspired from the next repository :
        https://github.com/yzerlaut/notebook_papers/tree/master/modeling_mesoscopic_dynamics
        :param Fe: firing rate of excitatory population
        :param Fi: firing rate of inhibitory population
        :param Fe_ext: external excitatory input
        :param Fi_ext: external inhibitory input
        :param W: level of adaptation
        :param Q_e: excitatory quantal conductance
        :param tau_e: excitatory decay
        :param E_e: excitatory reversal potential
        :param Q_i: inhibitory quantal conductance
        :param tau_i: inhibitory decay
        :param E_i: inhibitory reversal potential
        :param E_L: leakage reversal voltage of neurons
        :param g_L: leak conductance
        :param C_m: membrane capacitance
        :param E_L: leak reversal potential
        :param N_tot: cell number
        :param p_connect_e: connectivity probability of excitatory neurons
        :param p_connect_i: connectivity probability of inhibitory neurons
        :param g: fraction of inhibitory cells
        :return: mean and variance of membrane voltage of neurons and autocorrelation time constant
        """
        # firing rate
        fe = Fe*(1.-g)*p_connect_e*N_tot + Fe_ext*K_ext_e
        fi = Fi*g*p_connect_i*N_tot + Fi_ext*K_ext_i

        # conductance fluctuation and effective membrane time constant
        mu_Ge, mu_Gi = Q_e*tau_e*fe, Q_i*tau_i*fi  # Eqns 5 from [MV_2018]
        mu_G = g_L+mu_Ge+mu_Gi  # Eqns 6 from [MV_2018]
        T_m = C_m/mu_G # Eqns 6 from [MV_2018]

        # membrane potential
        mu_V = (mu_Ge*E_e+mu_Gi*E_i+g_L*E_L-W)/mu_G  # Eqns 7 from [MV_2018]
        # post-synaptic membrane potential event s around muV
        U_e, U_i = Q_e/mu_G*(E_e-mu_V), Q_i/mu_G*(E_i-mu_V)
        # Standard deviation of the fluctuations
        # Eqns 8 from [MV_2018]
        sigma_V = numpy.sqrt(fe*(U_e*tau_e)**2/(2.*(tau_e+T_m))+fi*(U_i*tau_i)**2/(2.*(tau_i+T_m)))
        fe, fi = fe+1e-9, fi+1e-9
        # Autocorrelation-time of the fluctuations Eqns 9 from [MV_2018]
        T_V_numerator = (fe*(U_e*tau_e)**2 + fi*(U_i*tau_i)**2)
        T_V_denominator = (fe*(U_e*tau_e)**2/(tau_e+T_m) + fi*(U_i*tau_i)**2/(tau_i+T_m))
        T_V = T_V_numerator/T_V_denominator
        return mu_V, sigma_V+1e-12, T_V