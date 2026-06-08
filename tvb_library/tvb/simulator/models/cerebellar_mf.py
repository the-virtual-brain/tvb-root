# -*- coding: utf-8 -*-
"""
Cerebellar Mean-Field (CRBL MF) model for the TVB hybrid simulator.

This model implements the cerebellar microcircuit mean-field dynamics from
Lorenzi et al. (2023), with four neural populations:

- Granule cells (GrC)  — excitatory
- Golgi cells (GoC)    — inhibitory
- Molecular layer interneurons (MLI) — inhibitory
- Purkinje cells (PC)  — inhibitory

Plus a 5th state variable for Ornstein-Uhlenbeck noise.

**Coupling design**: The model has 2 coupling variables:

``cvar = [0, 1]``

- cvar[0] = "mossy"    — receives cortical mossy-fiber input + DCN feedback
- cvar[1] = "parallel" — receives CRBL→CRBL parallel-fiber input

In the dfun, ``coupling[0,:]`` is the mossy signal and ``coupling[1,:]`` is
the parallel signal.  This cleanly decomposes the index-mask-based routing
used in the monolithic `crbl_cortical_first_ord` class.

References
----------
.. [Lorenzi_2023] Lorenzi, R.M., Geminiani, A., Zerbi, A. et al.
   Dirichlet-Escalon Fractal Network model of cerebellar cortex.
   Nat Commun 14, 6826 (2023).

.. [MV_2018] Montbrió, E., Pazó, D. & Roxin, A.
   Macroscopic Description for Networks of Spiking Neurons.
   Phys Rev X 5, 021028 (2015).

State variables
---------------
GrC   : Granule cell firing rate [Hz]
GoC   : Golgi cell firing rate [Hz]
MLI   : Molecular layer interneuron firing rate [Hz]
PC    : Purkinje cell firing rate [Hz]
noise : Ornstein-Uhlenbeck noise process
"""

import numpy
from scipy.special import erfc as _erfc
from tvb.basic.neotraits.api import NArray, Final, List, Range
from tvb.simulator.models.base import ModelNumbaDfun


class CerebellarMF(ModelNumbaDfun):
    r"""
    Cerebellar microcircuit mean-field model.

    Four neural populations (GrC, GoC, MLI, PC) modelled as firing-rate
    mean-field approximations, plus an OU noise variable.

    The model exposes two coupling variables:

    - **mossy** (cvar 0): receives input from cortex / DCN via mossy fibers
    - **parallel** (cvar 1): receives input from other cerebellar regions
      via parallel fibers

    In the dfun:

    * GrC receives mossy input only
    * GoC receives mossy + parallel input
    * MLI and PC receive parallel input only
    * OU noise is added to mossy input for GrC and GoC
    """

    # -----------------------------------------------------------------------
    # Membrane / passive parameters
    # -----------------------------------------------------------------------
    g_L_grc = NArray(
        label=":math:`g_{L}^{GrC}`",
        default=numpy.array([0.29]),
        domain=Range(lo=0.25, hi=0.35, step=0.01),
        doc="Granule cell leak conductance [nS]")

    g_L_goc = NArray(
        label=":math:`g_{L}^{GoC}`",
        default=numpy.array([3.30]),
        domain=Range(lo=3.25, hi=3.35, step=0.01),
        doc="Golgi cell leak conductance [nS]")

    g_L_mli = NArray(
        label=":math:`g_{L}^{MLI}`",
        default=numpy.array([1.60]),
        domain=Range(lo=1.55, hi=1.65, step=0.01),
        doc="MLI leak conductance [nS]")

    g_L_pc = NArray(
        label=":math:`g_{L}^{PC}`",
        default=numpy.array([7.10]),
        domain=Range(lo=7.05, hi=7.15, step=0.01),
        doc="Purkinje cell leak conductance [nS]")

    E_L_grc = NArray(
        label=":math:`E_{L}^{GrC}`",
        default=numpy.array([-62.0]),
        domain=Range(lo=-62.1, hi=-61.9, step=0.1),
        doc="GrC leak reversal potential [mV]")

    E_L_goc = NArray(
        label=":math:`E_{L}^{GoC}`",
        default=numpy.array([-62.0]),
        domain=Range(lo=-73.0, hi=-51.0, step=0.1),
        doc="GoC leak reversal potential [mV]")

    E_L_mli = NArray(
        label=":math:`E_{L}^{MLI}`",
        default=numpy.array([-68.0]),
        domain=Range(lo=-68.01, hi=-67.9, step=0.1),
        doc="MLI leak reversal potential [mV]")

    E_L_pc = NArray(
        label=":math:`E_{L}^{PC}`",
        default=numpy.array([-59.0]),
        domain=Range(lo=-65.0, hi=-53.0, step=0.1),
        doc="PC leak reversal potential [mV]")

    C_m_grc = NArray(
        label=":math:`C_{m}^{GrC}`",
        default=numpy.array([7.0]),
        domain=Range(lo=5.0, hi=7.5, step=1.0),
        doc="GrC membrane capacitance [pF]")

    C_m_goc = NArray(
        label=":math:`C_{m}^{GoC}`",
        default=numpy.array([145.0]),
        domain=Range(lo=72.0, hi=218.0, step=10.0),
        doc="GoC membrane capacitance [pF]")

    C_m_mli = NArray(
        label=":math:`C_{m}^{MLI}`",
        default=numpy.array([14.6]),
        domain=Range(lo=14.5, hi=14.7, step=0.1),
        doc="MLI membrane capacitance [pF]")

    C_m_pc = NArray(
        label=":math:`C_{m}^{PC}`",
        default=numpy.array([334.0]),
        domain=Range(lo=228.0, hi=440.0, step=10.0),
        doc="PC membrane capacitance [pF]")

    E_e = NArray(
        label=":math:`E_e`",
        default=numpy.array([0.0]),
        domain=Range(lo=-20.0, hi=20.0, step=0.01),
        doc="Excitatory reversal potential [mV]")

    E_i = NArray(
        label=":math:`E_i`",
        default=numpy.array([-80.0]),
        domain=Range(lo=-100.0, hi=-60.0, step=1.0),
        doc="Inhibitory reversal potential [mV]")

    # -----------------------------------------------------------------------
    # Synaptic quantal conductances
    # -----------------------------------------------------------------------
    Q_mf_grc = NArray(
        label=":math:`Q_{mf\\to GrC}`",
        default=numpy.array([0.230]),
        domain=Range(lo=0.225, hi=0.235, step=0.001),
        doc="Mossy→GrC quantal conductance [nS]")

    Q_mf_goc = NArray(
        label=":math:`Q_{mf\\to GoC}`",
        default=numpy.array([0.240]),
        domain=Range(lo=0.235, hi=0.245, step=0.001),
        doc="Mossy→GoC quantal conductance [nS]")

    Q_grc_goc = NArray(
        label=":math:`Q_{GrC\\to GoC}`",
        default=numpy.array([0.437]),
        domain=Range(lo=0.432, hi=0.542, step=0.001),
        doc="GrC→GoC quantal conductance [nS]")

    Q_grc_mli = NArray(
        label=":math:`Q_{GrC\\to MLI}`",
        default=numpy.array([0.154]),
        domain=Range(lo=0.149, hi=0.159, step=0.001),
        doc="GrC→MLI quantal conductance [nS]")

    Q_grc_pc = NArray(
        label=":math:`Q_{GrC\\to PC}`",
        default=numpy.array([1.126]),
        domain=Range(lo=1.120, hi=1.131, step=0.001),
        doc="GrC→PC quantal conductance [nS]")

    Q_goc_grc = NArray(
        label=":math:`Q_{GoC\\to GrC}`",
        default=numpy.array([0.336]),
        domain=Range(lo=0.330, hi=0.341, step=0.001),
        doc="GoC→GrC quantal conductance [nS]")

    Q_goc_goc = NArray(
        label=":math:`Q_{GoC\\to GoC}`",
        default=numpy.array([1.120]),
        domain=Range(lo=1.115, hi=1.130, step=0.001),
        doc="GoC→GoC quantal conductance [nS]")

    Q_mli_mli = NArray(
        label=":math:`Q_{MLI\\to MLI}`",
        default=numpy.array([0.532]),
        domain=Range(lo=0.527, hi=0.537, step=0.001),
        doc="MLI→MLI quantal conductance [nS]")

    Q_mli_pc = NArray(
        label=":math:`Q_{MLI\\to PC}`",
        default=numpy.array([1.244]),
        domain=Range(lo=1.240, hi=1.250, step=0.001),
        doc="MLI→PC quantal conductance [nS]")

    # -----------------------------------------------------------------------
    # Synaptic time constants
    # -----------------------------------------------------------------------
    tau_mf_grc = NArray(
        label=":math:`\\tau_{mf\\to GrC}`",
        default=numpy.array([1.9]),
        domain=Range(lo=1.5, hi=2.3, step=0.1),
        doc="Mossy→GrC decay time [ms]")

    tau_mf_goc = NArray(
        label=":math:`\\tau_{mf\\to GoC}`",
        default=numpy.array([5.0]),
        domain=Range(lo=4.5, hi=5.5, step=0.1),
        doc="Mossy→GoC decay time [ms]")

    tau_grc_goc = NArray(
        label=":math:`\\tau_{GrC\\to GoC}`",
        default=numpy.array([1.25]),
        domain=Range(lo=1.05, hi=1.45, step=0.1),
        doc="GrC→GoC decay time [ms]")

    tau_grc_mli = NArray(
        label=":math:`\\tau_{GrC\\to MLI}`",
        default=numpy.array([0.64]),
        domain=Range(lo=0.44, hi=0.84, step=0.1),
        doc="GrC→MLI decay time [ms]")

    tau_grc_pc = NArray(
        label=":math:`\\tau_{GrC\\to PC}`",
        default=numpy.array([1.1]),
        domain=Range(lo=1.0, hi=1.2, step=0.1),
        doc="GrC→PC decay time [ms]")

    tau_goc_grc = NArray(
        label=":math:`\\tau_{GoC\\to GrC}`",
        default=numpy.array([4.5]),
        domain=Range(lo=4.0, hi=5.0, step=0.1),
        doc="GoC→GrC decay time [ms]")

    tau_goc_goc = NArray(
        label=":math:`\\tau_{GoC\\to GoC}`",
        default=numpy.array([5.0]),
        domain=Range(lo=4.5, hi=5.5, step=0.1),
        doc="GoC→GoC decay time [ms]")

    tau_mli_mli = NArray(
        label=":math:`\\tau_{MLI\\to MLI}`",
        default=numpy.array([2.0]),
        domain=Range(lo=1.5, hi=2.5, step=0.1),
        doc="MLI→MLI decay time [ms]")

    tau_mli_pc = NArray(
        label=":math:`\\tau_{MLI\\to PC}`",
        default=numpy.array([2.8]),
        domain=Range(lo=2.3, hi=3.2, step=0.1),
        doc="MLI→PC decay time [ms]")

    # -----------------------------------------------------------------------
    # Synaptic convergence
    # -----------------------------------------------------------------------
    K_mossy_grc = NArray(
        label=":math:`K_{mossy\\to GrC}`",
        default=numpy.array([4.0]),
        domain=Range(lo=0.0, hi=10.0, step=1.0),
        doc="Mossy→GrC convergence [-]")

    K_mossy_goc = NArray(
        label=":math:`K_{mossy\\to GoC}`",
        default=numpy.array([35.0]),
        domain=Range(lo=15.0, hi=55.0, step=10.0),
        doc="Mossy→GoC convergence [-]")

    K_grc_goc = NArray(
        label=":math:`K_{GrC\\to GoC}`",
        default=numpy.array([501.98]),
        domain=Range(lo=451.98, hi=551.0, step=10.0),
        doc="GrC→GoC convergence [-]")

    K_grc_mli = NArray(
        label=":math:`K_{GrC\\to MLI}`",
        default=numpy.array([243.96]),
        domain=Range(lo=193.96, hi=293.96, step=10.0),
        doc="GrC→MLI convergence [-]")

    K_grc_pc = NArray(
        label=":math:`K_{GrC\\to PC}`",
        default=numpy.array([374.50]),
        domain=Range(lo=334.50, hi=404.50, step=10.0),
        doc="GrC→PC convergence [-]")

    K_goc_goc = NArray(
        label=":math:`K_{GoC\\to GoC}`",
        default=numpy.array([16.2]),
        domain=Range(lo=10.2, hi=20.2, step=1.0),
        doc="GoC→GoC convergence [-]")

    K_mli_mli = NArray(
        label=":math:`K_{MLI\\to MLI}`",
        default=numpy.array([14.20]),
        domain=Range(lo=10.20, hi=20.20, step=1.0),
        doc="MLI→MLI convergence [-]")

    K_mli_pc = NArray(
        label=":math:`K_{MLI\\to PC}`",
        default=numpy.array([10.28]),
        domain=Range(lo=5.28, hi=15.28, step=1.0),
        doc="MLI→PC convergence [-]")

    # -----------------------------------------------------------------------
    # Population sizes
    # -----------------------------------------------------------------------
    N_grc = NArray(
        dtype=int,
        label=":math:`N_{GrC}`",
        default=numpy.array([28615]),
        domain=Range(lo=25615, hi=31615, step=1000),
        doc="Granule cell count [-]")

    N_goc = NArray(
        dtype=int,
        label=":math:`N_{GoC}`",
        default=numpy.array([70]),
        domain=Range(lo=10, hi=100, step=10),
        doc="Golgi cell count [-]")

    N_mli = NArray(
        dtype=int,
        label=":math:`N_{MLI}`",
        default=numpy.array([446]),
        domain=Range(lo=146, hi=946, step=100),
        doc="MLI count [-]")

    N_pc = NArray(
        dtype=int,
        label=":math:`N_{PC}`",
        default=numpy.array([99]),
        domain=Range(lo=29, hi=149, step=10),
        doc="Purkinje cell count [-]")

    N_mossy = NArray(
        dtype=int,
        label=":math:`N_{mossy}`",
        default=numpy.array([2336]),
        domain=Range(lo=336, hi=5336, step=1000),
        doc="Mossy fiber count [-]")

    # -----------------------------------------------------------------------
    # Alpha parameters (effective gain in firing rate estimation)
    # -----------------------------------------------------------------------
    alpha_grc = NArray(
        label=":math:`\\alpha_{GrC}`",
        default=numpy.array([2.0]),
        doc="GrC alpha gain [-]")

    alpha_goc = NArray(
        label=":math:`\\alpha_{GoC}`",
        default=numpy.array([1.3]),
        doc="GoC alpha gain [-]")

    alpha_mli = NArray(
        label=":math:`\\alpha_{MLI}`",
        default=numpy.array([1.8]),
        doc="MLI alpha gain [-]")

    alpha_pc = NArray(
        label=":math:`\\alpha_{PC}`",
        default=numpy.array([5.0]),
        doc="PC alpha gain [-]")

    # -----------------------------------------------------------------------
    # Time scale & phenomenological thresholds
    # -----------------------------------------------------------------------
    T = NArray(
        label=":math:`T`",
        default=numpy.array([3.5]),
        domain=Range(lo=3.45, hi=3.55, step=0.01),
        doc="Time scale of network activity [ms]")

    P_grc = NArray(
        label=":math:`P_{GrC}`",
        default=numpy.array([-0.426, 0.007, 0.023, 0.482, 0.216]),
        doc="GrC phenomenological threshold polynomial (order 5)")

    P_goc = NArray(
        label=":math:`P_{GoC}`",
        default=numpy.array([-0.144, 0.003, 0.011, 0.031, 0.011]),
        doc="GoC phenomenological threshold polynomial (order 5)")

    P_mli = NArray(
        label=":math:`P_{MLI}`",
        default=numpy.array([-0.128, -0.001, 0.012, -0.093, -0.063]),
        doc="MLI phenomenological threshold polynomial (order 5)")

    P_pc = NArray(
        label=":math:`P_{PC}`",
        default=numpy.array([-0.080, 0.009, 0.004, 0.006, 0.014]),
        doc="PC phenomenological threshold polynomial (order 5)")

    # -----------------------------------------------------------------------
    # Noise & external input
    # -----------------------------------------------------------------------
    tau_OU = NArray(
        label=":math:`\\tau_{OU}`",
        default=numpy.array([5.0]),
        domain=Range(lo=0.1, hi=10.0, step=0.01),
        doc="OU noise time constant [ms]")

    weight_noise = NArray(
        label=":math:`w_{noise}`",
        default=numpy.array([10.5]),
        domain=Range(lo=0.0, hi=50.0, step=1.0),
        doc="OU noise weight")

    external_input_ex_ex = NArray(
        label=":math:`\\nu_e^{drive}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=0.1, step=0.001),
        doc="External excitatory drive to excitatory population")

    external_input_ex_in = NArray(
        label=":math:`\\nu_i^{drive}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=0.1, step=0.001),
        doc="External inhibitory drive to excitatory population")

    # -----------------------------------------------------------------------
    # Model metadata
    # -----------------------------------------------------------------------
    coupling_terms = Final(
        label="Coupling terms",
        default=["mossy", "parallel"])

    parameter_names = List(
        of=str,
        label="List of parameters for this model",
        default=[
            "g_L_grc", "g_L_goc", "g_L_mli", "g_L_pc",
            "E_L_grc", "E_L_goc", "E_L_mli", "E_L_pc",
            "C_m_grc", "C_m_goc", "C_m_mli", "C_m_pc",
            "E_e", "E_i",
            "Q_mf_grc", "Q_mf_goc", "Q_grc_goc", "Q_grc_mli", "Q_grc_pc",
            "Q_goc_grc", "Q_goc_goc", "Q_mli_mli", "Q_mli_pc",
            "tau_mf_grc", "tau_mf_goc", "tau_grc_goc", "tau_grc_mli", "tau_grc_pc",
            "tau_goc_grc", "tau_goc_goc", "tau_mli_mli", "tau_mli_pc",
            "K_mossy_grc", "K_mossy_goc", "K_grc_goc", "K_grc_mli", "K_grc_pc",
            "K_goc_goc", "K_mli_mli", "K_mli_pc",
            "N_grc", "N_goc", "N_mli", "N_pc", "N_mossy",
            "alpha_grc", "alpha_goc", "alpha_mli", "alpha_pc",
            "T", "P_grc", "P_goc", "P_mli", "P_pc",
            "tau_OU", "weight_noise",
            "external_input_ex_ex", "external_input_ex_in",
        ],
    )

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "GrC":   numpy.array([0.0, 150.0]),
            "GoC":   numpy.array([0.0, 50.0]),
            "MLI":   numpy.array([0.0, 100.0]),
            "PC":    numpy.array([0.0, 150.0]),
            "noise": numpy.array([0.0, 1.0]),
        },
        doc="""Expected dynamic range for each state variable.
        Used for bounding random initial conditions.""")

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "GrC":   numpy.array([0.0, None]),
            "GoC":   numpy.array([0.0, None]),
            "MLI":   numpy.array([0.0, None]),
            "PC":    numpy.array([0.0, None]),
            "noise": numpy.array([None, None]),
        },
        doc="""Firing rates must be non-negative.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("GrC", "GoC", "MLI", "PC", "noise"),
        default=("GrC", "PC"),
        doc="""Default state variables to be monitored.""")

    state_variables = ["GrC", "GoC", "MLI", "PC", "noise"]
    _nvar = 5
    cvar = numpy.array([0, 1], dtype=numpy.int32)

    # -----------------------------------------------------------------------
    # Transfer-function helpers (pure numpy, for readability/debugging)
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_fluct_regime_vars(Fe, Fi, Fe_ext, Fi_ext, XX,
                               Q_e, tau_e, Ee, Q_i, tau_i, Ei,
                               Gl, Cm, El, Ke, Ki,
                               K_ext_e=0.0, tau_ext_e=0.0,
                               Q_ext_e=0.0, K_ext_i=0.0,
                               tau_ext_i=0.0, Q_ext_i=0.0):
        """2D fluctuation-regime variables (GrC, MLI, PC)."""
        fe = (Fe + 1.0e-6) + Fe_ext
        fi = (Fi + 1.0e-6) + Fi_ext

        mu_Ge = Q_e * tau_e * fe * Ke + Q_ext_e * tau_ext_e * Fe_ext * K_ext_e
        mu_Gi = Q_i * tau_i * fi * Ki + Q_ext_i * tau_ext_i * Fi_ext * K_ext_i
        mu_G = Gl + mu_Ge + mu_Gi

        mu_V = (numpy.e * (mu_Ge * Ee + mu_Gi * Ei + Gl * El) - XX) / mu_G
        muGn, Tm = mu_G / Gl, Cm / mu_G

        Ue = Q_e / mu_G * (Ee - mu_V)
        Ui = Q_i / mu_G * (Ei - mu_V)

        sVe = (2.0 * Tm + tau_e) * ((numpy.e * Ue * tau_e) / (2.0 * (tau_e + Tm))) ** 2 * Ke * fe
        sVi = (2.0 * Tm + tau_i) * ((numpy.e * Ui * tau_i) / (2.0 * (tau_i + Tm))) ** 2 * Ki * fi
        sigma_V = numpy.sqrt(sVe + sVi)

        fe, fi = fe + 1.0e-9, fi + 1.0e-9

        Tv_num = (Ke * fe * Ue ** 2 * tau_e ** 2 * numpy.e ** 2
                  + Ki * fi * Ui ** 2 * tau_i ** 2 * numpy.e ** 2)
        Tv = 0.5 * Tv_num / ((sigma_V + 1.0e-20) ** 2)

        T_V = Tv * Gl / Cm
        return mu_V, sigma_V, T_V, muGn

    @staticmethod
    def _get_fluct_regime_vars_goc(Fe, Fi, Fe_ext, XX,
                                   Qe_g, Te_g, Ee, Qi, Ti, Ei,
                                   Gl, Cm, El, Ke_g, Ki,
                                   Ke_ext, Qe_ext, Te_ext, Ki_ext=0.0):
        """3D fluctuation-regime variables (GoC, two excitatory inputs)."""
        fe_g = Fe + 1.0e-6
        fe_m = Fe_ext
        fi = Fi + 1.0e-6

        muGe_g = Qe_g * Ke_g * Te_g * fe_g
        muGe_m = Qe_ext * Ke_ext * Te_ext * fe_m
        muGi = Qi * Ki * Ti * fi
        muG = Gl + muGe_g + muGe_m + muGi

        mu_V = (numpy.e * (muGe_g * Ee + muGe_m * Ee + muGi * Ei + Gl * El) - XX) / muG
        muGn, Tm = muG / Gl, Cm / muG  # normalization

        Ue_g = Qe_g / muG * (Ee - mu_V)
        Ue_m = Qe_ext / muG * (Ee - mu_V)
        Ui = Qi / muG * (Ei - mu_V)

        Tm = Cm / muG
        sVe_g = (2.0 * Tm + Te_g) * ((numpy.e * Ue_g * Te_g) / (2.0 * (Te_g + Tm))) ** 2 * Ke_g * fe_g
        sVe_m = (2.0 * Tm + Te_ext) * ((numpy.e * Ue_m * Te_ext) / (2.0 * (Te_ext + Tm))) ** 2 * Ke_ext * fe_m
        sVi = (2.0 * Tm + Ti) * ((numpy.e * Ui * Ti) / (2.0 * (Ti + Tm))) ** 2 * Ki * fi
        sigma_V = numpy.sqrt(sVe_g + sVe_m + sVi)

        fe_m, fe_g, fi = fe_m + 1.0e-15, fe_g + 1.0e-15, fi + 1.0e-15

        Tv_num = (Ke_g * fe_g * Ue_g ** 2 * Te_g ** 2 * numpy.e ** 2
                  + Ke_ext * fe_m * Ue_m ** 2 * Te_ext ** 2 * numpy.e ** 2
                  + Ki * fi * Ui ** 2 * Ti ** 2 * numpy.e ** 2)
        Tv = 0.5 * Tv_num / ((sigma_V + 1.0e-20) ** 2)
        T_V = Tv * Gl / Cm
        return mu_V, sigma_V, T_V, muGn

    @staticmethod
    def _threshold_func(muV, sigmaV, TvN, muGn, P0, P1, P2, P3, P4):
        """Phenomenological voltage threshold (5th-order polynomial)."""
        muV0, DmuV0 = -60.0, 10.0
        sV0, DsV0 = 4.0, 6.0
        TvN0, DTvN0 = 0.5, 1.0
        V = (muV - muV0) / DmuV0
        S = (sigmaV - sV0) / DsV0
        T = (TvN - TvN0) / DTvN0
        return P0 + P1 * V + P2 * S + P3 * T + P4 * numpy.log(muGn)

    @staticmethod
    def _estimate_firing_rate(muV, sV, TvN, Vthre, Gl, Cm, alpha):
        """Firing rate from erfc (Escalón et al. 2018, Eq. 10)."""
        return 0.5 / TvN * Gl / Cm * _erfc((Vthre - muV) / numpy.sqrt(2.0) / sV) * alpha

    # -----------------------------------------------------------------------
    # Per-population TF wrappers
    # -----------------------------------------------------------------------

    def TF_excitatory_grc(self, fe_ext, fi, fe, fi_ext, W=0):
        return self._TF(fe_ext, fi, fe, fi_ext, W,
                        self.P_grc, self.Q_mf_grc, self.Q_goc_grc,
                        self.tau_mf_grc, self.tau_goc_grc,
                        self.E_e, self.E_i,
                        self.g_L_grc, self.C_m_grc, self.E_L_grc,
                        self.K_mossy_grc, self.K_mossy_goc, self.alpha_grc)

    def TF_inhibitory_goc(self, fe, fi, fe_ext, fi_ext, W=0):
        return self._TF_goc(fe, fi, fe_ext, fi_ext, W,
                            self.P_goc, self.Q_grc_goc, self.Q_goc_goc,
                            self.tau_grc_goc, self.tau_goc_goc,
                            self.E_i, self.E_i,
                            self.g_L_goc, self.C_m_goc, self.E_L_goc,
                            self.K_grc_goc, self.K_goc_goc,
                            self.Q_mf_goc, self.tau_mf_goc,
                            self.K_mossy_goc, self.alpha_goc)

    def TF_inhibitory_mli(self, fe, fi, fe_ext, fi_ext, W=0):
        return self._TF(fe, fi, fe_ext, fi_ext, W,
                        self.P_mli, self.Q_grc_mli, self.Q_mli_mli,
                        self.tau_grc_mli, self.tau_mli_mli,
                        self.E_e, self.E_i,
                        self.g_L_mli, self.C_m_mli, self.E_L_mli,
                        self.K_grc_mli, self.K_mli_mli, self.alpha_mli)

    def TF_inhibitory_pc(self, fe, fi, fe_ext, fi_ext, W=0):
        return self._TF(fe, fi, fe_ext, fi_ext, W,
                        self.P_pc, self.Q_grc_pc, self.Q_mli_pc,
                        self.tau_grc_pc, self.tau_mli_pc,
                        self.E_e, self.E_i,
                        self.g_L_pc, self.C_m_pc, self.E_L_pc,
                        self.K_grc_pc, self.K_mli_pc, self.alpha_pc)

    def _TF(self, Fe, Fi, Fe_ext, Fi_ext, W, P, Q_e, Q_i,
            tau_e, tau_i, E_e, E_i, g_L, C_m, E_L, Ke, Ki, alpha):
        mu_V, sigma_V, T_V, muGn = self._get_fluct_regime_vars(
            Fe, Fi, Fe_ext, Fi_ext, W,
            Q_e, tau_e, E_e, Q_i, tau_i, E_i,
            g_L, C_m, E_L, Ke, Ki)
        V_thre = self._threshold_func(mu_V, sigma_V, T_V, muGn,
                                      P[0], P[1], P[2], P[3], P[4])
        V_thre *= 1.0e3
        return self._estimate_firing_rate(mu_V, sigma_V, T_V, V_thre,
                                          g_L, C_m, alpha)

    def _TF_goc(self, Fe, Fi, Fe_ext, Fi_ext, W, P,
                Qe_gr, Qi, Te_gr, Ti, Ee, Ei, Gl, Cm, El,
                Ke_grc, Ki, Ke_ext, Qe_ext, Te_ext, alpha, Ki_ext=0.0):
        mu_V, sigma_V, T_V, muGn = self._get_fluct_regime_vars_goc(
            Fe, Fi, Fe_ext, W,
            Qe_gr, Te_gr, Ee, Qi, Ti, Ei,
            Gl, Cm, El, Ke_grc, Ki,
            Ke_ext, Qe_ext, Te_ext, Ki_ext)
        V_thre = self._threshold_func(mu_V, sigma_V, T_V, muGn,
                                      P[0], P[1], P[2], P[3], P[4])
        V_thre *= 1.0e3
        return self._estimate_firing_rate(mu_V, sigma_V, T_V, V_thre,
                                          Gl, Cm, alpha)

    # -----------------------------------------------------------------------
    # Main dfun
    # -----------------------------------------------------------------------

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
        """Pure-numpy dfun for debugging."""
        d1 = state_variables[0, :]  # GrC
        d2 = state_variables[1, :]  # GoC
        d3 = state_variables[2, :]  # MLI
        d4 = state_variables[3, :]  # PC
        noise = state_variables[4, :]
        derivative = numpy.empty_like(state_variables)

        c_mossy = coupling[0, :]
        c_parallel = coupling[1, :]

        # Mossy + noise feeds GrC and GoC
        Fe_ext_tod1 = c_mossy + self.weight_noise * noise
        Fe_ext_tod2 = c_mossy + c_parallel + self.weight_noise * noise

        # Parallel only feeds MLI and PC
        Fe_ext_tod3 = c_parallel
        Fe_ext_tod4 = c_parallel

        # Clamp negative inputs
        idx = numpy.where(Fe_ext_tod1 * self.K_mossy_grc < 0)
        Fe_ext_tod1[idx] = 0.0
        idx = numpy.where(Fe_ext_tod2 * self.K_mossy_goc < 0)
        Fe_ext_tod2[idx] = 0.0
        idx = numpy.where(Fe_ext_tod3 * self.K_grc_mli < 0)
        Fe_ext_tod3[idx] = 0.0
        idx = numpy.where(Fe_ext_tod4 * self.K_grc_pc < 0)
        Fe_ext_tod4[idx] = 0.0

        Fi_ext = 0.0

        # GrC
        derivative[0] = (self.TF_excitatory_grc(
            Fe_ext_tod1 + self.external_input_ex_ex,
            d2, 0.0,
            Fi_ext + self.external_input_ex_in, 0.0) - d1) / self.T

        # GoC
        derivative[1] = (self.TF_inhibitory_goc(
            d1, d2,
            Fe_ext_tod2 + self.external_input_ex_ex,
            Fi_ext, 0.0) - d2) / self.T

        # MLI
        derivative[2] = (self.TF_inhibitory_mli(
            d1, d3,
            Fe_ext_tod3, Fi_ext, 0.0) - d3) / self.T

        # PC
        derivative[3] = (self.TF_inhibitory_pc(
            d1, d3,
            Fe_ext_tod4, Fi_ext, 0.0) - d4) / self.T

        # OU noise
        derivative[4] = -noise / self.tau_OU

        return derivative

    def dfun(self, x, c, local_coupling=0.0, **kwargs):
        r"""
        Cerebellar MF dynamics.

        Parameters
        ----------
        x : ndarray, shape (5, nnodes, 1)
            State variables [GrC, GoC, MLI, PC, noise].
        c : ndarray, shape (2, nnodes, 1)
            Coupling variables: c[0]=mossy, c[1]=parallel.
        """
        # Use numpy dfun — the gufunc requires more work to handle
        # the complex TF chains vectorized.  Numba acceleration
        # can be added later as a numba-jitted full dfun.
        return self._numpy_dfun(x, c, local_coupling)
