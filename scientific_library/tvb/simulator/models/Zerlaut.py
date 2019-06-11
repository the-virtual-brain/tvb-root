"""
Mean field model based on Master equation about adaptative exponential leacky integrate and fire neurons population
"""

from tvb.simulator.models.base import Model, LOG, numpy, basic, arrays, core
import scipy.special as sp_spec


class Zerlaut_adaptation_first_order(Model):
    r"""
    **References**:
    .. [ZD_2018]  Zerlaut, Y., Chemla, S., Chavane, F. et al. *Modeling mesoscopic cortical dynamics using a mean-field
    model of conductance-based networks of adaptive
    exponential integrate-and-fire neurons*,
    J Comput Neurosci (2018) 44: 45. https://doi-org.lama.univ-amu.fr/10.1007/s10827-017-0668-2
    .. [MV_2018]  Matteo di Volo, Alberto Romagnoni, Cristiano Capone, Alain Destexhe (2018)
    *Mean-field model for the dynamics of conductance-based networks of excitatory and inhibitory spiking neurons
    with adaptation*, bioRxiv, doi: https://doi.org/10.1101/352393

    Used Eqns 4 from [MV_2018]_ in ``dfun``.

    The default parameters are taken from table 1 of [ZD_2018]_, pag.47 and modify for the adaptation [MV_2018]
    +---------------------------+------------+
    |                 Table 1                |
    +--------------+------------+------------+
    |Parameter     |  Value     | Unit       |
    +==============+============+============+
    |             cellular property          |
    +--------------+------------+------------+
    | g_L          |   10.00    |   nS       |
    +--------------+------------+------------+
    | E_L_e        |  -67.00    |   mV       |
    +--------------+------------+------------+
    | E_L_i        |  -63.00    |   mV       |
    +--------------+------------+------------+
    | C_m          |   150.0    |   pF       |
    +--------------+------------+------------+
    | b            |   60.0     |   nS       |
    +--------------+------------+------------+
    | tau_w        |   500.0    |   ms       |
    +--------------+------------+------------+
    | T            |   5.0      |   ms       |
    +--------------+------------+------------+
    |          synaptic properties           |
    +--------------+------------+------------+
    | E_e          |    0.0     | mV         |
    +--------------+------------+------------+
    | E_i          |   -80.0    | mV         |
    +--------------+------------+------------+
    | Q_e          |    1.0     | nS         |
    +--------------+------------+------------+
    | Q_i          |    5.0     | nS         |
    +--------------+------------+------------+
    | tau_e        |    5.0     | ms         |
    +--------------+------------+------------+
    | tau_i        |    5.0     | ms         |
    +--------------+------------+------------+
    |          numerical network             |
    +--------------+------------+------------+
    | N_tot        |  10000     |            |
    +--------------+------------+------------+
    | p_connect    |    5.0 %   |            |
    +--------------+------------+------------+
    | g            |   20.0 %   |            |
    +--------------+------------+------------+
    |external_input|    0.001   | Hz         |
    +--------------+------------+------------+

    The default coefficients of the transfert function are taken from table I of [MV_2018]_, pag.49
    +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
    |      excitatory cell      |
    +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
    |  -4.98e-02  |   5.06e-03  |  -2.5e-02   |   1.4e-03   |  -4.1e-04   |   1.05e-02  |  -3.6e-02   |   7.4e-03   |   1.2e-03   |  -4.07e-02  |
    +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
    |      inhibitory cell      |
    +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
    |  -5.14e-02  |   4.0e-03   |  -8.3e-03   |   2.0e-04   |  -5.0e-04   |   1.4e-03   |  -1.46e-02  |   4.5e-03   |   2.8e-03   |  -1.53e-02  |
    +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+

    The models (:math:`E`, :math:`I`) phase-plane, including a representation of
    the vector field as well as its nullclines, using default parameters, can be
    seen below:

    .. automethod:: Zerlaut_adaptation_first_order.__init__

    The general formulation for the \textit{\textbf{Zerlaut_adaptation_first_order}} model as a
    dynamical unit at a node $k$ in a BNM with $l$ nodes reads:

    .. math::
            T\dot{E}_k &= F_e-E_k  \\
            T\dot{I}_k &= F_i-I_k  \\
            dot{W}_k &= W_k/tau_w-b*E_k  \\
            F_\lambda = Erfc(V^{eff}_{thre}-\mu_V/\sqrt(2)\sigma_V)

    """
    _ui_name = "Zerlaut_adaptation_first_order"
    ui_configurable_parameters = ['g_L', 'E_L_e', 'E_L_i', 'C_m', 'b', 'tau_w',
                                  'E_e', 'E_i', 'Q_e', 'Q_i', 'tau_e', 'tau_i',
                                  'N_tot', 'p_connect', 'g', 'T',
                                  'external_input']

    # Define traited attributes for this model, these represent possible kwargs.
    g_L = arrays.FloatArray(
        label=":math:`g_{L}`",
        default=numpy.array([10.]),  # 10 nS by default, i.e. ~ 100MOhm input resitance at rest
        range=basic.Range(lo=0.1, hi=100.0, step=0.1),  # 0.1nS would be a very small cell, 100nS a very big one
        doc="""leak conductance [nS]""",
        order=1)

    E_L_e = arrays.FloatArray(
        label=":math:`E_{L}`",
        default=numpy.array([-67.0]),
        range=basic.Range(lo=-90.0, hi=-60.0, step=0.1),  # resting potential, usually between -85mV and -65mV
        doc="""leak reversal potential for excitatory [mV]""",
        order=2)

    E_L_i = arrays.FloatArray(
        label=":math:`E_{L}`",
        default=numpy.array([-63.0]),
        range=basic.Range(lo=-90.0, hi=-60.0, step=0.1),  # resting potential, usually between -85mV and -65mV
        doc="""leak reversal potential for inhibitory [mV]""",
        order=3)

    # N.B. Not independent of g_L, C_m should scale linearly with g_L
    C_m = arrays.FloatArray(
        label=":math:`C_{m}`",
        default=numpy.array([150]),
        range=basic.Range(lo=10.0, hi=500.0, step=10.0),  # 20pF very small cell, 400pF very
        doc="""membrane capacitance [pF]""",
        order=4)

    b = arrays.FloatArray(
        label=":math:`b`",
        default=numpy.array([60.0]),
        range=basic.Range(lo=0.0, hi=150.0, step=1.0),
        doc="""Adaptation [nS]""",
        order=5)

    tau_w = arrays.FloatArray(
        label=":math:`tau_w`",
        default=numpy.array([500.0]),
        range=basic.Range(lo=5.0, hi=1000.0, step=1.0),
        doc="""Adaptation time constant [ms]""",
        order=6)

    E_e = arrays.FloatArray(
        label=r":math:`E_e`",
        default=numpy.array([0.0]),
        range=basic.Range(lo=-20., hi=20., step=0.01),
        doc="""excitatory reversal potential [mV]""",
        order=7)

    E_i = arrays.FloatArray(
        label=":math:`E_i`",
        default=numpy.array([-80.0]),
        range=basic.Range(lo=-100.0, hi=-60.0, step=1.0),
        doc="""inhibitory reversal potential [mV]""",
        order=8)

    Q_e = arrays.FloatArray(
        label=r":math:`Q_e`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.0, hi=5.0, step=0.1),
        doc="""excitatory quantal conductance [nS]""",
        order=9)

    Q_i = arrays.FloatArray(
        label=r":math:`Q_i`",
        default=numpy.array([5.0]),
        range=basic.Range(lo=0.0, hi=10.0, step=0.1),
        doc="""inhibitory quantal conductance [nS]""",
        order=10)

    tau_e = arrays.FloatArray(
        label=":math:`\tau_e`",
        default=numpy.array([5.0]),
        range=basic.Range(lo=1.0, hi=10.0, step=1.0),
        doc="""excitatory decay [ms]""",
        order=11)

    tau_i = arrays.FloatArray(
        label=":math:`\tau_i`",
        default=numpy.array([5.0]),
        range=basic.Range(lo=0.5, hi=10.0, step=0.01),
        doc="""inhibitory decay [ms]""",
        order=12)

    N_tot = arrays.IntegerArray(
        label=":math:`N_{tot}`",
        default=numpy.array([10000]),
        range=basic.Range(lo=1000, hi=50000, step=1000),
        doc="""cell number""",
        order=13)

    p_connect = arrays.FloatArray(
        label=":math:`\epsilon`",
        default=numpy.array([0.05]),
        range=basic.Range(lo=0.001, hi=0.2, step=0.001),  # valid only for relatively sparse connectivities
        doc="""connectivity probability""",
        order=14)

    g = arrays.FloatArray(
        label=":math:`g`",
        default=numpy.array([0.2]),
        range=basic.Range(lo=0.01, hi=0.4, step=0.01),  # inhibitory cell number never overcomes excitatory ones
        doc="""fraction of inhibitory cells""",
        order=15)

    T = arrays.FloatArray(
        label=":math:`T`",
        default=numpy.array([5.0]),
        range=basic.Range(lo=1., hi=20.0, step=0.1),
        doc="""time scale of describing network activity""",
        order=16)

    P_e = arrays.IndexArray(
        label=":math:`P_e`",  # TODO need to check the size of the array when it's used
        default=numpy.array([-4.98e-02,  5.06e-03,  -2.5e-02,  1.4e-03,
                              -4.1e-04,  1.05e-02,  -3.6e-02,  7.4e-03,
                              1.2e-03,  -4.07e-02]),
        doc="""Polynome of excitatory phenomenological threshold (order 9)""",
        order=17)

    P_i = arrays.IndexArray(
        label=":math:`P_i`",  # TODO need to check the size of the array when it's used
        default=numpy.array([-5.14e-02,  4.0e-03, -8.3e-03,  2.0e-04,
                              -5.0e-04,  1.4e-03, -1.46e-02, 4.5e-03,
                              2.8e-03, -1.53e-02]),
        doc="""Polynome of inhibitory phenomenological threshold (order 9)""",
        order=18)


    external_input = arrays.FloatArray(
        label=":math:`\nu_e^{drive}`",
        default=numpy.array([0.0001]),
        range=basic.Range(lo=0.00, hi=0.1, step=0.001),
        doc="""external drive""",
        order=19)

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = basic.Dict(
        label="State Variable ranges [lo, hi]",
        default={"E": numpy.array([0.0, 0.1]),  # actually the 100Hz should be replaced by 1/T_refrac
                 "I": numpy.array([0.0, 0.1]),
                 "W": numpy.array([0.0,100.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random initial
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.\n
        E: firing rate of excitatory population in KHz\n
        I: firing rate of inhibitory population in KHz\N
        W: level of adaptation
        """,
        order=20)

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["E", "I","W"],
        default=["E"],
        select_multiple=True,
        doc="""This represents the default state-variables of this Model to be
               monitored. It can be overridden for each Monitor if desired. The
               corresponding state-variable indices for this model are :math:`E = 0`,
               :math:`I = 1` and :math:`W = 2`.""",
        order=21)

    state_variables = 'E I W'.split()
    _nvar = 3
    cvar = numpy.array([0, 1, 2], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.00):
        r"""
        .. math::
            T \dot{\nu_\mu} &= -F_\mu(\nu_e,\nu_i) + \nu_\mu ,\all\mu\in\{e,i\}\\
            dot{W}_k &= W_k/tau_w-b*E_k  \\

        """
        E = state_variables[0, :]
        I = state_variables[1, :]
        W = state_variables[2, :]
        derivative = numpy.empty_like(state_variables)

        # long-range coupling
        c_0 = coupling[0, :]

        # short-range (local) coupling
        lc_E = local_coupling * E
        lc_I = local_coupling * I

        # Excitatory firing rate derivation
        derivative[0] = (self.TF_excitatory(E+c_0+lc_E+self.external_input, I+lc_I+self.external_input,W)-E)/self.T
        # Inhibitory firing rate derivation
        derivative[1] = (self.TF_inhibitory(E+lc_E+self.external_input, I+lc_I+self.external_input,W)-I)/self.T
        # Adaptation
        derivative[2] = -W/self.tau_w+self.b*E

        return derivative

    def TF_excitatory(self, fe, fi, W):
        """
        transfer function for excitatory population
        :param fe: firing rate of excitatory population
        :param fi: firing rate of inhibitory population
        :param W: level of adaptation
        :return: result of transfer function
        """
        return self.TF(fe, fi, W, self.P_e, self.E_L_e)

    def TF_inhibitory(self, fe, fi, W):
        """
        transfer function for inhibitory population
        :param fe: firing rate of excitatory population
        :param fi: firing rate of inhibitory population
        :param W: level of adaptation
        :return: result of transfer function
        """
        return self.TF(fe, fi, W, self.P_i, self.E_L_i)

    def TF(self, fe, fi, W, P, E_L):
        """
        transfer function for inhibitory population
        Inspired from the next repository :
        https://github.com/yzerlaut/notebook_papers/tree/master/modeling_mesoscopic_dynamics
        :param fe: firing rate of excitatory population
        :param fi: firing rate of inhibitory population
        :param W: level of adaptation
        :param P: Polynome of neurons phenomenological threshold (order 9)
        :param E_L: leak reversal potential
        :return: result of transfer function
        """
        mu_V, sigma_V, T_V = self.get_fluct_regime_vars(fe, fi, W, self.Q_e, self.tau_e, self.E_e,
                                                           self.Q_i, self.tau_i, self.E_i,
                                                           self.g_L, self.C_m, E_L, self.N_tot,
                                                           self.p_connect, self.g)
        V_thre = self.threshold_func(mu_V, sigma_V, T_V*self.g_L/self.C_m,
                                     P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9])
        V_thre *= 1e3  # the threshold need to be in mv and not in Volt
        f_out = self.estimate_firing_rate(mu_V, sigma_V, T_V, V_thre)
        return f_out

    @staticmethod
    def get_fluct_regime_vars(Fe, Fi, W, Q_e, tau_e, E_e, Q_i, tau_i, E_i, g_L, C_m, E_L, N_tot, p_connect, g):
        """
        Compute the mean characteristic of neurons.
        Inspired from the next repository :
        https://github.com/yzerlaut/notebook_papers/tree/master/modeling_mesoscopic_dynamics
        :param Fe: firing rate of excitatory population
        :param Fi: firing rate of inhibitory population
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
        :param p_connect: connectivity probability
        :param g: fraction of inhibitory cells
        :return: mean and variance of membrane voltage of neurons and autocorrelation time constant
        """
        # firing rate
        # 1e-6 represent spontaneous release of synaptic neurotransmitter or some intrinsic currents of neurons
        fe = (Fe+1e-6)*(1.-g)*p_connect*N_tot
        fi = (Fi+1e-6)*g*p_connect*N_tot

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
        # Autocorrelation-time of the fluctuations Eqns 9 from [MV_2018]
        T_V_numerator = (fe*(U_e*tau_e)**2 + fi*(U_i*tau_i)**2)
        T_V_denominator = (fe*(U_e*tau_e)**2/(tau_e+T_m) + fi*(U_i*tau_i)**2/(tau_i+T_m))
        T_V = numpy.divide(T_V_numerator, T_V_denominator, out=numpy.ones_like(T_V_numerator),
                           where=T_V_denominator != 0.0)
        return mu_V, sigma_V, T_V

    @staticmethod
    def threshold_func(muV, sigmaV, TvN, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9):
        """
        The threshold function of the neurons
        :param muV: mean of membrane voltage
        :param sigmaV: variance of membrane voltage
        :param TvN: autocorrelation time constant
        :param P: Fitted coefficients of the transfer functions
        :return: threshold of neurons
        """
        # Normalization factors page 48 after the equation 4 from [ZD_2018]
        muV0, DmuV0 = -60.0, 10.0
        sV0, DsV0 = 4.0, 6.0
        TvN0, DTvN0 = 0.5, 1.
        V = (muV-muV0)/DmuV0
        S = (sigmaV-sV0)/DsV0
        T = (TvN-TvN0)/DTvN0
        # Eqns 11 from [MV_2018]
        return P0 + P1*V + P2*S + P3*T + P4*V**2 + P5*S**2 + P6*T**2 + P7*V*S + P8*V*T + P9*S*T

    @staticmethod
    def estimate_firing_rate(muV, sigmaV, Tv, Vthre):
        # Eqns 10 from [MV_2018]
        return sp_spec.erfc((Vthre-muV) / (numpy.sqrt(2)*sigmaV)) / (2*Tv)


class Zerlaut_adaptation_second_order(Zerlaut_adaptation_first_order):
    r"""
    **References**:
    .. [ZD_2018]  Zerlaut, Y., Chemla, S., Chavane, F. et al. *Modeling mesoscopic cortical dynamics using a mean-field
    model of conductance-based networks of adaptive
    exponential integrate-and-fire neurons*,
    J Comput Neurosci (2018) 44: 45. https://doi-org.lama.univ-amu.fr/10.1007/s10827-017-0668-2
    .. [MV_2018]  Matteo di Volo, Alberto Romagnoni, Cristiano Capone, Alain Destexhe (2018)
    *Mean-field model for the dynamics of conductance-based networks of excitatory and inhibitory spiking neurons
    with adaptation*, bioRxiv, doi: https://doi.org/10.1101/352393

    Used Eqns 4 from [MV_2018]_ in ``dfun``.

    (See Zerlaut_adaptation_first_order for the default value)

    The models (:math:`E`, :math:`I`) phase-plane, including a representation of
    the vector field as well as its nullclines, using default parameters, can be
    seen below:

    .. automethod:: Zerlaut_adaptation_second_order.__init__

    The general formulation for the \textit{\textbf{Zerlaut_adaptation_second_order}} model as a
    dynamical unit at a node $k$ in a BNM with $l$ nodes reads:

    .. math::
        \forall \mu,\lambda,\eta \in \{e,i\}^3\, ,
        \left\{
        \begin{split}
        T \, \frac{\partial \nu_\mu}{\partial t} = & (\mathcal{F}_\mu - \nu_\mu )
        + \frac{1}{2} \, c_{\lambda \eta} \,
        \frac{\partial^2 \mathcal{F}_\mu}{\partial \nu_\lambda \partial \nu_\eta} \\
        T \, \frac{\partial c_{\lambda \eta} }{\partial t}  =  & A_{\lambda \eta} +
        (\mathcal{F}_\lambda - \nu_\lambda ) \, (\mathcal{F}_\eta - \nu_\eta ) + \\
        & c_{\lambda \mu} \frac{\partial \mathcal{F}_\mu}{\partial \nu_\lambda} +
        c_{\mu \eta} \frac{\partial \mathcal{F}_\mu}{\partial \nu_\eta}
        - 2  c_{\lambda \eta}
        \end{split}
        \right.
        dot{W}_k &= W_k/tau_w-b*E_k  \\

        with:
        A_{\lambda \eta} =
        \left\{
        \begin{split}
        \frac{\mathcal{F}_\lambda \, (1/T - \mathcal{F}_\lambda)}{N_\lambda}
        \qquad & \textrm{if  } \lambda=\eta \\
        0 \qquad & \textrm{otherwise}
        \end{split}
        \right.
    """

    _ui_name = "Zerlaut_adaptation_second_order"

    #  Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = basic.Dict(
        label="State Variable ranges [lo, hi]",
        default={"E": numpy.array([0.0, 0.1]), # actually the 100Hz should be replaced by 1/T_refrac
                 "I": numpy.array([0.0, 0.1]),
                 "C_ee": numpy.array([0.0, 0.0]),  # variance is positive or null
                 "C_ei": numpy.array([0.0, 0.0]),  # the co-variance is in [-c_ee*c_ii,c_ee*c_ii]
                 "C_ii": numpy.array([0.0, 0.0]),  # variance is positive or null
                 "W":numpy.array([0.0, 100.0]),
                 },
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.\n
        E: firing rate of excitatory population in KHz\n
        I: firing rate of inhibitory population in KHz\n
        C_ee: the variance of the excitatory population activity \n
        C_ei: the covariance between the excitatory and inhibitory population activities (always symetric) \n
        C_ie: the variance of the inhibitory population activity \n
        W: level of adaptation
        """,
        order=20)

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["E", "I", "C_ee","C_ei","C_ii"],
        default=["E"],
        select_multiple=True,
        doc="""This represents the default state-variables of this Model to be
               monitored. It can be overridden for each Monitor if desired. The
               corresponding state-variable indices for this model are :math:`E = 0`,
               :math:`I = 1`, :math:`C_ee = 2`, :math:`C_ei = 3`, :math:`C_ii = 4` and :math:`W = 5`.""",
        order=21)

    state_variables = 'E I C_ee C_ei C_ii W'.split()
    _nvar = 6
    cvar = numpy.array([0, 1, 2, 3, 4, 5], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.00):
        r"""
        .. math::
            \forall \mu,\lambda,\eta \in \{e,i\}^3\, ,
            \left\{
            \begin{split}
            T \, \frac{\partial \nu_\mu}{\partial t} = & (\mathcal{F}_\mu - \nu_\mu )
            + \frac{1}{2} \, c_{\lambda \eta} \,
            \frac{\partial^2 \mathcal{F}_\mu}{\partial \nu_\lambda \partial \nu_\eta} \\
            T \, \frac{\partial c_{\lambda \eta} }{\partial t}  =  & A_{\lambda \eta} +
            (\mathcal{F}_\lambda - \nu_\lambda ) \, (\mathcal{F}_\eta - \nu_\eta ) + \\
            & c_{\lambda \mu} \frac{\partial \mathcal{F}_\mu}{\partial \nu_\lambda} +
            c_{\mu \eta} \frac{\partial \mathcal{F}_\mu}{\partial \nu_\eta}
            - 2  c_{\lambda \eta}
            \end{split}
            \right.
            dot{W}_k &= W_k/tau_w-b*E_k  \\

            with:
            A_{\lambda \eta} =
            \left\{
            \begin{split}
            \frac{\mathcal{F}_\lambda \, (1/T - \mathcal{F}_\lambda)}{N_\lambda}
            \qquad & \textrm{if  } \lambda=\eta \\
            0 \qquad & \textrm{otherwise}
            \end{split}
            \right.

        """
        N_e = self.N_tot * (1-self.g)
        N_i = self.N_tot * self.g

        E = state_variables[0, :]
        I = state_variables[1, :]
        C_ee = state_variables[2, :]
        C_ei = state_variables[3, :]
        C_ii = state_variables[4, :]
        W = state_variables[5,:]
        derivative = numpy.empty_like(state_variables)

        # long-range coupling
        c_0 = coupling[0, :]

        # short-range (local) coupling
        lc_E = local_coupling * E
        lc_I = local_coupling * I

        E_input_excitatory = E+c_0+lc_E+self.external_input
        E_input_inhibitory = E+lc_E+self.external_input
        I_input_excitatory = I+lc_I+self.external_input
        I_input_inhibitory = I+lc_I+self.external_input

        # Transfer function of excitatory and inhibitory neurons
        _TF_e = self.TF_excitatory(E_input_excitatory, I_input_excitatory, W)
        _TF_i = self.TF_inhibitory(E_input_inhibitory, I_input_inhibitory, W)

        # Derivatives taken numerically : use a central difference formula with spacing `dx`
        def _diff_fe(TF, fe, fi, W, df=1e-7):
            return (TF(fe+df, fi, W)-TF(fe-df, fi, W))/(2*df*1e3)

        def _diff_fi(TF, fe, fi, W, df=1e-7):
            return (TF(fe, fi+df, W)-TF(fe, fi-df, W))/(2*df*1e3)

        def _diff2_fe_fe_e(fe, fi, W, df=1e-7):
            TF = self.TF_excitatory
            return (TF(fe+df, fi, W)-2*_TF_e+TF(fe-df, fi, W))/((df*1e3)**2)

        def _diff2_fe_fe_i(fe, fi, W, df=1e-7):
            TF = self.TF_inhibitory
            return (TF(fe+df, fi, W)-2*_TF_i+TF(fe-df, fi, W))/((df*1e3)**2)

        def _diff2_fi_fe(TF, fe, fi, W, df=1e-7):
            return (_diff_fi(TF, fe+df, fi, W)-_diff_fi(TF, fe-df, fi, W))/(2*df*1e3)

        def _diff2_fe_fi(TF, fe, fi, W, df=1e-7):
            return (_diff_fe(TF, fe, fi+df, W)-_diff_fe(TF, fe, fi-df, W))/(2*df*1e3)

        def _diff2_fi_fi_e(fe, fi, W, df=1e-7):
            TF = self.TF_excitatory
            return (TF(fe, fi+df, W)-2*_TF_e+TF(fe, fi-df, W))/((df*1e3)**2)

        def _diff2_fi_fi_i(fe, fi, W, df=1e-7):
            TF = self.TF_inhibitory
            return (TF(fe, fi+df, W)-2*_TF_i+TF(fe, fi-df, W))/((df*1e3)**2)

        #Precompute some result
        _diff_fe_TF_e = _diff_fe(self.TF_excitatory, E_input_excitatory, I_input_excitatory, W)
        _diff_fe_TF_i = _diff_fe(self.TF_inhibitory, E_input_inhibitory, I_input_inhibitory, W)
        _diff_fi_TF_e = _diff_fi(self.TF_excitatory, E_input_excitatory, I_input_excitatory, W)
        _diff_fi_TF_i = _diff_fi(self.TF_inhibitory, E_input_inhibitory, I_input_inhibitory, W)

        # equation is inspired from github of Zerlaut :
        # https://github.com/yzerlaut/notebook_papers/blob/master/modeling_mesoscopic_dynamics/mean_field/master_equation.py
        # Excitatory firing rate derivation
        derivative[0] = (_TF_e - E
                         + .5*C_ee*_diff2_fe_fe_e(E_input_excitatory, I_input_excitatory, W)
                         + .5*C_ei*_diff2_fe_fi(self.TF_excitatory, E_input_excitatory, I_input_excitatory, W)
                         + .5*C_ei*_diff2_fi_fe(self.TF_excitatory, E_input_excitatory, I_input_excitatory, W)
                         + .5*C_ii*_diff2_fi_fi_e(E_input_excitatory, I_input_excitatory, W)
                         )/self.T
        # Inhibitory firing rate derivation
        derivative[1] = (_TF_i - I
                         + .5*C_ee*_diff2_fe_fe_i(E_input_inhibitory, I_input_inhibitory, W)
                         + .5*C_ei*_diff2_fe_fi(self.TF_inhibitory, E_input_inhibitory, I_input_inhibitory, W)
                         + .5*C_ei*_diff2_fi_fe(self.TF_inhibitory, E_input_inhibitory, I_input_inhibitory, W)
                         + .5*C_ii*_diff2_fi_fi_i(E_input_inhibitory, I_input_inhibitory, W)
                         )/self.T
        # Covariance excitatory-excitatory derivation
        derivative[2] = (_TF_e*(1./self.T-_TF_e)/N_e
                         + (_TF_e-E)**2
                         + 2.*C_ee*_diff_fe_TF_e
                         + 2.*C_ei*_diff_fi_TF_i
                         - 2.*C_ee
                         )/self.T
        # Covariance excitatory-inhibitory or inhibitory-excitatory derivation
        derivative[3] = ((_TF_e-E)*(_TF_i-I)
                         + C_ee*_diff_fe_TF_e
                         + C_ei*_diff_fe_TF_i
                         + C_ei*_diff_fi_TF_e
                         + C_ii*_diff_fi_TF_i
                         - 2.*C_ei
                         )/self.T
        # Covariance inhibitory-inhibitory derivation
        derivative[4] = (_TF_i*(1./self.T-_TF_i)/N_i
                         + (_TF_i-I)**2
                         + 2.*C_ii*_diff_fi_TF_i
                         + 2.*C_ei*_diff_fe_TF_e
                         - 2.*C_ii
                         )/self.T
        # Adaptation
        derivative[5] = -W/self.tau_w+self.b*E

        return derivative

