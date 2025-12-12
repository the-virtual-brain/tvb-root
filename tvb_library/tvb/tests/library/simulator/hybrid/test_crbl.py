"""
Test case for hybrid from Roberta Lorenzi.

https://github.com/RobertaMLo/cMF-TVB
"""

from tvb.simulator.models.base import Model, numpy
from tvb.basic.neotraits.api import NArray, Range, Final, List
import scipy.special as sp_spec
from numba import jit
import numpy as np  # TO BE DELETED PERCHE' USO QUELLO DI TVB!!!!!
import scipy.sparse
from tvb.simulator.integrators import HeunDeterministic, HeunStochastic
from tvb.simulator.hybrid import (
    NetworkSet,
    Subnetwork,
    InterProjection,
    Recorder,
    Simulator,
)
from tvb.datatypes.connectivity import Connectivity

# from tvb.simulator.models import ZerlautAdaptationFirstOrder, CrblCorticalFirstOrder, JansenRit
from tvb.simulator.models import ZerlautAdaptationFirstOrder, JansenRit
from tvb.simulator.monitors import TemporalAverage, Raw, Bold


def test_crbl_model():
    "transcribed from notebook"
    nets, vars = setup(k=1e-3)
    locals().update(vars)  # XXX: avoid using locals()

    # zero state seems unstable
    inits = nets.zero_states()
    inits.cortex[:] += 0.1
    inits.thalamus[:] += 0.01

    # tavg = TemporalAverage(period=0.1)
    traw = Raw()
    sim = Simulator(
        nets=nets,
        simulation_length=100.0,
        monitors=[traw],
    )
    sim.configure()
    ((t, y),) = sim.run(initial_conditions=inits)

    # check shape
    assert y.shape == (1000, 1, 76, 1)  # time, state var, node, mode

    # check finit
    assert np.all(np.isfinite(y)), "non-finite values in simulation output"


def setup(k=1e-3):
    conn = Connectivity.from_file()
    conn.configure()
    np.random.seed(42)

    class AtlasPart:
        CORTEX = 0
        THALAMUS = 1

    # randomly choose to assign roi to cortex or thalamus
    ix = np.random.randint(low=0, high=2, size=conn.number_of_regions)

    from tvb.simulator.noise import Additive

    noise = Additive(nsig=np.r_[1e-5])
    scheme = HeunStochastic(dt=0.1, noise=noise)
    noise.configure_white(scheme.dt)
    scheme.configure()

    jrkwargs = {}
    fhnkwargs = {}
    nvois = 2  # use 2 variables of interest instead of all
    fhnkwargs["variables_of_interest"] = (
        ZerlautAdaptationFirstOrder.variables_of_interest.default[:nvois]
    )
    jrkwargs["variables_of_interest"] = (
        CrblCorticalFirstOrder.variables_of_interest.default[:nvois]
    )

    # Create subnetworks with just their size and behavior
    cortex = Subnetwork(
        name="cortex",
        model=ZerlautAdaptationFirstOrder(**fhnkwargs),
        scheme=scheme,
        nnodes=(ix == AtlasPart.CORTEX).sum(),
    ).configure()

    thalamus = Subnetwork(
        name="thalamus",
        model=CrblCorticalFirstOrder(**jrkwargs),
        scheme=scheme,
        nnodes=(ix == AtlasPart.THALAMUS).sum(),
    ).configure()

    # Create projections with explicit weights from global connectivity
    cortex_indices = np.where(ix == AtlasPart.CORTEX)[0]
    thalamus_indices = np.where(ix == AtlasPart.THALAMUS)[0]

    # Use valid coupling variable indices for each model
    # JansenRit has 2 coupling variables (0,1)
    # ReducedSetFitzHughNagumo has 2 coupling variables (0,1)

    # Prepare sparse weights and dummy lengths/params for projections
    weights_c_t_dense = conn.weights[thalamus_indices][:, cortex_indices]
    weights_c_t_sparse = scipy.sparse.csr_matrix(weights_c_t_dense)
    weights_t_c_dense = conn.weights[cortex_indices][:, thalamus_indices]
    weights_t_c_sparse = scipy.sparse.csr_matrix(weights_t_c_dense)

    # Create sparse lengths with same sparsity pattern as weights
    # Dummy lengths data [0, 10)
    lengths_c_t_data = np.random.rand(weights_c_t_sparse.nnz) * 10.0
    lengths_c_t_sparse = scipy.sparse.csr_matrix(
        (lengths_c_t_data, weights_c_t_sparse.indices, weights_c_t_sparse.indptr),
        shape=weights_c_t_sparse.shape,
    )
    # Dummy lengths data [0, 10)
    lengths_t_c_data = np.random.rand(weights_t_c_sparse.nnz) * 10.0
    lengths_t_c_sparse = scipy.sparse.csr_matrix(
        (lengths_t_c_data, weights_t_c_sparse.indices, weights_t_c_sparse.indptr),
        shape=weights_t_c_sparse.shape,
    )

    # Dummy delay parameters (can be overridden in specific tests if needed)
    default_cv = 3.0
    default_dt = scheme.dt

    nets = NetworkSet(
        subnets=[cortex, thalamus],
        projections=[
            InterProjection(
                source=cortex,
                target=thalamus,
                source_cvar=np.r_[0],
                target_cvar=np.r_[0],  # eccitatorio from ctx su GrC of crbl mfm
                weights=weights_c_t_sparse,
                scale=k,
                lengths=lengths_c_t_sparse,
                cv=default_cv,
                dt=default_dt,  # Use sparse lengths
            ),
            InterProjection(
                source=cortex,
                target=thalamus,
                source_cvar=np.r_[0],
                target_cvar=np.r_[0],
                weights=weights_c_t_sparse,
                scale=k,
                # Reusing weights, so must reuse sparse lengths for consistency
                lengths=lengths_c_t_sparse,
                cv=default_cv,
                dt=default_dt,  # Use sparse lengths
            ),
            InterProjection(
                source=thalamus,
                target=cortex,
                source_cvar=np.r_[0],
                target_cvar=np.r_[0],
                scale=k,
                weights=weights_t_c_sparse,
                lengths=lengths_t_c_sparse,
                cv=default_cv,
                dt=default_dt,  # Use sparse lengths
            ),
        ],
    )

    # Configure time-delay buffers for all projections in the NetworkSet
    for p in nets.projections:
        p.configure_buffer(
            n_vars_src=p.source.model.nvar,
            n_nodes_src=p.source.nnodes,
            n_modes_src=p.source.model.number_of_modes,
        )

    return nets, locals()


class CrblCorticalFirstOrder(Model):
    r"""

    CEREBELLAR Equations taken from Lorenzi et al., 2023
    Code from https://github.com/RobertaMLo/cMF-TVB
    w/ permission from Roberta Lorenzi.

    """

    _ui_name = "crbl_cortical_first_ord"
    ui_configurable_parameters = [
        "g_L_grc",
        "g_L_goc",
        "g_L_mli",
        "g_L_pc",
        "E_L_grc",
        "E_L_goc",
        "E_L_mli",
        "E_L_pc",
        "C_m_grc",
        "C_m_goc",
        "C_m_mli",
        "C_m_pc",
        "E_e",
        "E_i",
        "Q_mf_grc",
        "Q_mf_goc",
        "Q_grc_goc",
        "Q_grc_mli",
        "Q_grc_pc",
        "Q_goc_goc",
        "Q_goc_grc",
        "Q_mli_mli",
        "Q_mli_pc",
        "tau_mf_grc",
        "tau_mf_goc",
        "tau_grc_goc",
        "tau_grc_mli",
        "tau_grc_pc",
        "tau_goc_goc",
        "tau_goc_grc",
        "tau_mli_mli",
        "tau_mli_pc",
        "K_mf_grc",
        "K_mf_goc",
        "K_mf_goc",
        "K_grc_mli",
        "K_grc_pc",
        "K_goc_goc",
        "K_goc_grc",
        "K_mli_mli",
        "K_mli_pc",
        "N_grc",
        "N_goc",
        "N_mli",
        "N_pc",
        "N_mossy",
        "T",
    ]

    # Define traited attributes for this model, these represent possible kwargs.

    ## =============================================================================================================================
    ## ================================ CRBL MF PARAMETERS =========================================================================
    ## =============================================================================================================================
    g_L_grc = NArray(
        label=":math:`gGrC_{L}`",
        default=numpy.array([0.29]),
        domain=Range(lo=0.25, hi=0.35, step=0.1),
        doc="""Granule cells leak conductance [nS]""",
    )

    g_L_goc = NArray(
        label=":math:`gGoC_{L}`",
        default=numpy.array([3.30]),
        domain=Range(lo=3.25, hi=3.35, step=0.1),
        doc="""Golgi cells leak conductance [nS]""",
    )

    g_L_mli = NArray(
        label=":math:`gMLI_{L}`",
        default=numpy.array([1.60]),
        domain=Range(lo=1.55, hi=1.65, step=0.1),
        doc="""leak conductance [nS]""",
    )

    g_L_pc = NArray(
        label=":math:`gPC_{L}`",
        default=numpy.array([7.10]),
        domain=Range(lo=7.15, hi=7.5, step=0.1),
        doc="""leak conductance [nS]""",
    )

    # Standard deviation (Domanin) from Geminiani et al. 2019
    E_L_grc = NArray(
        label=":math:`EGrC_{L}`",
        default=numpy.array([-62.0]),
        domain=Range(lo=-62.1, hi=-61.9, step=0.1),
        doc="""leak reversal potential for excitatory [mV]""",
    )

    E_L_goc = NArray(
        label=":math:`EGoC_{L}`",
        default=numpy.array([-62.0]),
        domain=Range(lo=-73.0, hi=-51.0, step=0.1),
        doc="""leak reversal potential for inhibitory [mV]""",
    )

    E_L_mli = NArray(
        label=":math:`EMLI_{L}`",
        default=numpy.array([-68.0]),
        domain=Range(lo=-68.01, hi=-67.9, step=0.1),
        doc="""leak reversal potential for excitatory [mV]""",
    )

    E_L_pc = NArray(
        label=":math:`E_{L}`",
        default=numpy.array([-59.0]),
        domain=Range(lo=-65.0, hi=-53.0, step=0.1),
        doc="""leak reversal potential for inhibitory [mV]""",
    )

    # N.B. Not independent of g_L, C_m should scale linearly with g_L
    C_m_grc = NArray(
        label=":math:`CGrC_{m}`",
        default=numpy.array([7.0]),
        domain=Range(lo=5.0, hi=7.5, step=1.0),
        doc="""membrane capacitance [pF]""",
    )

    C_m_goc = NArray(
        label=":math:`CGoC_{m}`",
        default=numpy.array([145.0]),
        domain=Range(lo=72.0, hi=218.0, step=10.0),
        doc="""membrane capacitance [pF]""",
    )

    C_m_mli = NArray(
        label=":math:`CMLI_{m}`",
        default=numpy.array([14.6]),
        domain=Range(lo=14.5, hi=14.7, step=0.1),
        doc="""membrane capacitance [pF]""",
    )

    C_m_pc = NArray(
        label=":math:`CPC_{m}`",
        default=numpy.array([334.0]),
        domain=Range(lo=228.0, hi=440.0, step=10.0),
        doc="""membrane capacitance [pF]""",
    )

    E_e = NArray(
        label=r":math:`E_e`",
        default=numpy.array([0.0]),
        domain=Range(lo=-20.0, hi=20.0, step=0.01),
        doc="""excitatory reversal potential [mV]""",
    )

    E_i = NArray(
        label=":math:`E_i`",
        default=numpy.array([-80.0]),
        domain=Range(lo=-100.0, hi=-60.0, step=1.0),
        doc="""inhibitory reversal potential [mV]""",
    )

    Q_mf_grc = NArray(
        label=r":math:`Q_mf_grc_e`",
        default=numpy.array([0.230]),
        domain=Range(lo=0.225, hi=0.235, step=0.001),
        doc="""excitatory quantal conductance [nS]""",
    )

    Q_mf_goc = NArray(
        label=r":math:`Q_mf_goc_e`",
        default=numpy.array([0.240]),
        domain=Range(lo=0.235, hi=0.245, step=0.001),
        doc="""inhibitory quantal conductance [nS]""",
    )

    Q_grc_goc = NArray(
        label=r":math:`Q_grc_goc_e`",
        default=numpy.array([0.437]),
        domain=Range(lo=0.432, hi=0.542, step=0.001),
        doc="""excitatory quantal conductance [nS]""",
    )

    Q_grc_mli = NArray(
        label=r":math:`Q_grc_mli_e`",
        default=numpy.array([0.154]),
        domain=Range(lo=0.149, hi=0.159, step=0.001),
        doc="""excitatory quantal conductance [nS]""",
    )

    Q_grc_pc = NArray(
        label=r":math:`Q_e`",
        default=numpy.array([1.126]),
        domain=Range(lo=1.120, hi=1.131, step=0.001),
        doc="""excitatory quantal conductance [nS]""",
    )

    Q_goc_grc = NArray(
        label=r":math:`Q_goc_grc_i`",
        default=numpy.array([0.336]),
        domain=Range(lo=0.330, hi=0.341, step=0.001),
        doc="""excitatory quantal conductance [nS]""",
    )

    Q_goc_goc = NArray(
        label=r":math:`Q_goc_goc_i`",
        default=numpy.array([1.120]),
        domain=Range(lo=1.115, hi=1.130, step=0.001),
        doc="""excitatory quantal conductance [nS]""",
    )

    Q_mli_mli = NArray(
        label=r":math:`Q_mli_mli_i`",
        default=numpy.array([0.532]),
        domain=Range(lo=0.527, hi=0.537, step=0.001),
        doc="""excitatory quantal conductance [nS]""",
    )

    Q_mli_pc = NArray(
        label=r":math:`Q_mli_pc_i`",
        default=numpy.array([1.244]),
        domain=Range(lo=1.240, hi=1.250, step=0.001),
        doc="""excitatory quantal conductance [nS]""",
    )

    tau_mf_grc = NArray(
        label=":math:`\tau_mf_grc_e`",
        default=numpy.array([1.9]),
        domain=Range(lo=1.15, hi=1.9, step=0.1),
        doc="""excitatory decay [ms]""",
    )

    tau_mf_goc = NArray(
        label=":math:`\tau_mf_goc_e`",
        default=numpy.array([5.0]),
        domain=Range(lo=4.5, hi=5.5, step=0.1),
        doc="""excitatory decay [ms]""",
    )

    tau_grc_goc = NArray(
        label=":math:`\tau_grc_goc_e`",
        default=numpy.array([1.25]),
        domain=Range(lo=1.05, hi=1.45, step=0.1),
        doc="""excitatory decay [ms]""",
    )

    tau_grc_mli = NArray(
        label=":math:`\tau_grc_mli_e`",
        default=numpy.array([0.64]),
        domain=Range(lo=0.44, hi=0.84, step=0.1),
        doc="""excitatory decay [ms]""",
    )

    tau_grc_pc = NArray(
        label=":math:`\tau_grc_pc_e`",
        default=numpy.array([1.1]),
        domain=Range(lo=1.0, hi=1.2, step=0.1),
        doc="""excitatory decay [ms]""",
    )

    tau_goc_grc = NArray(
        label=":math:`\tau_goc_grc_i`",
        default=numpy.array([4.5]),
        domain=Range(lo=4.0, hi=5.0, step=0.1),
        doc="""excitatory decay [ms]""",
    )

    tau_goc_goc = NArray(
        label=":math:`\tau_goc_goc_i`",
        default=numpy.array([5.0]),
        domain=Range(lo=4.5, hi=5.5, step=0.1),
        doc="""excitatory decay [ms]""",
    )

    tau_mli_mli = NArray(
        label=":math:`\tau_mli_mli_i`",
        default=numpy.array([2.0]),
        domain=Range(lo=1.5, hi=2.5, step=0.1),
        doc="""excitatory decay [ms]""",
    )

    tau_mli_pc = NArray(
        label=":math:`\tau_mli_pc_i`",
        default=numpy.array([2.8]),
        domain=Range(lo=2.3, hi=3.2, step=0.1),
        doc="""excitatory decay [ms]""",
    )

    K_mossy_grc = NArray(
        label=":math:`K_mossy_grc_e`",
        default=numpy.array([4.0]),
        domain=Range(lo=0.0, hi=10.0, step=1.0),
        doc="""synaptic convergence [-]""",
    )

    K_mossy_goc = NArray(
        label=":math:`K_mossy_goc_e`",
        default=numpy.array([35.0]),
        domain=Range(lo=15.0, hi=55.0, step=10.0),
        doc="""synaptic convergence [-]""",
    )

    K_grc_goc = NArray(
        label=":math:`K_grc_grc_e`",
        default=numpy.array([501.98]),
        domain=Range(lo=451.98, hi=551.0, step=10.0),
        doc="""synaptic convergence [-]""",
    )

    K_grc_mli = NArray(
        label=":math:`K_grc_mli_e`",
        default=numpy.array([243.96]),
        domain=Range(lo=193.96, hi=293.96, step=10.0),
        doc="""synaptic convergence [-]""",
    )

    K_grc_pc = NArray(
        label=":math:`K_grc_pc_e`",
        default=numpy.array([374.50]),
        domain=Range(lo=334.50, hi=404.50, step=10.0),
        doc="""synaptic convergence [-]""",
    )

    K_goc_goc = NArray(
        label=":math:`K_goc_goc_e`",
        default=numpy.array([16.2]),
        domain=Range(lo=10.2, hi=20.2, step=1.0),
        doc="""synaptic convergence [-]""",
    )

    K_mli_mli = NArray(
        label=":math:`K_mli_mli_i`",
        default=numpy.array([14.20]),
        domain=Range(lo=10.20, hi=20.20, step=1.0),
        doc="""synaptic convergence [-]""",
    )

    K_mli_pc = NArray(
        label=":math:`K_mli_pc_i`",
        default=numpy.array([10.28]),
        domain=Range(lo=5.28, hi=15.28, step=1.0),
        doc="""synaptic convergence [-]""",
    )

    N_grc = NArray(
        dtype=int,
        label=":math:`NGrC_{tot}`",
        default=numpy.array([28615]),
        domain=Range(lo=25615, hi=31615, step=1000),
        doc="""cell number""",
    )

    N_goc = NArray(
        dtype=int,
        label=":math:`NGoC_{tot}`",
        default=numpy.array([70]),
        domain=Range(lo=10, hi=100, step=10),
        doc="""cell number""",
    )

    N_mli = NArray(
        dtype=int,
        label=":math:`NMLI_{tot}`",
        default=numpy.array([446]),
        domain=Range(lo=146, hi=946, step=100),
        doc="""cell number""",
    )

    N_pc = NArray(
        dtype=int,
        label=":math:`NPC_{tot}`",
        default=numpy.array([99]),
        domain=Range(lo=29, hi=149, step=10),
        doc="""cell number""",
    )

    N_mossy = NArray(
        dtype=int,
        label=":math:`Nmossy_{tot}`",
        default=numpy.array([2336]),
        domain=Range(lo=336, hi=5336, step=1000),
        doc="""cell number""",
    )

    alpha_grc = NArray(
        dtype=float,
        label=":math:`alphaGrC`",
        default=numpy.array([2]),
        domain=Range(lo=2, hi=2, step=1),
        doc="""cell number""",
    )

    alpha_goc = NArray(
        dtype=float,
        label=":math: alphaGoC",
        default=numpy.array([1.3]),
        domain=Range(lo=1.3, hi=1.3, step=1),
        doc="""cell number""",
    )

    alpha_mli = NArray(
        dtype=float,
        label=":math:`alphaMLI`",
        default=numpy.array([5]),  # 5 official
        domain=Range(lo=5, hi=5, step=1),
        doc="""Number of excitatory connexions from external population""",
    )

    alpha_pc = NArray(
        dtype=float,
        label=":math:`alphaPC`",
        default=numpy.array([5]),
        domain=Range(lo=5, hi=5, step=1),
        doc="""Number of inhibitory connexions from external population""",
    )

    T = NArray(
        label=":math:`T`",
        default=numpy.array([3.5]),
        domain=Range(lo=3.45, hi=3.55, step=0.01),
        doc="""time scale of describing network activity""",
    )

    P_grc = NArray(
        label=":math:`PGrC_e`",
        default=numpy.array([-0.426, 0.007, 0.023, 0.482, 0.216]),
        doc="""Polynome of excitatory GrC phenomenological threshold (order 5)""",
    )

    P_goc = NArray(
        label=":math:`PGoC_i`",
        default=numpy.array([-0.144, 0.003, 0.011, 0.031, 0.011]),
        doc="""Polynome of inhibitory GoC phenomenological threshold (order 5)""",
    )

    P_mli = NArray(
        label=":math:`PMLI_i`",
        default=numpy.array([-0.128, -0.001, 0.012, -0.093, -0.063]),
        doc="""Polynome of inhibitory phenomenological threshold (order 5)""",
    )

    P_pc = NArray(
        label=":math:`PPC_i`",
        default=numpy.array([-0.080, 0.009, 0.004, 0.006, 0.014]),
        doc="""Polynome of inhibitory phenomenological threshold (order 5)""",
    )

    tau_OU = NArray(
        label=":math:`\ntau noise`",
        default=numpy.array([5.0]),
        domain=Range(lo=0.10, hi=10.0, step=0.01),
        doc="""time constant noise""",
    )

    ## =============================================================================================================================
    ## ================================ NOISE and EXT INPT (used for crblMF) ===================================================
    ## =========================================================================================================================

    weight_noise = NArray(
        label=":math:`\nweight noise`",
        default=numpy.array([10.5]),
        domain=Range(lo=0.0, hi=50.0, step=1.0),
        doc="""weight noise""",
    )

    external_input_ex_ex = NArray(
        label=":math:`\nu_e^{drive}`",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""external drive""",
    )

    external_input_ex_in = NArray(
        label=":math:`\nu_e^{drive}`",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""external drive""",
    )

    external_input_in_ex = NArray(
        label=":math:`\nu_e^{drive}`",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""external drive""",
    )

    external_input_in_in = NArray(
        label=":math:`\nu_e^{drive}`",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""external drive""",
    )

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "d1": numpy.array([0.0, 0.0]),  # GrC, E
            "d2": numpy.array([0.0, 0.0]),  # GoC, I
            "d3": numpy.array([0.0, 0.0]),  # MLI, We
            "d4": numpy.array([0.0, 0.0]),  # PC, Wi
            "noise": numpy.array([0.0, 0.0]),
        },
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random initial
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plot
        """,
    )

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=(
            "d1",
            "d2",
            "d3",
            "d4",
            "noise",
        ),  # "GrC","GoC","MLI","PC","noise"; # E, I, We, Wi, noise
        default=("d1",),
        doc="""This represents the default state-variables of this Model to be
               monitored. It can be overridden for each Monitor if desired""",
    )

    state_variable_boundaries = Final(
        label="Firing rate of population is always positive",
        default={
            "d1": numpy.array([0.0, None]),
            "d2": numpy.array([0.0, None]),
            "d3": numpy.array([0.0, None]),
            "d4": numpy.array([0.0, None]),
        },
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. Set None for one-sided boundaries""",
    )

    state_variables = "d1 d2 d3 d4 noise".split()

    _nvar = 5
    cvar = numpy.array([0], dtype=int)

    _inds = {}

    def dfun(self, state_variables, coupling, local_coupling=0.00):
        r"""
        .. math::
            T \dot{\nu_\mu} &= -F_\mu(\nu_e,\nu_i) + \nu_\mu ,\all\mu\in\{e,i\}\\
            dot{W}_k &= W_k/tau_w-b*E_k  \\

        """

        d1 = state_variables[0, :]
        d2 = state_variables[1, :]
        d3 = state_variables[2, :]
        d4 = state_variables[3, :]
        noise = state_variables[4, :]
        derivative = numpy.empty_like(state_variables)

        # print('dimension of derivative ********', np.shape(derivative))
        # print('dimension of d1', np.shape(d1))

        # long-range coupling ---> ALREADY IN Hz. SEE THE FEXT EQUATION
        c_0 = coupling[0, :]
        # print('SHAPE C0', np.shape(c_0))

        # local coupling --> Not yet in Hz because I multiplied it for the activity
        lc_d1 = local_coupling * d1
        lc_d2 = local_coupling * d2
        lc_d3 = local_coupling * d3
        lc_d4 = local_coupling * d4

        # # #external firing rate -------
        # # standard config
        # # Fe_ext_tod1 = c_0 + lc_d1 + self.weight_noise * noise

        # # ---------  TO GRC : only from mossy fibers!!!!!! From DCN or from cerebrum!!!!

        Fe_ext_tod1 = (
            c_0 * 0.57
        ) * 0.97 + self.weight_noise * noise  # background noise from cerebrum

        # print('************************** Fext to grc *********************************')
        # print(Fe_ext_toGrC)

        Fe_ext_tod2 = (
            (c_0 * 0.57) * 0.03 + (c_0 * 0.43) * 0.14 + self.weight_noise * noise
        )

        # # --------- TO MLI : From PARALLEL of ADJACENT MODULE
        Fe_ext_tod3 = (c_0 * 0.43) * 0.55 + self.weight_noise * noise

        # # --------- TO PC : From PARALLEL of ADJACENT MODULE
        Fe_ext_tod4 = (c_0 * 0.43) * 0.31 + self.weight_noise * noise

        # check on F_ext value. Must be positive
        index_bad_input = numpy.where(Fe_ext_tod1 * self.K_mossy_grc < 0)
        Fe_ext_tod1[index_bad_input] = 0.0

        index_bad_input = numpy.where(Fe_ext_tod2 * self.K_mossy_goc < 0)
        Fe_ext_tod2[index_bad_input] = 0.0

        index_bad_input = numpy.where(Fe_ext_tod3 * self.K_grc_mli < 0)
        Fe_ext_tod3[index_bad_input] = 0.0

        index_bad_input = numpy.where(Fe_ext_tod4 * self.K_grc_pc < 0)
        Fe_ext_tod4[index_bad_input] = 0.0

        # Fe_ext_tod2 = Fe_ext_tod2 * 2.5 #*5 #DI CUORE CNSIDERANDO RAPP GRC GOC FISSO
        Fi_ext = 0.0

        ############################## DERIVATIVE 1 ####################################################################
        # d1 = GrC
        derivative[0] = (
            self.TF_excitatory_grc(
                Fe_ext_tod1 + self.external_input_ex_ex,
                d2,
                0,
                Fi_ext + self.external_input_ex_in,
                0,
            )
            - d1
        ) / self.T

        ############################## DERIVATIVE 2 ####################################################################
        # d2 = GoC
        derivative[1] = (
            self.TF_inhibitory_goc(
                d1, d2, Fe_ext_tod2 + self.external_input_in_ex, Fi_ext, 0
            )
            - d2
        ) / self.T

        ############################## DERIVATIVE 3 ####################################################################
        # d3 = MLI
        derivative[2] = (
            self.TF_inhibitory_mli(d1, d3, Fe_ext_tod3, Fi_ext, 0) - d3
        ) / self.T

        ############################## DERIVATIVE 4 ####################################################################
        # d4 = PC
        derivative[3] = (
            self.TF_inhibitory_pc(d1, d3, Fe_ext_tod4, Fi_ext, 0) - d4
        ) / self.T

        ############################## DERIVATIVE 5 ####################################################################
        derivative[4] = -noise / self.tau_OU

        return derivative

    def TF_excitatory_grc(
        self, fe_ext, fi, fe, fi_ext=0, W=0
    ):  ### FRANCAMENTE MI SEMBRA UGUALE A TF INHIB, A PARTE IL RETURN
        """
        transfer function for excitatory population: Granule cells
        return: result of transfer function
        """
        # input TF : Fe, Fi, Fe_ext, Fi_ext, W, P, Q_e, Q_i, tau_e, tau_i, E_e, E_i, g_L, C_m, E_L, Ke, Ki
        return self.TF(
            fe_ext,
            fi,
            fe,
            fi_ext,
            W,
            self.P_grc,
            self.Q_mf_grc,
            self.Q_goc_grc,
            self.tau_mf_grc,
            self.tau_goc_grc,
            self.E_e,
            self.E_i,
            self.g_L_grc,
            self.C_m_grc,
            self.E_L_grc,
            self.K_mossy_grc,
            self.K_mossy_goc,
            self.alpha_grc,
        )

    def TF_inhibitory_goc(self, fe, fi, fe_ext, fi_ext=0, W=0):
        """
        transfer function for inhibitory population: Golgi cells
        :return: result of transfer function
        """
        return self.TF_goc(
            fe,
            fi,
            fe_ext,
            fi_ext,
            W,
            self.P_goc,
            self.Q_grc_goc,
            self.Q_goc_goc,
            self.tau_grc_goc,
            self.tau_goc_goc,
            self.E_i,
            self.E_i,
            self.g_L_goc,
            self.C_m_goc,
            self.E_L_goc,
            self.K_grc_goc,
            self.K_goc_goc,
            self.Q_mf_goc,
            self.tau_mf_goc,
            self.K_mossy_goc,
            self.alpha_goc,
        )

    # (self, Fe, Fi, Fe_ext, Fi_ext, W, P, Qe_gr, Qi, Te_gr, #Ti, Ee, Ei, Gl, Cm, El, Ke_grc, Ki, Qe_ext, Te_ext, Ke_ext, Ki_ext=0):

    def TF_inhibitory_mli(self, fe, fi, fe_ext, fi_ext=0, W=0):
        """
        transfer function for inhibitory population: Molecular layer interneurons
        :return: result of transfer function
        """
        return self.TF(
            fe,
            fi,
            fe_ext,
            fi_ext,
            W,
            self.P_mli,
            self.Q_grc_mli,
            self.Q_mli_mli,
            self.tau_grc_mli,
            self.tau_mli_mli,
            self.E_e,
            self.E_i,
            self.g_L_mli,
            self.C_m_mli,
            self.E_L_mli,
            self.K_grc_mli,
            self.K_mli_mli,
            self.alpha_mli,
        )

    def TF_inhibitory_pc(self, fe, fi, fe_ext, fi_ext=0, W=0):
        """
        transfer function for inhibitory population: Purkinje cells
        :return: result of transfer function
        """
        return self.TF(
            fe,
            fi,
            fe_ext,
            fi_ext,
            W,
            self.P_pc,
            self.Q_grc_pc,
            self.Q_mli_pc,
            self.tau_grc_pc,
            self.tau_mli_pc,
            self.E_e,
            self.E_i,
            self.g_L_pc,
            self.C_m_pc,
            self.E_L_pc,
            self.K_grc_pc,
            self.K_mli_pc,
            self.alpha_pc,
        )

    def TF(
        self,
        Fe,
        Fi,
        Fe_ext,
        Fi_ext,
        W,
        P,
        Q_e,
        Q_i,
        tau_e,
        tau_i,
        E_e,
        E_i,
        g_L,
        C_m,
        E_L,
        Ke,
        Ki,
        alpha,
    ):
        """
        2D transfer functions
        https://github.com/RobertaMLo/CRBL_MF

        :return: result of transfer function
        """
        # mu_V, sigma_V, T_V = self.get_fluct_regime_vars(fe, fi, fe_ext, fi_ext, W, self.Q_e, self.tau_e, self.E_e,
        #                                                   self.Q_i, self.tau_i, self.E_i,
        #                                                   self.g_L, self.C_m, E_L, self.N_tot,
        #                                                   self.p_connect_e,self.p_connect_i, self.g,self.K_ext_e,self.K_ext_i)

        mu_V, sigma_V, T_V, muGn = self.get_fluct_regime_vars(
            Fe,
            Fi,
            Fe_ext,
            Fi_ext,
            W,
            Q_e,
            tau_e,
            E_e,
            Q_i,
            tau_i,
            E_i,
            g_L,
            C_m,
            E_L,
            Ke,
            Ki,
            K_ext_e=0,
            K_ext_i=0,
        )

        V_thre = self.threshold_func(
            mu_V, sigma_V, T_V, muGn, P[0], P[1], P[2], P[3], P[4]
        )
        V_thre *= (
            1e3  # the threshold need to be in mv and not in Volt #OK LASCIATA IN VOLT
        )
        f_out = self.estimate_firing_rate(mu_V, sigma_V, T_V, V_thre, g_L, C_m, alpha)
        return f_out

    def TF_goc(
        self,
        Fe,
        Fi,
        Fe_ext,
        Fi_ext,
        W,
        P,
        Qe_gr,
        Qi,
        Te_gr,
        Ti,
        Ee,
        Ei,
        Gl,
        Cm,
        El,
        Ke_grc,
        Ki,
        Qe_ext,
        Te_ext,
        Ke_ext,
        alpha,
        Ki_ext=0,
    ):
        """
        3D transfer functions
        https://github.com/RobertaMLo/CRBL_MF
        :return: result of transfer function
        """
        # mu_V, sigma_V, T_V = self.get_fluct_regime_vars(fe, fi, fe_ext, fi_ext, W, self.Q_e, self.tau_e, self.E_e,
        #                                                   self.Q_i, self.tau_i, self.E_i,
        #                                                   self.g_L, self.C_m, E_L, self.N_tot,
        #                                                   self.p_connect_e,self.p_connect_i, self.g,self.K_ext_e,self.K_ext_i)

        ### TO CHECK CHE ABBIA SENSO CORRISPONDENZA CON INPUT FUNCTION!
        # (fe, fi, fe_ext, fi_ext, W, self.P_goc, self.Q_grc_goc, self.Q_goc_goc, self.tau_grc_goc, self.tau_goc_goc,
        # self.E_i, self.E_i, self.g_L_goc, self.C_m_goc, self.E_L_goc, self.K_grc_goc, self.K_goc_goc,
        # self.Q_mf_goc, self.tau_mf_goc, self.K_mossy_goc)

        # Fe, Fi, Fe_ext, Fi_ext, XX, Qe_g, Te_g, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke_g, Ki, Ke_ext, Qe_ext, Te_ext, Ki_ext = 0

        mu_V, sigma_V, T_V, muGn = self.get_fluct_regime_vars_goc(
            Fe,
            Fi,
            Fe_ext,
            Fi_ext,
            W,
            Qe_gr,
            Te_gr,
            Ee,
            Qi,
            Ti,
            Ei,
            Gl,
            Cm,
            El,
            Ke_grc,
            Ki,
            Ke_ext,
            Qe_ext,
            Te_ext,
            Ki_ext=0,
        )

        V_thre = self.threshold_func(
            mu_V, sigma_V, T_V, muGn, P[0], P[1], P[2], P[3], P[4]
        )
        V_thre *= 1e3  # the threshold need to be in mv and not in Volt
        f_out = self.estimate_firing_rate(mu_V, sigma_V, T_V, V_thre, Gl, Cm, alpha)
        return f_out

    @staticmethod
    def get_fluct_regime_vars(
        Fe,
        Fi,
        Fe_ext,
        Fi_ext,
        XX,
        Q_e,
        tau_e,
        Ee,
        Q_i,
        tau_i,
        Ei,
        Gl,
        Cm,
        El,
        Ke,
        Ki,
        K_ext_e=0.0,
        tau_ext_e=0.0,
        Q_ext_e=0.0,
        tau_ext_i=0.0,
        Q_ext_i=0.0,
        K_ext_i=0.0,
    ):
        # COMMENTO GENERALE PER CHI PASSO IN INPUT A QUESTA FUNZIONE:
        # Ke e Ki sono i miei K standard di connettività. K_ext_e è quello che arriva da esterno. Per me in teoria li hanno solo i granuli e le golgi e sono eccitatori. Setto Ke_ext_i fisso = 0.
        """
        Compute the mean characteristic of neurons.
        Repository :
        https://github.com/RobertaMLo/CRBL_MF

        :return: mean and variance of membrane voltage of neurons and autocorrelation time constant
        """
        # firing rate
        # 1e-6 represent spontaneous release of synaptic neurotransmitter or some intrinsic currents of neurons
        fe = (Fe + 1.0e-6) + Fe_ext
        fi = (Fi + 1.0e-6) + Fi_ext

        # conductance fluctuation and effective membrane time constant
        # # stantard
        # mu_Ge, mu_Gi = Q_e*tau_e*fe*Ke + Q_ext_e*tau_ext_e*Fe_ext*K_ext_e, Q_i*tau_i*fi*Ki + Q_ext_i*tau_ext_i*Fi_ext*K_ext_i

        # # to include parallel:
        mu_Ge, mu_Gi = (
            Q_e * tau_e * fe * Ke + Q_e * tau_e * Fe_ext * Ke,
            Q_i * tau_i * fi * Ki + Q_ext_i * tau_ext_i * Fi_ext * K_ext_i,
        )

        mu_G = Gl + mu_Ge + mu_Gi

        # membrane potential
        mu_V = (np.e * (mu_Ge * Ee + mu_Gi * Ei + Gl * El) - XX) / mu_G
        muGn, Tm = mu_G / Gl, Cm / mu_G  # normalization

        # post-synaptic membrane potential event s around muV
        Ue, Ui = Q_e / mu_G * (Ee - mu_V), Q_i / mu_G * (Ei - mu_V)  # EQUAL TO EXP

        # Standard deviation of the fluctuations
        # Eqns 8 from [MV_2018]
        if np.any(tau_e <= 0) or np.any(tau_i <= 0):
            raise ValueError("tau values must be positive")
        sVe = (
            (2 * Tm + tau_e) * ((np.e * Ue * tau_e) / (2 * (tau_e + Tm))) ** 2 * Ke * fe
        )
        sVi = (
            (2 * Tm + tau_i) * ((np.e * Ui * tau_i) / (2 * (tau_i + Tm))) ** 2 * Ki * fi
        )
        sigma_V = np.sqrt(sVe + sVi)  # THIS IS MY SIGMA_V

        fe, fi = fe + 1e-9, fi + 1e-9  # just to insure a non zero division

        # Autocorrelation-time of the fluctuations Eqns 9 from [MV_2018]
        Tv_num = (
            Ke * fe * Ue**2 * tau_e**2 * np.e**2 + Ki * fi * Ui**2 * tau_i**2 * np.e**2
        )
        Tv = 0.5 * Tv_num / ((sigma_V + 1e-20) ** 2)

        T_V = Tv * Gl / Cm  # normalization. THIS IS MY TVN

        return mu_V, sigma_V, T_V, muGn

    @staticmethod
    def get_fluct_regime_vars_goc(
        Fe,
        Fi,
        Fe_ext,
        Fi_ext,
        XX,
        Qe_g,
        Te_g,
        Ee,
        Qi,
        Ti,
        Ei,
        Gl,
        Cm,
        El,
        Ke_g,
        Ki,
        Ke_ext,
        Qe_ext,
        Te_ext,
        Ki_ext=0,
    ):
        """
        Compute the mean characteristic of neurons.
        Repository :
        https://github.com/RobertaMLo/CRBL_MF

        :return: mean and variance of membrane voltage of neurons and autocorrelation time constant
        """
        # firing rate
        # 1e-6 represent spontaneous release of synaptic neurotransmitter or some intrinsic currents of neurons
        fe_g = Fe + 1e-6
        fe_m = Fe_ext
        fi = Fi + 1e-6 + Fi_ext

        # conductance fluctuation and effective membrane time constant
        # ---------------------------- Pop cond:  mu GrC and MLI ---------------------------------------
        muGe_g, muGe_m, muGi = (
            Qe_g * Ke_g * Te_g * fe_g,
            Qe_ext * Ke_ext * Te_ext * fe_m,
            Qi * Ki * Ti * fi,
        )  # EQUAL TO EXP
        # ---------------------------- Input cond:  mu PC -----------------------------------------------
        muG = Gl + muGe_g + muGe_m + muGi  # EQUAL TO EXP
        # ---------------------------- Membrane Fluctuation Properties ----------------------------------
        mu_V = (
            np.e * (muGe_g * Ee + muGe_m * Ee + muGi * Ei + Gl * El) - XX
        ) / muG  # XX = adaptation

        muGn, Tm = muG / Gl, Cm / muG  # normalization

        Ue_g, Ue_m, Ui = (
            Qe_g / muG * (Ee - mu_V),
            Qe_ext / muG * (Ee - mu_V),
            Qi / muG * (Ei - mu_V),
        )  # EQUAL TO EXP

        if np.any(Te_g <= 0) or np.any(Ti <= 0):
            raise ValueError("tau values must be positive")
        sVe_g = (
            (2 * Tm + Te_g)
            * ((np.e * Ue_g * Te_g) / (2 * (Te_g + Tm))) ** 2
            * Ke_g
            * fe_g
        )
        sVe_m = (
            (2 * Tm + Te_ext)
            * ((np.e * Ue_m * Te_ext) / (2 * (Te_ext + Tm))) ** 2
            * Ke_ext
            * fe_m
        )
        sVi = (2 * Tm + Ti) * ((np.e * Ui * Ti) / (2 * (Ti + Tm))) ** 2 * Ki * fi

        sigma_V = np.sqrt(sVe_g + sVe_m + sVi)  # SV IN MIEI CODICI

        fe_m, fe_g, fi = (
            fe_m + 1e-15,
            fe_g + 1e-15,
            fi + 1e-15,
        )  # just to insure a non zero division

        Tv_num = (
            Ke_g * fe_g * Ue_g**2 * Te_g**2 * np.e**2
            + Ke_ext * fe_m * Ue_m**2 * Te_ext**2 * np.e**2
            + Ki * fi * Ui**2 * Ti**2 * np.e**2
        )
        Tv = 0.5 * Tv_num / ((sigma_V + 1e-20) ** 2)

        T_V = Tv * Gl / Cm  # normalization TVN IN MIEI CODICI

        return mu_V, sigma_V, T_V, muGn

    @staticmethod
    def threshold_func(muV, sigmaV, TvN, muGn, P0, P1, P2, P3, P4):
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
        TvN0, DTvN0 = 0.5, 1.0
        V = (muV - muV0) / DmuV0
        S = (sigmaV - sV0) / DsV0
        T = (TvN - TvN0) / DTvN0

        return P0 + P1 * V + P2 * S + P3 * T + P4 * np.log(muGn)

    @staticmethod
    def estimate_firing_rate(muV, sV, TvN, Vthre, Gl, Cm, alpha):
        """
        The threshold function of the neurons
        :param muV: mean of membrane voltage
        :param sigmaV: variance of membrane voltage
        :param Tv: autocorrelation time constant
        :param Vthre:threshold of neurons
        """
        # Eqns 10 from [MV_2018]
        return (
            0.5
            / TvN
            * Gl
            / Cm
            * (sp_spec.erfc((Vthre - muV) / np.sqrt(2) / sV))
            * alpha
        )
