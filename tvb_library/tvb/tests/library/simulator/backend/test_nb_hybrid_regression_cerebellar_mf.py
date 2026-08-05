import numpy as np

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import EulerDeterministic, HeunDeterministic
from tvb.simulator.models.cerebellar_mf import CerebellarMF


N_NODES = 4
DT = 0.02


def _column(values):
    return np.asarray(values, dtype=np.float64)[:, np.newaxis]


def _model():
    model = CerebellarMF()
    model.variables_of_interest = tuple(model.state_variables)
    model.use_legacy_goc_e_e = np.array([False])

    # These are intentionally node-specific. In particular, the three mossy
    # GoC parameters have different scales so neither indexing nor permutation
    # errors can accidentally produce the reference trajectory.
    model.T = _column([2.8, 4.1, 5.7, 7.6])
    model.tau_OU = _column([2.2, 3.1, 4.4, 6.0])
    model.weight_noise = _column([0.0010, 0.0025, 0.0045, 0.0070])
    model.external_input_ex_ex = _column([0.012, 0.028, 0.047, 0.071])
    model.external_input_ex_in = _column([0.001, 0.003, 0.006, 0.010])
    model.external_input_in_ex = _column([0.018, 0.039, 0.064, 0.095])
    model.Q_mf_goc = _column([0.31, 0.39, 0.48, 0.58])
    model.tau_mf_goc = _column([2.4, 3.2, 4.3, 5.6])
    model.K_mossy_goc = _column([23.0, 37.0, 54.0, 76.0])
    model.alpha_grc = _column([1.5, 1.8, 2.2, 2.7])
    model.alpha_goc = _column([0.8, 1.1, 1.5, 2.0])
    model.alpha_mli = _column([3.8, 4.5, 5.3, 6.2])
    model.alpha_pc = _column([3.5, 4.4, 5.5, 6.8])
    return model


def _network(integrator):
    subnet = Subnetwork(
        name="cerebellar_regression",
        model=_model(),
        scheme=integrator,
        nnodes=N_NODES,
    )
    subnet.projections = []
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    return network


def _initial_state():
    state = np.zeros((5, N_NODES, 1), dtype=np.float64)
    state[0, :, 0] = [0.045, 0.071, 0.103, 0.142]
    state[1, :, 0] = [0.012, 0.019, 0.029, 0.043]
    state[2, :, 0] = [0.083, 0.116, 0.154, 0.201]
    state[3, :, 0] = [0.064, 0.092, 0.128, 0.173]
    state[4, :, 0] = [-0.30, 0.17, 0.42, -0.11]
    return state


def _python_trajectory(network, initial_state, nstep):
    states = network.States(initial_state.copy())
    network.init_projection_buffers(states)
    trajectory = []
    for step in range(1, nstep + 1):
        states = network.step(step, states)
        trajectory.append(np.asarray(states[0]).copy())
    return np.stack(trajectory)


def _numba_run(network, initial_state, nstep):
    compiled = NbHybridBackend().compile(network, eager=True)
    outputs, snapshot = compiled.run(
        nstep=nstep,
        chunk_size=1,
        initial_states=[initial_state.copy()],
        return_snapshot=True,
    )
    return outputs[0][1], snapshot["states"][0]


def _assert_all_nodes_close(actual, expected, *, rtol, atol, message):
    assert actual.shape == expected.shape
    for node in range(N_NODES):
        np.testing.assert_allclose(
            actual[..., node, :],
            expected[..., node, :],
            rtol=rtol,
            atol=atol,
            err_msg=f"{message} at node {node}",
        )


def test_goc_wrapper_maps_external_mossy_parameters_in_signature_order():
    model = CerebellarMF()
    model.use_legacy_goc_e_e = np.array([False])
    model.Q_mf_goc = np.array([0.41])
    model.tau_mf_goc = np.array([2.7])
    model.K_mossy_goc = np.array([37.0])

    common = (
        0.08,
        0.015,
        0.06,
        0.0,
        0.0,
        model.P_goc,
        model.Q_grc_goc,
        model.Q_goc_goc,
        model.tau_grc_goc,
        model.tau_goc_goc,
        model.E_e,
        model.E_i,
        model.g_L_goc,
        model.C_m_goc,
        model.E_L_goc,
        model.K_grc_goc,
        model.K_goc_goc,
    )
    expected = model._TF_goc(
        *common,
        model.K_mossy_goc,
        model.Q_mf_goc,
        model.tau_mf_goc,
        model.alpha_goc,
    )
    permuted = model._TF_goc(
        *common,
        model.Q_mf_goc,
        model.tau_mf_goc,
        model.K_mossy_goc,
        model.alpha_goc,
    )
    actual = model.TF_inhibitory_goc(0.08, 0.015, 0.06, 0.0)

    assert abs(float(expected[0]) - float(permuted[0])) > 0.05
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_heterogeneous_cerebellar_derivative_matches_numba_at_every_node():
    initial = _initial_state()
    python_network = _network(EulerDeterministic(dt=DT))
    numba_network = _network(EulerDeterministic(dt=DT))
    coupling = python_network.subnets[0].zero_cvars()
    python_derivative = python_network.subnets[0].model.dfun(initial, coupling)

    _, numba_state = _numba_run(numba_network, initial, nstep=1)
    numba_derivative = (numba_state - initial.astype(np.float32)) / DT

    assert np.ptp(python_derivative[1, :, 0]) > 1e-3
    _assert_all_nodes_close(
        numba_derivative,
        python_derivative,
        rtol=8e-5,
        atol=3e-5,
        message="Euler-derived CerebellarMF derivative differs from Python",
    )


def test_heterogeneous_cerebellar_heun_trajectory_matches_at_every_node():
    initial = _initial_state()
    nstep = 12
    python = _python_trajectory(
        _network(HeunDeterministic(dt=DT)), initial, nstep=nstep
    )
    numba, final_state = _numba_run(
        _network(HeunDeterministic(dt=DT)), initial, nstep=nstep
    )

    assert np.all(np.isfinite(python))
    assert np.ptp(python[-1, 1, :, 0]) > 1e-2
    _assert_all_nodes_close(
        numba,
        python,
        rtol=5e-5,
        atol=5e-6,
        message="Heun CerebellarMF trajectory differs from Python",
    )
    _assert_all_nodes_close(
        final_state,
        python[-1],
        rtol=5e-5,
        atol=5e-6,
        message="Heun CerebellarMF final state differs from Python",
    )
