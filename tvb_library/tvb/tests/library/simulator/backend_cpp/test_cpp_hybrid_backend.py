# -*- coding: utf-8 -*-
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
#  analysers necessary to run brain-simulations. You can use it stand alone or
#  in conjunction with TheVirtualBrain-Framework Package. See content of the
#  documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
#

import os
import tempfile
import unittest

import numpy as np

os.environ.setdefault("TVB_USER_HOME", os.path.join(tempfile.gettempdir(), "tvb-user"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.backend_cpp import CppHybridBackend, DelayedSelfFeedbackConfig
from tvb.simulator.hybrid import NetworkSet, Simulator, Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import TemporalAverage


DT = 0.1


class ScopedNbHybridBackend(NbHybridBackend):
    """Limit Numba comparison to the currently supported native C++ path."""

    def _check_compatibility(self, network_set: NetworkSet) -> None:
        if not network_set.subnets:
            raise ValueError("NetworkSet must contain at least one subnetwork.")
        dt0 = float(network_set.subnets[0].scheme.dt)
        for subnet in network_set.subnets:
            if not isinstance(subnet.model, MontbrioPazoRoxin):
                raise NotImplementedError(
                    "ScopedNbHybridBackend comparison path supports only MontbrioPazoRoxin."
                )
            if not isinstance(subnet.scheme, HeunDeterministic):
                raise NotImplementedError(
                    "ScopedNbHybridBackend comparison path supports only HeunDeterministic."
                )
            if float(subnet.scheme.dt) != dt0:
                raise ValueError("All subnetworks must share the same dt.")


def _make_subnet(name: str, n_nodes: int) -> Subnetwork:
    model = MontbrioPazoRoxin(I=np.array([2.0]))
    model.configure()
    subnet = Subnetwork(
        name=name,
        model=model,
        scheme=HeunDeterministic(dt=DT),
        nnodes=n_nodes,
    ).configure()
    subnet.node_indices = np.arange(subnet.nnodes)
    return subnet


def _make_network(n_nodes: int) -> tuple[NetworkSet, Subnetwork]:
    subnet = _make_subnet("sn", n_nodes)
    network = NetworkSet(subnets=[subnet], projections=[], stimuli=[])
    network.configure()
    return network, subnet


def _make_initial_state(subnetwork: Subnetwork) -> np.ndarray:
    model = subnetwork.model
    x0 = np.zeros(
        (model.nvar, subnetwork.nnodes, model.number_of_modes),
        dtype=np.float64,
    )
    for i, state_var in enumerate(model.state_variables):
        if state_var not in model.state_variable_range:
            continue
        low, high = map(float, model.state_variable_range[state_var])
        x0[i, :, :] = (low + high) / 2.0
    return x0


def _run_python(
    network: NetworkSet,
    initial_state: np.ndarray,
    simulation_length: float,
    tavg_period: float,
) -> tuple[np.ndarray, np.ndarray]:
    sim = Simulator(
        nets=network,
        simulation_length=simulation_length,
        monitors=[TemporalAverage(period=tavg_period)],
    )
    sim.configure()
    ((times, data),) = sim.run(initial_conditions=[initial_state.copy()])
    return np.asarray(times, dtype=np.float64), np.asarray(data, dtype=np.float64)


def _run_numba(
    network: NetworkSet,
    initial_state: np.ndarray,
    nstep: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    backend = ScopedNbHybridBackend()
    results = backend.run_network(
        network,
        nstep=nstep,
        chunk_size=chunk_size,
        initial_states=[initial_state.copy()],
    )
    times, data, _ctavg = results[0]
    return np.asarray(times, dtype=np.float64), np.asarray(data, dtype=np.float64)


def _run_native(
    network: NetworkSet,
    initial_state: np.ndarray,
    nstep: int,
    chunk_size: int,
    build_root: str,
    delayed_self_feedback: DelayedSelfFeedbackConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    backend = CppHybridBackend(build_root=build_root)
    compiled = backend.compile(
        network,
        monitors=[TemporalAverage(period=chunk_size * DT)],
        user_source_hint="test_cpp_hybrid_backend",
        delayed_self_feedback=delayed_self_feedback,
    )
    times, data = compiled.run(
        initial_states=[initial_state.copy()],
        nstep=nstep,
        chunk_size=chunk_size,
    )
    return np.asarray(times, dtype=np.float64), np.asarray(data, dtype=np.float64)


def _run_python_delayed_self_feedback(
    subnetwork: Subnetwork,
    initial_state: np.ndarray,
    nstep: int,
    chunk_size: int,
    delay_steps: int,
    gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = subnetwork.model
    tau = float(np.atleast_1d(model.tau)[0])
    delta = float(np.atleast_1d(model.Delta)[0])
    eta = float(np.atleast_1d(model.eta)[0])
    j = float(np.atleast_1d(model.J)[0])
    current_i = float(np.atleast_1d(model.I)[0])
    cr = float(np.atleast_1d(model.cr)[0])
    cv = float(np.atleast_1d(model.cv)[0])
    n_nodes = subnetwork.nnodes

    def compute_dfun(state: np.ndarray, delayed_r: np.ndarray) -> np.ndarray:
        r = np.maximum(0.0, state[0, :, 0])
        v = state[1, :, 0]
        dx = np.zeros_like(state)
        dx[0, :, 0] = (1.0 / tau) * (delta / (np.pi * tau) + 2.0 * v * r)
        dx[1, :, 0] = (1.0 / tau) * (
            v * v
            - (np.pi * np.pi) * tau * tau * r * r
            + eta
            + j * tau * r
            + current_i
            + cr * 0.0
            + cv * (gain * delayed_r)
        )
        return dx

    history = [initial_state.copy() for _ in range(delay_steps + 1)]
    state = initial_state.copy()
    num_chunks = (nstep + chunk_size - 1) // chunk_size
    times = np.zeros(num_chunks, dtype=np.float64)
    data = np.zeros((num_chunks, model.nvar, n_nodes, 1), dtype=np.float64)
    accum = np.zeros((model.nvar, n_nodes, 1), dtype=np.float64)
    current_chunk = 0
    steps_in_chunk = 0
    chunk_start_step = 1

    for step in range(1, nstep + 1):
        delayed_state = history[-1 - delay_steps]
        delayed_r = delayed_state[0, :, 0]

        dx0 = compute_dfun(state, delayed_r)
        predictor = state + DT * dx0
        predictor[0, :, 0] = np.maximum(0.0, predictor[0, :, 0])

        dx1 = compute_dfun(predictor, delayed_r)
        state = state + 0.5 * DT * (dx0 + dx1)
        state[0, :, 0] = np.maximum(0.0, state[0, :, 0])
        history.append(state.copy())
        if len(history) > delay_steps + 1:
            history.pop(0)

        accum += state
        steps_in_chunk += 1
        close_chunk = steps_in_chunk == chunk_size or step == nstep
        if not close_chunk:
            continue

        mid_step = chunk_start_step + (steps_in_chunk - 1.0) / 2.0
        times[current_chunk] = mid_step * DT
        data[current_chunk] = accum / float(steps_in_chunk)
        accum.fill(0.0)
        current_chunk += 1
        chunk_start_step = step + 1
        steps_in_chunk = 0

    return times, data


class TestCppHybridBackend(unittest.TestCase):
    def test_compile_emits_runtime_header_and_statebuffer(self):
        network, subnet = _make_network(2)
        initial_state = _make_initial_state(subnet)

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            backend = CppHybridBackend(build_root=build_root)
            compiled = backend.compile(
                network,
                monitors=[TemporalAverage(period=0.2)],
                user_source_hint="test_compile_emits_runtime_header_and_statebuffer",
            )

            generated_cpp = compiled.generated_cpp_path.read_text(encoding="utf-8")
            runtime_header = compiled.generated_source.runtime_header_path.read_text(
                encoding="utf-8"
            )

            self.assertIn('#include "runtime/runtime.hpp"', generated_cpp)
            self.assertIn(
                "tvb::hybrid::runtime::describe<GeneratedModel>()", generated_cpp
            )
            self.assertIn(
                "tvb::hybrid::runtime::run_simulation<GeneratedModel>(",
                generated_cpp,
            )
            self.assertIn("class StateBuffer", runtime_header)
            self.assertIn("class HistoryBuffer", runtime_header)
            self.assertIn("const StateBuffer& read", runtime_header)
            self.assertIn("double read_value", runtime_header)
            self.assertIn("inline double delayed_state_value", runtime_header)
            self.assertIn("inline void heun_step", runtime_header)
            self.assertIn("inline SimulationResult run_simulation", runtime_header)

            history_probe = compiled.load_module().debug_probe_history()
            self.assertEqual(history_probe["capacity"], 3)
            self.assertEqual(history_probe["size"], 3)
            self.assertEqual(history_probe["delay_0"], 40.0)
            self.assertEqual(history_probe["delay_1"], 30.0)
            self.assertEqual(history_probe["delay_2"], 20.0)
            self.assertEqual(history_probe["helper_delay_0"], 40.0)
            self.assertEqual(history_probe["helper_delay_1"], 30.0)
            self.assertEqual(history_probe["helper_delay_2"], 20.0)

            times, data = compiled.run(
                initial_states=[initial_state],
                nstep=4,
                chunk_size=2,
            )
            self.assertEqual(times.shape, (2,))
            self.assertEqual(data.shape, (2, 2, 2, 1))

    def test_single_mpr_matches_python_and_numba_references(self):
        simulation_length = 2.0
        chunk_size = 2
        tavg_period = chunk_size * DT
        nstep = int(round(simulation_length / DT))

        network, subnet = _make_network(3)
        initial_state = _make_initial_state(subnet)

        py_times, py_data = _run_python(
            network=network,
            initial_state=initial_state,
            simulation_length=simulation_length,
            tavg_period=tavg_period,
        )
        nb_times, nb_data = _run_numba(
            network=network,
            initial_state=initial_state,
            nstep=nstep,
            chunk_size=chunk_size,
        )
        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            native_times, native_data = _run_native(
                network=network,
                initial_state=initial_state,
                nstep=nstep,
                chunk_size=chunk_size,
                build_root=build_root,
            )

        self.assertEqual(py_data.shape, native_data.shape)
        self.assertEqual(nb_data.shape, native_data.shape)
        self.assertEqual(py_times.shape, native_times.shape)
        self.assertEqual(nb_times.shape, native_times.shape)

        np.testing.assert_allclose(native_data, py_data, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(native_data, nb_data, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(native_times, nb_times, rtol=1e-6, atol=1e-6)

        # The current native runtime timestamps chunk midpoints, while the Python
        # monitor path reports chunk endpoints for this scenario.
        np.testing.assert_allclose(py_times - native_times, -0.5 * DT, atol=1e-12)

    def test_delayed_self_feedback_matches_python_reference(self):
        chunk_size = 2
        nstep = 12
        delay_steps = 3
        gain = 0.25

        network, subnet = _make_network(3)
        initial_state = _make_initial_state(subnet)
        py_times, py_data = _run_python_delayed_self_feedback(
            subnetwork=subnet,
            initial_state=initial_state,
            nstep=nstep,
            chunk_size=chunk_size,
            delay_steps=delay_steps,
            gain=gain,
        )

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            native_times, native_data = _run_native(
                network=network,
                initial_state=initial_state,
                nstep=nstep,
                chunk_size=chunk_size,
                build_root=build_root,
                delayed_self_feedback=DelayedSelfFeedbackConfig(
                    delay_steps=delay_steps,
                    gain=gain,
                    source_state_var="r",
                    target_state_var="V",
                ),
            )

        np.testing.assert_allclose(native_times, py_times, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(native_data, py_data, rtol=1e-12, atol=1e-12)
