# -*- coding: utf-8 -*-

import importlib.machinery
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.sparse as sp

os.environ.setdefault("TVB_USER_HOME", os.path.join(tempfile.gettempdir(), "tvb-user"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

from tvb.datatypes import equations, patterns
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.backend_cpp import CppHybridBackend
from tvb.simulator.hybrid import (
    InterProjection,
    IntraProjection,
    NetworkSet,
    Simulator,
    Subnetwork,
)
from tvb.simulator.integrators import (
    EulerDeterministic,
    EulerStochastic,
    HeunDeterministic,
    HeunStochastic,
)
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.noise import Additive
from tvb.simulator.monitors import (
    AfferentCoupling,
    AfferentCouplingTemporalAverage,
    Bold,
    Raw,
    RawVoi,
    SubSample,
    TemporalAverage,
)


DT = 0.1


class ScopedNbHybridBackend(NbHybridBackend):
    """Limit Numba comparison to the currently supported native C++ path."""

    def _check_compatibility(self, network_set) -> None:
        if not network_set.subnets:
            raise ValueError("NetworkSet must contain at least one subnetwork.")
        dt0 = float(network_set.subnets[0].scheme.dt)
        for subnet in network_set.subnets:
            if not isinstance(subnet.model, MontbrioPazoRoxin):
                raise NotImplementedError(
                    "ScopedNbHybridBackend comparison path supports only MontbrioPazoRoxin."
                )
            if not isinstance(subnet.scheme, (EulerDeterministic, HeunDeterministic)):
                raise NotImplementedError(
                    "ScopedNbHybridBackend comparison path supports only deterministic Euler/Heun."
                )
            if float(subnet.scheme.dt) != dt0:
                raise ValueError("All subnetworks must share the same dt.")


def _make_subnet(
    name: str,
    n_nodes: int,
    projections=None,
    scheme=None,
) -> Subnetwork:
    model = MontbrioPazoRoxin(I=np.array([2.0]))
    model.configure()
    subnet = Subnetwork(
        name=name,
        model=model,
        scheme=scheme or HeunDeterministic(dt=DT),
        nnodes=n_nodes,
        projections=projections or [],
    ).configure()
    subnet.node_indices = np.arange(n_nodes)
    return subnet


def _make_network(
    n_nodes: int,
    projections=None,
    scheme=None,
) -> tuple[NetworkSet, Subnetwork]:
    subnet = _make_subnet("sn", n_nodes, projections=projections, scheme=scheme)
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    return network, subnet


def _make_stochastic_network(
    n_nodes: int,
    nsig: np.ndarray,
    noise_seed: int = 1234,
    integrator_cls=HeunStochastic,
) -> tuple[NetworkSet, Subnetwork]:
    model = MontbrioPazoRoxin(I=np.array([2.0]))
    model.configure()
    subnet = Subnetwork(
        name="sn",
        model=model,
        scheme=integrator_cls(
            dt=DT,
            noise=Additive(nsig=np.asarray(nsig, dtype=np.float64), noise_seed=noise_seed),
        ),
        nnodes=n_nodes,
        projections=[],
    ).configure()
    subnet.node_indices = np.arange(n_nodes)
    network = NetworkSet(subnets=[subnet], projections=[])
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
) -> tuple[np.ndarray, np.ndarray]:
    backend = CppHybridBackend(build_root=build_root)
    compiled = backend.compile(
        network,
        monitors=[TemporalAverage(period=chunk_size * DT)],
        user_source_hint="test_cpp_hybrid_backend",
    )
    ((times, data),) = compiled.run(
        initial_states=[initial_state.copy()],
        nstep=nstep,
        chunk_size=chunk_size,
    )
    return np.asarray(times, dtype=np.float64), np.asarray(data, dtype=np.float64)


class TestCppHybridBackend(unittest.TestCase):
    def test_compile_reuses_cached_extension_without_regenerating_or_building(self):
        network, _subnet = _make_network(2)
        monitors = [TemporalAverage(period=0.2)]
        source_hint = "test_compile_reuses_cached_extension"

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            backend = CppHybridBackend(build_root=build_root)
            lowering = backend.lower(
                network,
                monitors=monitors,
                user_source_hint=source_hint,
            )
            cache_key = lowering.spec.cache_key()
            module_name = f"tvb_hybrid_cpp_{cache_key[:16]}"
            build_dir = Path(build_root) / module_name
            build_dir.mkdir(parents=True)
            extension_path = build_dir / (
                module_name + importlib.machinery.EXTENSION_SUFFIXES[0]
            )
            extension_path.touch()

            with (
                mock.patch(
                    "tvb.simulator.backend_cpp.backend.generate_cpp_source",
                    side_effect=AssertionError("cache hit should not regenerate source"),
                ) as generate_cpp_source,
                mock.patch(
                    "tvb.simulator.backend_cpp.backend.build_generated_extension",
                    side_effect=AssertionError("cache hit should not invoke native build"),
                ) as build_generated_extension,
            ):
                compiled = backend.compile(
                    network,
                    monitors=monitors,
                    user_source_hint=source_hint,
                    build_native=True,
                )

            generate_cpp_source.assert_not_called()
            build_generated_extension.assert_not_called()
            self.assertEqual(compiled.pipeline_stage, "extension_cached")
            self.assertEqual(compiled.module_name, module_name)
            self.assertEqual(compiled.generated_source.extension_path, extension_path)
            self.assertEqual(compiled.debug_summary()["cache_key"], cache_key)

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
            self.assertIn("inline SimulationMetadata describe()", generated_cpp)
            self.assertIn(
                "inline std::vector<SimulationResult> run_simulation(",
                generated_cpp,
            )
            self.assertIn("kNumCouplingVars", generated_cpp)
            self.assertIn("class StateBuffer", runtime_header)
            self.assertIn("class HistoryBuffer", runtime_header)
            self.assertIn("double read_value", runtime_header)
            self.assertIn("std::vector<double> data_", runtime_header)
            self.assertIn("struct ProjectionArrays", runtime_header)
            self.assertIn("accumulate_projection", runtime_header)
            self.assertIn("inline void heun_step", runtime_header)

            history_probe = compiled.load_module().debug_probe_history()
            self.assertEqual(history_probe["capacity"], 3)
            self.assertEqual(history_probe["size"], 3)
            self.assertEqual(history_probe["delay_0"], 40.0)
            self.assertEqual(history_probe["delay_1"], 30.0)
            self.assertEqual(history_probe["delay_2"], 20.0)

            ((times, data),) = compiled.run(
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

        # Timestamp convention: native and Numba both record the chunk midpoint
        # counting steps from 1 (see runtime.hpp run_simulation comments).
        # Python's TemporalAverage counts from step 0, so its midpoint is
        # exactly 0.5 * dt earlier: python_time == native_time - 0.5 * dt.
        np.testing.assert_allclose(py_times - native_times, -0.5 * DT, atol=1e-12)

    def test_euler_deterministic_matches_python_and_numba_references(self):
        simulation_length = 2.0
        chunk_size = 2
        tavg_period = chunk_size * DT
        nstep = int(round(simulation_length / DT))

        network, subnet = _make_network(3, scheme=EulerDeterministic(dt=DT))
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
        np.testing.assert_allclose(py_times - native_times, -0.5 * DT, atol=1e-12)

    def test_deterministic_reproducibility(self):
        network, subnet = _make_network(4)
        initial_state = _make_initial_state(subnet)
        nstep = 10
        chunk_size = 2

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            backend = CppHybridBackend(build_root=build_root)
            compiled = backend.compile(
                network,
                monitors=[TemporalAverage(period=chunk_size * DT)],
                user_source_hint="test_deterministic_reproducibility",
            )
            ((times1, data1),) = compiled.run(
                initial_states=[initial_state.copy()], nstep=nstep, chunk_size=chunk_size
            )
            ((times2, data2),) = compiled.run(
                initial_states=[initial_state.copy()], nstep=nstep, chunk_size=chunk_size
            )

        np.testing.assert_array_equal(times1, times2)
        np.testing.assert_array_equal(data1, data2)

    def test_heun_stochastic_zero_noise_matches_deterministic(self):
        nstep = 5
        chunk_size = 1
        det_network, det_subnet = _make_network(2)
        stoch_network, stoch_subnet = _make_stochastic_network(
            2, nsig=np.array([0.0, 0.0])
        )
        det_initial_state = _make_initial_state(det_subnet)
        stoch_initial_state = _make_initial_state(stoch_subnet)

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            det_backend = CppHybridBackend(build_root=build_root)
            det_compiled = det_backend.compile(
                det_network,
                monitors=[Raw()],
                user_source_hint="test_heun_stochastic_zero_noise_deterministic",
            )
            stoch_backend = CppHybridBackend(build_root=build_root)
            stoch_compiled = stoch_backend.compile(
                stoch_network,
                monitors=[Raw()],
                user_source_hint="test_heun_stochastic_zero_noise_stochastic",
            )

            ((det_times, det_data),) = det_compiled.run(
                initial_states=[det_initial_state.copy()],
                nstep=nstep,
                chunk_size=chunk_size,
            )
            ((stoch_times, stoch_data),) = stoch_compiled.run(
                initial_states=[stoch_initial_state.copy()],
                nstep=nstep,
                chunk_size=chunk_size,
            )

            generated_cpp = stoch_compiled.generated_cpp_path.read_text(
                encoding="utf-8"
            )
            runtime_header = stoch_compiled.generated_source.runtime_header_path.read_text(
                encoding="utf-8"
            )

        self.assertIn("heun_step_stochastic<SubnetModel_0>", generated_cpp)
        self.assertIn("inline void heun_step_stochastic", runtime_header)
        np.testing.assert_allclose(stoch_times, det_times, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(stoch_data, det_data, rtol=1e-12, atol=1e-12)

    def test_euler_stochastic_zero_noise_matches_deterministic(self):
        nstep = 5
        det_network, det_subnet = _make_network(2, scheme=EulerDeterministic(dt=DT))
        stoch_network, stoch_subnet = _make_stochastic_network(
            2, nsig=np.array([0.0, 0.0]), integrator_cls=EulerStochastic
        )
        det_initial_state = _make_initial_state(det_subnet)
        stoch_initial_state = _make_initial_state(stoch_subnet)

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            det_backend = CppHybridBackend(build_root=build_root)
            det_compiled = det_backend.compile(
                det_network,
                monitors=[Raw()],
                user_source_hint="test_euler_stochastic_zero_noise_deterministic",
            )
            stoch_backend = CppHybridBackend(build_root=build_root)
            stoch_compiled = stoch_backend.compile(
                stoch_network,
                monitors=[Raw()],
                user_source_hint="test_euler_stochastic_zero_noise_stochastic",
            )

            ((det_times, det_data),) = det_compiled.run(
                initial_states=[det_initial_state.copy()],
                nstep=nstep,
                chunk_size=1,
            )
            ((stoch_times, stoch_data),) = stoch_compiled.run(
                initial_states=[stoch_initial_state.copy()],
                nstep=nstep,
                chunk_size=1,
            )

            generated_cpp = stoch_compiled.generated_cpp_path.read_text(
                encoding="utf-8"
            )
            runtime_header = stoch_compiled.generated_source.runtime_header_path.read_text(
                encoding="utf-8"
            )

        self.assertIn("euler_step_stochastic<SubnetModel_0>", generated_cpp)
        self.assertIn("inline void euler_step_stochastic", runtime_header)
        np.testing.assert_allclose(stoch_times, det_times, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(stoch_data, det_data, rtol=1e-12, atol=1e-12)

    def test_heun_stochastic_noise_is_seed_reproducible_after_reset(self):
        nstep = 5
        network, subnet = _make_stochastic_network(
            2, nsig=np.array([0.01, 0.01]), noise_seed=123
        )
        initial_state = _make_initial_state(subnet)

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            backend = CppHybridBackend(build_root=build_root)
            compiled = backend.compile(
                network,
                monitors=[Raw()],
                user_source_hint="test_heun_stochastic_noise_reproducible",
            )

            ((times1, data1),) = compiled.run(
                initial_states=[initial_state.copy()],
                nstep=nstep,
                chunk_size=1,
            )
            subnet.scheme.noise.reset_random_stream()
            ((times2, data2),) = compiled.run(
                initial_states=[initial_state.copy()],
                nstep=nstep,
                chunk_size=1,
            )

        np.testing.assert_array_equal(times1, times2)
        np.testing.assert_array_equal(data1, data2)

        det_network, det_subnet = _make_network(2)
        det_initial_state = _make_initial_state(det_subnet)
        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            det_backend = CppHybridBackend(build_root=build_root)
            det_compiled = det_backend.compile(
                det_network,
                monitors=[Raw()],
                user_source_hint="test_heun_stochastic_noise_deterministic_compare",
            )
            ((_det_times, det_data),) = det_compiled.run(
                initial_states=[det_initial_state.copy()],
                nstep=nstep,
                chunk_size=1,
            )

        self.assertGreater(float(np.max(np.abs(data1 - det_data))), 1e-6)

    def test_raw_monitor_forces_one_sample_per_step(self):
        network, subnet = _make_network(3)
        initial_state = _make_initial_state(subnet)
        nstep = 5

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            backend = CppHybridBackend(build_root=build_root)
            compiled = backend.compile(
                network,
                monitors=[Raw()],
                user_source_hint="test_raw_monitor_forces_one_sample_per_step",
            )
            ((times, data),) = compiled.run(
                initial_states=[initial_state.copy()],
                nstep=nstep,
                chunk_size=4,
            )

        self.assertEqual(times.shape, (nstep,))
        self.assertEqual(data.shape, (nstep, 2, subnet.nnodes, 1))
        np.testing.assert_allclose(times, DT * np.arange(1, nstep + 1))

    def test_rawvoi_monitor_forces_one_sample_per_step(self):
        network, subnet = _make_network(2)
        initial_state = _make_initial_state(subnet)
        nstep = 6

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            backend = CppHybridBackend(build_root=build_root)
            compiled = backend.compile(
                network,
                monitors=[RawVoi()],
                user_source_hint="test_rawvoi_monitor_forces_one_sample_per_step",
            )
            ((times, data),) = compiled.run(
                initial_states=[initial_state.copy()],
                nstep=nstep,
                chunk_size=3,
            )

        self.assertEqual(times.shape, (nstep,))
        self.assertEqual(data.shape, (nstep, 2, subnet.nnodes, 1))
        np.testing.assert_allclose(times, DT * np.arange(1, nstep + 1))

    def test_afferent_coupling_returns_ctavg_and_forces_one_sample_per_step(self):
        network, subnet = _make_network(3)
        initial_state = _make_initial_state(subnet)
        nstep = 5

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            backend = CppHybridBackend(build_root=build_root)
            compiled = backend.compile(
                network,
                monitors=[AfferentCoupling()],
                user_source_hint="test_afferent_coupling_returns_ctavg",
            )
            ((times, ctavg),) = compiled.run(
                initial_states=[initial_state.copy()],
                nstep=nstep,
                chunk_size=4,
            )

        self.assertEqual(times.shape, (nstep,))
        self.assertEqual(ctavg.shape, (nstep, 2, subnet.nnodes, 1))
        np.testing.assert_allclose(times, DT * np.arange(1, nstep + 1))
        np.testing.assert_array_equal(ctavg, np.zeros_like(ctavg))

    def test_afferent_coupling_temporal_average_uses_requested_chunk_size(self):
        network, subnet = _make_network(3)
        initial_state = _make_initial_state(subnet)
        nstep = 6
        chunk_size = 3

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            backend = CppHybridBackend(build_root=build_root)
            compiled = backend.compile(
                network,
                monitors=[AfferentCouplingTemporalAverage(period=chunk_size * DT)],
                user_source_hint="test_afferent_coupling_temporal_average",
            )
            ((times, ctavg),) = compiled.run(
                initial_states=[initial_state.copy()],
                nstep=nstep,
                chunk_size=chunk_size,
            )

        self.assertEqual(times.shape, (2,))
        self.assertEqual(ctavg.shape, (2, 2, subnet.nnodes, 1))
        np.testing.assert_allclose(times, np.array([0.2, 0.5]))
        np.testing.assert_array_equal(ctavg, np.zeros_like(ctavg))

    def test_bold_monitor_matches_python_monitor_sample_path(self):
        network, subnet = _make_network(2)
        initial_state = _make_initial_state(subnet)
        nstep = 80
        bold_period = 4.0

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            backend = CppHybridBackend(build_root=build_root)

            raw_compiled = backend.compile(
                network,
                monitors=[Raw()],
                user_source_hint="test_bold_monitor_reference_raw",
            )
            ((_, raw_data),) = raw_compiled.run(
                initial_states=[initial_state.copy()],
                nstep=nstep,
                chunk_size=1,
            )

            bold = Bold(period=bold_period)
            bold_compiled = backend.compile(
                network,
                monitors=[bold],
                user_source_hint="test_bold_monitor_matches_python_monitor",
            )
            ((bold_times, bold_data),) = bold_compiled.run(
                initial_states=[initial_state.copy()],
                nstep=nstep,
                chunk_size=1,
            )

        ref_bold = Bold(period=bold_period)
        ref_bold.voi = slice(None)
        ref_bold._config_dt(DT)
        ref_bold.compute_hrf()
        ref_bold._config_stock(
            len(subnet.model.variables_of_interest),
            subnet.nnodes,
            subnet.model.number_of_modes,
        )

        ref_times = []
        ref_data = []
        for step, state in enumerate(raw_data, start=1):
            maybe_bold = ref_bold.sample(step, state)
            if maybe_bold is not None:
                ref_times.append(maybe_bold[0])
                ref_data.append(maybe_bold[1])

        ref_times = np.asarray(ref_times, dtype=np.float64)
        ref_data = np.asarray(ref_data, dtype=np.float64)

        self.assertEqual(bold_times.shape, (2,))
        self.assertEqual(bold_data.shape, (2, 2, subnet.nnodes, 1))
        np.testing.assert_allclose(bold_times, ref_times)
        np.testing.assert_allclose(bold_data, ref_data, rtol=1e-12, atol=1e-12)


    def test_unsupported_monitor_raises_clear_error(self):
        network, _subnet = _make_network(2)

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            backend = CppHybridBackend(build_root=build_root)
            with self.assertRaisesRegex(
                NotImplementedError,
                "Monitor 'SubSample' is not yet supported by the C\\+\\+ backend",
            ):
                backend.compile(
                    network,
                    monitors=[SubSample(period=DT)],
                    user_source_hint="test_unsupported_monitor_raises_clear_error",
                )

    def test_intra_projection_matches_numba_reference(self):
        """Intra-projection (r→r self-coupling, zero delay) matches NbHybridBackend."""
        nstep = 20
        # NbHybridBackend requires chunk_size <= min_horizon (1 for zero-delay projections)
        chunk_size = 1
        n_nodes = 4

        # Build an intra-projection: r state var → Coupling_Term_r (slot 0)
        w = sp.eye(n_nodes, format="csr") * 0.05
        l = sp.csr_matrix((n_nodes, n_nodes))  # zero delays
        proj = IntraProjection(
            source_cvar=np.array([0]),   # r
            target_cvar=np.array([0]),   # Coupling_Term_r slot
            weights=w,
            lengths=l,
            cv=7.0,
            dt=DT,
            scale=0.1,
        )
        network, subnet = _make_network(n_nodes, projections=[proj])
        initial_state = _make_initial_state(subnet)

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

        self.assertEqual(nb_data.shape, native_data.shape)
        # Numba uses float32; allow for float32↔float64 rounding
        np.testing.assert_allclose(native_data, nb_data, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(native_times, nb_times, rtol=1e-6, atol=1e-6)

    def test_multi_subnet_compiled_run_matches_independent_single_subnets(self):
        """A compiled two-subnet NetworkSet runs and returns one result per subnet."""
        nstep = 6
        chunk_size = 2

        subnet_a = _make_subnet("a", 1)
        subnet_a.node_indices = np.array([0])
        subnet_b = _make_subnet("b", 3)
        subnet_b.node_indices = np.array([1, 2, 3])
        multi_network = NetworkSet(subnets=[subnet_a, subnet_b], projections=[])
        multi_network.configure()

        initial_a = _make_initial_state(subnet_a)
        initial_b = _make_initial_state(subnet_b)
        initial_a[0, :, :] = 0.7
        initial_a[1, :, :] = -0.25
        initial_b[0, :, :] = np.array([[1.0], [1.1], [1.2]])
        initial_b[1, :, :] = np.array([[-0.45], [-0.35], [-0.30]])

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            backend = CppHybridBackend(build_root=build_root)
            multi_compiled = backend.compile(
                multi_network,
                monitors=[TemporalAverage(period=chunk_size * DT)],
                user_source_hint="test_multi_subnet_compiled_run",
            )
            multi_results = multi_compiled.run(
                initial_states=[initial_a.copy(), initial_b.copy()],
                nstep=nstep,
                chunk_size=chunk_size,
            )

            single_a, single_subnet_a = _make_network(1)
            single_b, single_subnet_b = _make_network(3)
            single_initial_a = initial_a.copy()
            single_initial_b = initial_b.copy()
            times_a, data_a = _run_native(
                single_a,
                single_initial_a,
                nstep,
                chunk_size,
                build_root,
            )
            times_b, data_b = _run_native(
                single_b,
                single_initial_b,
                nstep,
                chunk_size,
                build_root,
            )

        self.assertEqual(len(multi_results), 2)
        (multi_times_a, multi_data_a), (multi_times_b, multi_data_b) = multi_results
        self.assertEqual(multi_data_a.shape, (3, 2, single_subnet_a.nnodes, 1))
        self.assertEqual(multi_data_b.shape, (3, 2, single_subnet_b.nnodes, 1))
        np.testing.assert_allclose(multi_times_a, times_a, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(multi_times_b, times_b, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(multi_data_a, data_a, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(multi_data_b, data_b, rtol=1e-12, atol=1e-12)

    def test_inter_projection_delayed_coupling_is_nonzero_and_shaped(self):
        """Inter-projection delayed coupling reaches the target subnet."""
        nstep = 6
        delay_steps = 2
        cv = 10.0
        scale = 0.5
        weights = np.array([[0.20], [0.35]], dtype=np.float64)
        lengths = np.full_like(weights, delay_steps * cv * DT)

        source = _make_subnet("source", 1)
        source.node_indices = np.array([0])
        target = _make_subnet("target", 2)
        target.node_indices = np.array([1, 2])
        projection = InterProjection(
            source=source,
            target=target,
            source_cvar=np.array([0]),  # source r state variable
            target_cvar=np.array([0]),  # target Coupling_Term_r slot
            weights=sp.csr_matrix(weights),
            lengths=sp.csr_matrix(lengths),
            cv=cv,
            dt=DT,
            scale=scale,
        )
        network = NetworkSet(subnets=[source, target], projections=[projection])
        network.configure()

        initial_source = _make_initial_state(source)
        initial_target = _make_initial_state(target)
        initial_source[0, :, :] = 0.8
        initial_source[1, :, :] = -0.2
        initial_target[0, :, :] = 1.1
        initial_target[1, :, :] = -0.3

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            backend = CppHybridBackend(build_root=build_root)
            raw_compiled = backend.compile(
                network,
                monitors=[Raw()],
                user_source_hint="test_inter_projection_delayed_raw",
            )
            raw_results = raw_compiled.run(
                initial_states=[initial_source.copy(), initial_target.copy()],
                nstep=nstep,
                chunk_size=1,
            )
            (raw_source_times, raw_source_data), (_raw_target_times, _raw_target_data) = (
                raw_results
            )

            aff_compiled = backend.compile(
                network,
                monitors=[AfferentCoupling()],
                user_source_hint="test_inter_projection_delayed_coupling",
            )
            aff_results = aff_compiled.run(
                initial_states=[initial_source.copy(), initial_target.copy()],
                nstep=nstep,
                chunk_size=1,
            )
            (source_times, source_coupling), (target_times, target_coupling) = (
                aff_results
            )

        self.assertEqual(projection.idelays.tolist(), [delay_steps, delay_steps])
        self.assertEqual(source_coupling.shape, (nstep, 2, 1, 1))
        self.assertEqual(target_coupling.shape, (nstep, 2, 2, 1))
        np.testing.assert_allclose(source_times, DT * np.arange(1, nstep + 1))
        np.testing.assert_array_equal(source_coupling, np.zeros_like(source_coupling))
        self.assertGreater(float(np.max(np.abs(target_coupling[:, 0, :, 0]))), 0.0)

        source_r_after_steps = raw_source_data[:, 0, 0, 0]
        delayed_source_r = np.concatenate(
            (
                np.full(delay_steps + 1, initial_source[0, 0, 0]),
                source_r_after_steps[: nstep - delay_steps - 1],
            )
        )
        expected = (
            scale
            * delayed_source_r[:, np.newaxis]
            * weights[:, 0][np.newaxis, :]
        )
        np.testing.assert_allclose(target_times, raw_source_times, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            target_coupling[:, 0, :, 0],
            expected,
            rtol=1e-7,
            atol=1e-9,
        )
        np.testing.assert_array_equal(
            target_coupling[:, 1, :, 0],
            np.zeros_like(target_coupling[:, 1, :, 0]),
        )

    def test_zero_weight_projection_matches_no_projection(self):
        """A projection with zero weights must produce identical output to no projection."""
        nstep = 10
        chunk_size = 2
        n_nodes = 3

        network_no_proj, subnet = _make_network(n_nodes)
        initial_state = _make_initial_state(subnet)

        w_zero = sp.csr_matrix((n_nodes, n_nodes))  # all zeros
        l_zero = sp.csr_matrix((n_nodes, n_nodes))
        proj = IntraProjection(
            source_cvar=np.array([0]),
            target_cvar=np.array([0]),
            weights=w_zero,
            lengths=l_zero,
            cv=7.0,
            dt=DT,
            scale=1.0,
        )
        network_with_proj, _ = _make_network(n_nodes, projections=[proj])

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            times_no, data_no = _run_native(
                network_no_proj, initial_state, nstep, chunk_size, build_root
            )
            times_with, data_with = _run_native(
                network_with_proj, initial_state, nstep, chunk_size, build_root
            )

        np.testing.assert_array_equal(times_no, times_with)
        np.testing.assert_array_equal(data_no, data_with)

    def test_stimulus_output_differs_from_no_stimulus_baseline(self):
        """Non-zero stimulus must change the output on targeted nodes."""
        n_nodes = 4
        nstep = 50
        chunk_size = 5
        simulation_length = nstep * DT

        network_no_stim, subnet = _make_network(n_nodes)
        initial_state = _make_initial_state(subnet)

        # Build a stimulus network sharing the same initial state.
        subnet_stim = _make_subnet("sn", n_nodes)
        subnet_stim.node_indices = np.arange(n_nodes)
        network_stim = NetworkSet(subnets=[subnet_stim], projections=[])

        temporal = equations.PulseTrain()
        temporal.parameters["onset"] = 0.0
        temporal.parameters["T"] = simulation_length * 2  # one long pulse covering full run
        temporal.parameters["tau"] = simulation_length
        temporal.parameters["amp"] = 0.5

        weights = np.zeros(n_nodes, dtype=np.float64)
        weights[0] = 1.0  # node 0 receives full stimulus; node 1+ receive none

        conn = Connectivity(
            centres=np.ones((n_nodes, 3)),
            weights=np.ones((n_nodes, n_nodes), dtype=np.float64),
            tract_lengths=np.zeros((n_nodes, n_nodes), dtype=np.float64),
            region_labels=np.array([f"r{i}" for i in range(n_nodes)]),
            speed=np.array([1.0]),
        )
        conn.configure()

        subnet_stim.add_stimulus(
            stimulus=patterns.StimuliRegion(
                temporal=temporal, connectivity=conn, weight=weights
            ),
            stimulus_cvar=np.r_[0],
            projection_scale=1.0,
        )
        network_stim.configure()
        for stim in subnet_stim.stimuli:
            stim.configure(simulation_length)

        with tempfile.TemporaryDirectory(prefix="tvb-cpp-backend-build-") as build_root:
            times_no, data_no = _run_native(
                network_no_stim, initial_state.copy(), nstep, chunk_size, build_root
            )
            times_stim, data_stim = _run_native(
                network_stim, initial_state.copy(), nstep, chunk_size, build_root
            )

        self.assertEqual(data_no.shape, data_stim.shape)
        np.testing.assert_array_equal(times_no, times_stim)
        # Node 0 received stimulus — its trajectory must diverge from the baseline.
        diff_node0 = float(np.max(np.abs(data_stim[:, :, 0, :] - data_no[:, :, 0, :])))
        self.assertGreater(diff_node0, 1e-6, "stimulus had no effect on node 0")
