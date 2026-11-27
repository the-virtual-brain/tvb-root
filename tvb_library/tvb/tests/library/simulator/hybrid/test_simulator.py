# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
Tests for the hybrid :class:`~tvb.simulator.hybrid.Simulator` class.

Covers full end-to-end simulation runs, including:

- :class:`~tvb.simulator.monitors.TemporalAverage` monitors operating over the
  whole network;
- per-subnetwork monitors via :class:`~tvb.simulator.hybrid.Recorder`;
- mixed-model networks with :class:`~tvb.simulator.models.JansenRit` and
  :class:`~tvb.simulator.models.ReducedSetFitzHughNagumo` subnetworks;
- the canonical usage example reproduced from the module docstring of
  :mod:`tvb.simulator.hybrid`.

Output shape convention: ``(time_steps, n_vois, n_nodes, n_modes)``.
"""

import sys
import warnings
from unittest import mock

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from tvb.simulator.models import (
    DecoBalancedExcInh,
    JansenRit,
    Linear,
    ReducedSetFitzHughNagumo,
)
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.monitors import (
    AfferentCouplingTemporalAverage,
    Bold,
    GlobalAverage,
    Raw,
    SpatialAverage,
    TemporalAverage,
)
from tvb.simulator.hybrid import (
    Recorder,
    Simulator,
    Subnetwork,
    InterProjection,
    NetworkSet,
)
from .test_base import BaseHybridTest


def _has_cpp_hybrid():
    try:
        from tvb.simulator.backend.cpp_hybrid import CppHybridBackend  # noqa: F401
        return True
    except ImportError:
        return False


def _bounded_setup_ics(cortex, thalamus, seed=42):
    """Finite small initial conditions so FHN/JR trajectories stay non-NaN."""
    rng = np.random.RandomState(seed)
    return [
        rng.uniform(-0.1, 0.1, (cortex.model.nvar, cortex.nnodes, cortex.model.number_of_modes)),
        rng.uniform(-0.1, 0.1, (thalamus.model.nvar, thalamus.nnodes, thalamus.model.number_of_modes)),
    ]


class TestSimulator(BaseHybridTest):
    """End-to-end tests for :class:`~tvb.simulator.hybrid.Simulator`.

    Exercises the full simulation pipeline: network configuration, time
    stepping, monitor output collection, and output shape validation.
    """

    def test_sim(self):
        """Full simulation run with a global TemporalAverage monitor.

        Runs a 2-subnetwork (cortex + thalamus) simulation for 10 ms.
        Asserts 10 time points are returned and the output array has shape
        ``(10, n_vois=4, n_nodes=76, modes=1)``.
        """
        conn, ix, cortex, thalamus, a, nets = self.setup(nvois=2)
        tavg = TemporalAverage(period=1.0)
        sim = Simulator(
            nets=nets,
            simulation_length=10.0,
            monitors=[tavg],
        )
        sim.configure()
        ((t, y),) = sim.run()
        self.assert_equal(10, len(t))
        self.assert_equal((10, 4, 76, 1), y.shape)

    def test_sim_jrmon(self):
        """Simulation with a per-subnetwork Recorder monitor.

        When the monitor is attached directly to a subnetwork (not passed to
        :class:`~tvb.simulator.hybrid.Simulator`), ``sim.run()`` returns an
        empty list and results must be fetched from ``cortex.monitors[0]``.
        Output shape should be ``(10, n_vois, n_cortex_nodes, modes=1)``.
        """
        jrmon = TemporalAverage(period=1.0)
        conn, ix, cortex, thalamus, a, nets = self.setup(jrmon=jrmon)
        sim = Simulator(nets=nets, simulation_length=10.0)
        sim.configure()
        xs = sim.run()
        self.assert_equal(0, len(xs))
        rec: Recorder = cortex.monitors[0]
        nn = cortex.nnodes
        self.assert_equal((10, 4, nn, 1), rec.shape)
        t, y = rec.to_arrays()
        self.assert_equal((10,), t.shape)
        self.assert_equal((10, 4, nn, 1), y.shape)

    def test_module_example(self):
        """Reproduce the canonical usage example from the hybrid simulator documentation.

        Builds a two-subnetwork coupled system (JansenRit + ReducedSetFitzHughNagumo),
        connects them via an :class:`~tvb.simulator.hybrid.InterProjection`, and runs
        for 100 ms.  Confirms 100 time steps are returned and the simulation
        completes without error.
        """
        # Create subnetworks with different models
        # Specify the same number of variables of interest for both models
        jrkwargs = {
            "variables_of_interest": JansenRit.variables_of_interest.default[:2]
        }
        fhnkwargs = {
            "variables_of_interest": ReducedSetFitzHughNagumo.variables_of_interest.default[
                :2
            ]
        }

        cortex = Subnetwork(
            name="cortex",
            model=JansenRit(**jrkwargs),
            scheme=HeunDeterministic(dt=0.1),
            nnodes=76,
        ).configure()  # Configure the model

        thalamus = Subnetwork(
            name="thalamus",
            model=ReducedSetFitzHughNagumo(**fhnkwargs),
            scheme=HeunDeterministic(dt=0.1),
            nnodes=76,
        ).configure()  # Configure the model

        # Prepare projection parameters
        weights_data = np.random.randn(76, 76)
        weights_matrix = csr_matrix(weights_data)

        # Create lengths matrix with the same sparsity pattern as weights
        lengths_data = np.abs(np.random.randn(76, 76))  # Ensure positive lengths
        lengths_data[weights_data == 0] = 0  # Match sparsity
        lengths_matrix = csr_matrix(lengths_data)

        projection_dt = 0.1  # Match scheme dt
        projection_cv = 1.0

        # Define projections between subnetworks
        nets = NetworkSet(
            subnets=[cortex, thalamus],
            projections=[
                InterProjection(
                    source=cortex,
                    target=thalamus,
                    source_cvar=np.r_[0],
                    target_cvar=np.r_[1],
                    weights=weights_matrix,
                    lengths=lengths_matrix,
                    cv=projection_cv,
                    dt=projection_dt,
                )
            ],
        )

        # Simulate the coupled system
        tavg = TemporalAverage(period=1.0)  # Add a monitor
        sim = Simulator(
            nets=nets,
            simulation_length=100,
            monitors=[tavg],  # Include the monitor
        )
        sim.configure()
        ((t, y),) = sim.run()  # Unpack the first (and only) monitor result

        # Verify the simulation ran successfully
        self.assert_equal(100, len(t))
        # The output shape is (time_steps, variables_of_interest, total_nodes, modes)
        # Total nodes = cortex nodes + thalamus nodes = 76 + 76 = 152
        self.assert_equal((100, 4, 152, 1), y.shape)

    def test_backend_invalid_raises(self):
        """Simulator rejects unknown backend names with ValueError."""
        _, _, _, _, _, nets = self.setup(nvois=2)
        with pytest.raises(ValueError, match="backend must be one of"):
            Simulator(nets=nets, simulation_length=1.0, backend="invalid")

    def test_cpp_backend_missing_extension_raises_informative_import_error(self):
        """Simulator(backend='cpp') gives an informative ImportError if missing."""
        _, _, _, _, _, nets = self.setup(nvois=2)
        with mock.patch.dict(
            sys.modules, {"tvb.simulator.backend.cpp_hybrid": None}
        ):
            with pytest.raises(ImportError, match="_cpp_hybrid"):
                Simulator(nets=nets, simulation_length=1.0, backend="cpp")

    @pytest.mark.parametrize("backend", ["numba", "cpp"])
    def test_heterogeneous_vois_compiled_matches_python(self, backend):
        """Heterogeneous VOIs (JansenRit 4 + FHN 2) reassemble block-diagonally."""
        if backend == "cpp" and not _has_cpp_hybrid():
            pytest.skip("_cpp_hybrid extension is not available")

        _, _, cortex_py, thalamus_py, _, nets_py = self.setup()
        ics = _bounded_setup_ics(cortex_py, thalamus_py, seed=42)
        sim_py = Simulator(
            nets=nets_py,
            simulation_length=10.0,
            monitors=[TemporalAverage(period=1.0)],
            backend="python",
        )
        sim_py.configure()
        ((t_py, y_py),) = sim_py.run(initial_conditions=[x.copy() for x in ics])

        _, _, _, _, _, nets_be = self.setup()
        sim_be = Simulator(
            nets=nets_be,
            simulation_length=10.0,
            monitors=[TemporalAverage(period=1.0)],
            backend=backend,
        )
        sim_be.configure()
        ((t_be, y_be),) = sim_be.run(initial_conditions=[x.copy() for x in ics])

        self.assert_equal((10, 6, 76, 1), y_py.shape)
        self.assert_equal((10, 6, 76, 1), y_be.shape)
        assert np.all(np.isfinite(y_py))
        assert np.all(np.isfinite(y_be))
        np.testing.assert_array_equal(t_be, t_py)
        np.testing.assert_allclose(y_be, y_py, rtol=1e-3, atol=1e-4)

    def test_merged_mode_with_three_mode_fhn_first_subnet(self):
        """Merged mode with 3-mode FHN as subnet 0 emits 1 mode on numba and cpp."""

        def _build_merged_nets():
            fhn = Subnetwork(
                name="fhn",
                model=ReducedSetFitzHughNagumo(),
                scheme=HeunDeterministic(dt=0.1),
                nnodes=3,
                node_indices=np.array([0, 2, 4]),
            ).configure()
            jr = Subnetwork(
                name="jr",
                model=JansenRit(
                    variables_of_interest=JansenRit.variables_of_interest.default[:2]
                ),
                scheme=HeunDeterministic(dt=0.1),
                nnodes=2,
                node_indices=np.array([1, 3]),
            ).configure()
            proj = InterProjection(
                source=fhn,
                target=jr,
                source_cvar=np.r_[0],
                target_cvar=np.r_[1],
                weights=csr_matrix(np.array([[0.1, 0.0, 0.2], [0.0, 0.15, 0.0]])),
                lengths=csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 1.5, 0.0]])),
                cv=3.0,
                dt=0.1,
                scale=1e-3,
            )
            return NetworkSet(subnets=[fhn, jr], projections=[proj])

        ic_fhn = np.array(
            [
                [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35], [0.2, 0.3, 0.4]],
                [[-0.1, 0.0, 0.1], [-0.05, 0.05, 0.15], [0.0, 0.1, 0.2]],
                [[0.05, 0.1, 0.15], [0.1, 0.15, 0.2], [0.15, 0.2, 0.25]],
                [[0.0, 0.02, 0.04], [0.01, 0.03, 0.05], [0.02, 0.04, 0.06]],
            ],
            dtype=np.float64,
        )
        ic_jr = np.zeros((6, 2, 1), dtype=np.float64)
        ic_jr[1, :, 0] = [0.05, 0.08]

        # Pure-Python reference via NetworkSet.step + observe(flat=False)
        nets_ref = _build_merged_nets()
        nets_ref.configure()
        x = nets_ref.zero_states(initial_states=[ic_fhn.copy(), ic_jr.copy()])
        nets_ref.init_projection_buffers(x)
        mon = TemporalAverage(period=1.0)
        mon._config_dt(0.1)
        mon.voi = slice(None)
        mon._config_stock(2, 5, 1)
        ref_t, ref_y = [], []
        for step in range(1, 51):
            x = nets_ref.step(step, x)
            obs = nets_ref.observe(x, flat=False)
            merged_ox = np.zeros((2, 5, 1), dtype=np.float64)
            merged_ox[:, nets_ref.subnets[0].node_indices, :] = obs[0]
            merged_ox[:, nets_ref.subnets[1].node_indices, :] = obs[1]
            rec = mon.record(step, merged_ox)
            if rec is not None:
                ref_t.append(rec[0])
                ref_y.append(rec[1])
        t_ref = np.array(ref_t)
        y_ref = np.array(ref_y)

        backends = ["numba"]
        if _has_cpp_hybrid():
            backends.append("cpp")
        for backend in backends:
            sim_be = Simulator(
                nets=_build_merged_nets(),
                simulation_length=5.0,
                monitors=[TemporalAverage(period=1.0)],
                backend=backend,
            )
            sim_be.configure()
            ((t_be, y_be),) = sim_be.run(
                initial_conditions=[ic_fhn.copy(), ic_jr.copy()]
            )
            self.assert_equal((5, 2, 5, 1), y_be.shape)
            np.testing.assert_allclose(t_be, t_ref, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(y_be, y_ref, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("backend", ["numba", "cpp"])
    def test_sim_jrmon_compiled_backends(self, backend):
        """Per-subnetwork Recorder works on numba and cpp backends and matches python."""
        if backend == "cpp" and not _has_cpp_hybrid():
            pytest.skip("_cpp_hybrid extension is not available")

        _, _, cortex_py, thalamus_py, _, nets_py = self.setup(
            jrmon=TemporalAverage(period=1.0)
        )
        ics = _bounded_setup_ics(cortex_py, thalamus_py, seed=42)
        sim_py = Simulator(nets=nets_py, simulation_length=10.0, backend="python")
        sim_py.configure()
        xs_py = sim_py.run(initial_conditions=[x.copy() for x in ics])
        self.assert_equal(0, len(xs_py))
        t_py, y_py = cortex_py.monitors[0].to_arrays()

        _, _, cortex_be, _, _, nets_be = self.setup(jrmon=TemporalAverage(period=1.0))
        sim_be = Simulator(nets=nets_be, simulation_length=10.0, backend=backend)
        sim_be.configure()
        xs_be = sim_be.run(initial_conditions=[x.copy() for x in ics])
        self.assert_equal(0, len(xs_be))

        rec: Recorder = cortex_be.monitors[0]
        nn = cortex_be.nnodes
        self.assert_equal((10, 4, nn, 1), rec.shape)
        t_be, y_be = rec.to_arrays()
        self.assert_equal((10,), t_be.shape)
        self.assert_equal((10, 4, nn, 1), y_be.shape)
        assert np.all(np.isfinite(y_py))
        assert np.all(np.isfinite(y_be))
        np.testing.assert_allclose(t_be, t_py, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(y_be, y_py, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("backend", ["numba", "cpp"])
    def test_recorder_afferent_coupling_and_global_monitor(self, backend):
        """AfferentCouplingTemporalAverage Recorder works alongside global monitors."""
        if backend == "cpp" and not _has_cpp_hybrid():
            pytest.skip("_cpp_hybrid extension is not available")

        def _build():
            _, _, cortex, thalamus, _, nets = self.setup()  # heterogeneous 4 + 2 VOIs
            cortex.add_monitor(AfferentCouplingTemporalAverage(period=1.0))
            return cortex, thalamus, nets

        cortex_py, thalamus_py, nets_py = _build()
        ics = _bounded_setup_ics(cortex_py, thalamus_py, seed=7)
        sim_py = Simulator(
            nets=nets_py,
            simulation_length=5.0,
            monitors=[TemporalAverage(period=1.0)],
            backend="python",
        )
        sim_py.configure()
        ((tg_py, yg_py),) = sim_py.run(initial_conditions=[x.copy() for x in ics])
        tc_py, yc_py = cortex_py.monitors[0].to_arrays()

        cortex_be, _, nets_be = _build()
        sim_be = Simulator(
            nets=nets_be,
            simulation_length=5.0,
            monitors=[TemporalAverage(period=1.0)],
            backend=backend,
        )
        sim_be.configure()
        ((tg_be, yg_be),) = sim_be.run(initial_conditions=[x.copy() for x in ics])
        tc_be, yc_be = cortex_be.monitors[0].to_arrays()

        assert np.all(np.isfinite(yg_py))
        assert np.all(np.isfinite(yg_be))
        assert np.all(np.isfinite(yc_py))
        assert np.all(np.isfinite(yc_be))
        np.testing.assert_array_equal(tg_be, tg_py)
        np.testing.assert_allclose(yg_be, yg_py, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(tc_be, tc_py, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(yc_be, yc_py, rtol=1e-3, atol=1e-4)

    def test_multirate_dt_validation_and_run(self):
        """Multirate dt validates and runs on numba/cpp while python rejects it."""

        def _build_multirate(dt_fast=0.01, dt_slow=0.02):
            fast = Subnetwork(
                name="fast",
                model=MontbrioPazoRoxin(),
                scheme=HeunDeterministic(dt=dt_fast),
                nnodes=2,
            ).configure()
            slow = Subnetwork(
                name="slow",
                model=MontbrioPazoRoxin(),
                scheme=HeunDeterministic(dt=dt_slow),
                nnodes=3,
            ).configure()
            proj = InterProjection(
                source=fast,
                target=slow,
                source_cvar=np.r_[0],
                target_cvar=np.r_[0],
                weights=csr_matrix(np.ones((3, 2)) * 0.1),
                lengths=csr_matrix(np.zeros((3, 2))),
                cv=1.0,
                dt=dt_fast,
                scale=1.0,
            )
            return NetworkSet(subnets=[fast, slow], projections=[proj])

        with pytest.raises(ValueError, match="dt mismatch"):
            Simulator(
                nets=_build_multirate(0.01, 0.02),
                simulation_length=0.2,
                monitors=[Raw()],
                backend="python",
            )

        with pytest.raises(ValueError, match="integer multiples"):
            Simulator(
                nets=_build_multirate(0.01, 0.025),
                simulation_length=0.2,
                monitors=[Raw()],
                backend="numba",
            )

        sim_nb = Simulator(
            nets=_build_multirate(0.01, 0.02),
            simulation_length=0.2,
            monitors=[Raw()],
            backend="numba",
        )
        assert sim_nb._dt0 == pytest.approx(0.01)
        sim_nb.configure()
        ((t_nb, y_nb),) = sim_nb.run(random_state=11)
        self.assert_equal(20, len(t_nb))
        self.assert_equal((20, 2, 5, 1), y_nb.shape)

        if _has_cpp_hybrid():
            sim_cpp = Simulator(
                nets=_build_multirate(0.01, 0.02),
                simulation_length=0.2,
                monitors=[Raw()],
                backend="cpp",
            )
            assert sim_cpp._dt0 == pytest.approx(0.01)
            sim_cpp.configure()
            ((t_cpp, y_cpp),) = sim_cpp.run(random_state=11)
            np.testing.assert_array_equal(t_cpp, t_nb)
            np.testing.assert_allclose(y_cpp, y_nb, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("backend", ["python", "numba", "cpp"])
    def test_sim_all_backends(self, backend):
        """Merged-mode 2-VOI setup runs on python, numba, and cpp with parity."""
        if backend == "cpp" and not _has_cpp_hybrid():
            pytest.skip("_cpp_hybrid extension is not available")

        def _build_merged():
            _, ix, cortex, thalamus, AtlasPart, nets = self.setup(nvois=2)
            cortex.node_indices = np.where(ix == AtlasPart.CORTEX)[0]
            thalamus.node_indices = np.where(ix == AtlasPart.THALAMUS)[0]
            return cortex, thalamus, nets

        cortex_py, thalamus_py, nets_py = _build_merged()
        ics = _bounded_setup_ics(cortex_py, thalamus_py, seed=17)
        sim_py = Simulator(
            nets=nets_py,
            simulation_length=10.0,
            monitors=[TemporalAverage(period=1.0)],
            backend="python",
        )
        sim_py.configure()
        ((t_py, y_py),) = sim_py.run(initial_conditions=[x.copy() for x in ics])
        self.assert_equal(10, len(t_py))
        self.assert_equal((10, 2, 76, 1), y_py.shape)
        assert np.all(np.isfinite(y_py))

        if backend == "python":
            return

        _, _, nets_be = _build_merged()
        sim_be = Simulator(
            nets=nets_be,
            simulation_length=10.0,
            monitors=[TemporalAverage(period=1.0)],
            backend=backend,
        )
        sim_be.configure()
        ((t_be, y_be),) = sim_be.run(initial_conditions=[x.copy() for x in ics])
        self.assert_equal((10, 2, 76, 1), y_be.shape)
        assert np.all(np.isfinite(y_be))
        np.testing.assert_array_equal(t_be, t_py)
        np.testing.assert_allclose(y_be, y_py, rtol=1e-3, atol=1e-4)

    def test_sim_unmerged_nvois2_compiled_parity(self):
        """Unmerged 2-VOI setup matches between numba and cpp backends."""
        if not _has_cpp_hybrid():
            pytest.skip("_cpp_hybrid extension is not available")

        _, _, cortex_nb, thalamus_nb, _, nets_nb = self.setup(nvois=2)
        ics = _bounded_setup_ics(cortex_nb, thalamus_nb, seed=17)
        sim_nb = Simulator(
            nets=nets_nb,
            simulation_length=10.0,
            monitors=[TemporalAverage(period=1.0)],
            backend="numba",
        )
        sim_nb.configure()
        ((t_nb, y_nb),) = sim_nb.run(initial_conditions=[x.copy() for x in ics])

        _, _, _, _, _, nets_cpp = self.setup(nvois=2)
        sim_cpp = Simulator(
            nets=nets_cpp,
            simulation_length=10.0,
            monitors=[TemporalAverage(period=1.0)],
            backend="cpp",
        )
        sim_cpp.configure()
        ((t_cpp, y_cpp),) = sim_cpp.run(initial_conditions=[x.copy() for x in ics])

        self.assert_equal((10, 2, 76, 1), y_nb.shape)
        self.assert_equal((10, 2, 76, 1), y_cpp.shape)
        assert np.all(np.isfinite(y_nb))
        assert np.all(np.isfinite(y_cpp))
        np.testing.assert_array_equal(t_cpp, t_nb)
        np.testing.assert_allclose(y_cpp, y_nb, rtol=1e-3, atol=1e-4)

    def test_linear_model_no_syntax_warning_and_cpp_simulator(self):
        """Linear module compiles without SyntaxWarning and runs across backends."""
        from pathlib import Path
        import tvb.simulator.models.linear as linear_mod

        source = Path(linear_mod.__file__).read_text(encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntaxWarning)
            compile(source, linear_mod.__file__, "exec")

        def _build_linear():
            a = Subnetwork(
                name="a",
                model=Linear(gamma=np.array([-2.0])),
                scheme=HeunDeterministic(dt=0.1),
                nnodes=3,
                node_indices=np.array([0, 1, 2]),
            ).configure()
            b = Subnetwork(
                name="b",
                model=Linear(gamma=np.array([-3.0])),
                scheme=HeunDeterministic(dt=0.1),
                nnodes=2,
                node_indices=np.array([3, 4]),
            ).configure()
            proj = InterProjection(
                source=a,
                target=b,
                source_cvar=np.r_[0],
                target_cvar=np.r_[0],
                weights=csr_matrix(np.array([[0.2, 0.0, 0.1], [0.0, 0.3, 0.0]])),
                lengths=csr_matrix(np.zeros((2, 3))),
                cv=1.0,
                dt=0.1,
                scale=1.0,
            )
            return NetworkSet(subnets=[a, b], projections=[proj])

        ics = [
            np.array([[[0.5], [-0.2], [0.3]]], dtype=np.float64),
            np.array([[[0.1], [-0.4]]], dtype=np.float64),
        ]
        results = {}
        backends = ["python", "numba"]
        if _has_cpp_hybrid():
            backends.append("cpp")
        for backend in backends:
            sim = Simulator(
                nets=_build_linear(),
                simulation_length=2.0,
                monitors=[TemporalAverage(period=0.5)],
                backend=backend,
            )
            sim.configure()
            ((t, y),) = sim.run(initial_conditions=[x.copy() for x in ics])
            assert np.all(np.isfinite(y))
            results[backend] = (t, y)

        t_py, y_py = results["python"]
        for backend in backends[1:]:
            t_be, y_be = results[backend]
            np.testing.assert_array_equal(t_be, t_py)
            np.testing.assert_allclose(y_be, y_py, rtol=1e-4, atol=1e-5)

    def test_cpp_single_pass_with_global_and_subnet_recorders(self, monkeypatch):
        """Simulator(backend='cpp') collects global + Recorder monitors in one call."""
        if not _has_cpp_hybrid():
            pytest.skip("_cpp_hybrid extension is not available")

        from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend

        def _build():
            _, _, cortex, thalamus, _, nets = self.setup()  # heterogeneous 4 + 2 VOIs
            cortex.add_monitor(AfferentCouplingTemporalAverage(period=1.0))
            cortex.add_monitor(TemporalAverage(period=1.0))
            return cortex, thalamus, nets

        cortex_py, thalamus_py, nets_py = _build()
        ics = _bounded_setup_ics(cortex_py, thalamus_py, seed=13)
        sim_py = Simulator(
            nets=nets_py,
            simulation_length=5.0,
            monitors=[TemporalAverage(period=1.0)],
            backend="python",
        )
        sim_py.configure()
        ((tg_py, yg_py),) = sim_py.run(initial_conditions=[x.copy() for x in ics])
        tc_py, yc_py = cortex_py.monitors[0].to_arrays()
        tt_py, yt_py = cortex_py.monitors[1].to_arrays()

        call_count = 0
        orig_run_network = CppHybridBackend.run_network

        def _spy_run_network(be_self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return orig_run_network(be_self, *args, **kwargs)

        monkeypatch.setattr(CppHybridBackend, "run_network", _spy_run_network)

        cortex_cpp, thalamus_cpp, nets_cpp = _build()
        sim_cpp = Simulator(
            nets=nets_cpp,
            simulation_length=5.0,
            monitors=[TemporalAverage(period=1.0)],
            backend="cpp",
        )
        sim_cpp.configure()
        ((tg_cpp, yg_cpp),) = sim_cpp.run(initial_conditions=[x.copy() for x in ics])
        tc_cpp, yc_cpp = cortex_cpp.monitors[0].to_arrays()
        tt_cpp, yt_cpp = cortex_cpp.monitors[1].to_arrays()

        assert call_count == 1, f"expected 1 run_network call, got {call_count}"
        np.testing.assert_array_equal(tg_cpp, tg_py)
        np.testing.assert_allclose(yg_cpp, yg_py, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(tc_cpp, tc_py, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(yc_cpp, yc_py, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(tt_cpp, tt_py, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(yt_cpp, yt_py, rtol=1e-3, atol=1e-4)

    def test_cpp_single_pass_multirate_subnet_recorders(self, monkeypatch):
        """Multi-dt Simulator(backend='cpp') with global + Recorder monitors runs once."""
        if not _has_cpp_hybrid():
            pytest.skip("_cpp_hybrid extension is not available")

        from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend

        def _build():
            fast = Subnetwork(
                name="fast",
                model=MontbrioPazoRoxin(),
                scheme=HeunDeterministic(dt=0.05),
                nnodes=2,
            ).configure()
            slow = Subnetwork(
                name="slow",
                model=MontbrioPazoRoxin(),
                scheme=HeunDeterministic(dt=0.1),
                nnodes=3,
            ).configure()
            proj = InterProjection(
                source=fast,
                target=slow,
                source_cvar=np.r_[0],
                target_cvar=np.r_[0],
                weights=csr_matrix(np.full((3, 2), 0.1)),
                lengths=csr_matrix(np.zeros((3, 2))),
                cv=1.0,
                dt=0.05,
                scale=1.0,
            )
            fast.add_monitor(TemporalAverage(period=0.5))
            slow.add_monitor(TemporalAverage(period=0.5))
            slow.add_monitor(AfferentCouplingTemporalAverage(period=0.5))
            nets = NetworkSet(subnets=[fast, slow], projections=[proj])
            return fast, slow, nets

        ics = [
            np.array([[[0.2], [0.4]], [[-1.0], [-0.5]]], dtype=np.float64),
            np.array([[[0.1], [0.3], [0.5]], [[-0.8], [-0.4], [-0.2]]], dtype=np.float64),
        ]

        fast_nb, slow_nb, nets_nb = _build()
        sim_nb = Simulator(
            nets=nets_nb,
            simulation_length=2.0,
            monitors=[TemporalAverage(period=0.5)],
            backend="numba",
        )
        sim_nb.configure()
        ((tg_nb, yg_nb),) = sim_nb.run(initial_conditions=[x.copy() for x in ics])
        nb_recs = [
            fast_nb.monitors[0].to_arrays(),
            slow_nb.monitors[0].to_arrays(),
            slow_nb.monitors[1].to_arrays(),
        ]

        call_count = 0
        orig_run_network = CppHybridBackend.run_network

        def _spy_run_network(be_self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return orig_run_network(be_self, *args, **kwargs)

        monkeypatch.setattr(CppHybridBackend, "run_network", _spy_run_network)

        fast_cpp, slow_cpp, nets_cpp = _build()
        sim_cpp = Simulator(
            nets=nets_cpp,
            simulation_length=2.0,
            monitors=[TemporalAverage(period=0.5)],
            backend="cpp",
        )
        sim_cpp.configure()
        ((tg_cpp, yg_cpp),) = sim_cpp.run(initial_conditions=[x.copy() for x in ics])
        cpp_recs = [
            fast_cpp.monitors[0].to_arrays(),
            slow_cpp.monitors[0].to_arrays(),
            slow_cpp.monitors[1].to_arrays(),
        ]

        assert call_count == 1, f"expected 1 run_network call, got {call_count}"
        np.testing.assert_allclose(tg_cpp, tg_nb, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(yg_cpp, yg_nb, rtol=1e-3, atol=1e-4)
        for (t_cpp, d_cpp), (t_nb, d_nb) in zip(cpp_recs, nb_recs):
            np.testing.assert_allclose(t_cpp, t_nb, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(d_cpp, d_nb, rtol=1e-3, atol=1e-4)

    def test_cpp_simulator_monitors_and_deco_balanced_parity(self):
        """DecoBalancedExcInh monitor value parity, with native midpoint timing."""
        if not _has_cpp_hybrid():
            pytest.skip("_cpp_hybrid extension is not available")

        def _build():
            a = Subnetwork(
                name="a",
                model=DecoBalancedExcInh(),
                scheme=HeunDeterministic(dt=0.1),
                nnodes=2,
                node_indices=np.array([0, 1]),
            ).configure()
            b = Subnetwork(
                name="b",
                model=DecoBalancedExcInh(),
                scheme=HeunDeterministic(dt=0.1),
                nnodes=2,
                node_indices=np.array([2, 3]),
            ).configure()
            proj = InterProjection(
                source=a,
                target=b,
                source_cvar=np.r_[0],
                target_cvar=np.r_[0],
                weights=csr_matrix(np.array([[0.2, 0.1], [0.1, 0.2]])),
                lengths=csr_matrix(np.zeros((2, 2))),
                cv=1.0,
                dt=0.1,
                scale=1.0,
            )
            return NetworkSet(subnets=[a, b], projections=[proj])

        ics = [
            np.array([[[0.15], [0.20]], [[0.05], [0.08]]], dtype=np.float64),
            np.array([[[0.12], [0.18]], [[0.04], [0.07]]], dtype=np.float64),
        ]

        # Compare the compiled temporal and BOLD streams to the Python route.
        # Global and spatial values below are compared between compiled routes.
        def _make_py_monitors():
            return [TemporalAverage(period=1.0), Bold(period=1.0)]

        py_sim = Simulator(
            nets=_build(),
            simulation_length=6.0,
            monitors=_make_py_monitors(),
            backend="python",
        )
        py_sim.configure()
        py_out = py_sim.run(initial_conditions=[x.copy() for x in ics])

        def _make_all_monitors():
            sa = SpatialAverage(period=1.0, spatial_mask=np.array([0, 0, 1, 1]))
            sa.spatial_mean = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
            return [
                TemporalAverage(period=1.0),
                GlobalAverage(period=1.0),
                sa,
                Bold(period=1.0),
            ]

        outputs = {}
        for backend in ("numba", "cpp"):
            sim = Simulator(
                nets=_build(),
                simulation_length=6.0,
                monitors=_make_all_monitors(),
                backend=backend,
            )
            sim.configure()
            outputs[backend] = sim.run(initial_conditions=[x.copy() for x in ics])

        # TemporalAverage (idx 0) and Bold (idx 3) match Python directly
        for idx, py_pair in zip((0, 3), py_out):
            t_py, y_py = py_pair
            for backend in ("numba", "cpp"):
                t_be, y_be = outputs[backend][idx]
                assert np.all(np.isfinite(y_be))
                np.testing.assert_allclose(t_be, t_py, rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(y_be, y_py, rtol=1e-3, atol=1e-4)

        nb_out = outputs["numba"]
        cpp_out = outputs["cpp"]
        assert len(cpp_out) == len(nb_out)
        for idx, ((t_cpp, y_cpp), (t_nb, y_nb)) in enumerate(zip(cpp_out, nb_out)):
            assert np.all(np.isfinite(y_cpp))
            # Global/Spatial use complete-window midpoints in C++, not the
            # historical Numba centre of sampled post-update ticks.
            reference = 0.5 + np.arange(6) if idx in (1, 2) else t_nb
            np.testing.assert_allclose(t_cpp, reference, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(y_cpp, y_nb, rtol=1e-3, atol=1e-4)
