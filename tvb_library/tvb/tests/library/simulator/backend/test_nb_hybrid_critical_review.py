# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
#  analysers necessary to run brain-simulations. You can use it stand alone or
#  in conjunction with TheVirtualBrain-Framework Package.
#
#  (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
#  This program is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software Foundation,
#  either version 3 of the License, or (at your option) any later version.
#
"""
Focused regression tests for critical fast-path / sweep bugs identified in
code review of the hybrid Numba backend.

These tests intentionally use tiny networks so they run quickly and isolate
specific parity issues between the pure-Python reference loop and the Numba
backend / parameter sweep.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.coupling import (
    Linear,
    SigmoidalJansenRit,
    HyperbolicTangent,
)
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.integrators import HeunDeterministic, EulerStochastic, HeunStochastic
from tvb.simulator.noise import Additive
from tvb.simulator.models.jansen_rit import JansenRit
from tvb.simulator.models.stefanescu_jirsa import ReducedSetFitzHughNagumo
from tvb.simulator.models.wong_wang import ReducedWongWang

DT = 0.1


def _sparse_weights(n: int, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.RandomState(seed)
    w = rng.uniform(0.0, 0.5, (n, n)).astype(np.float64)
    np.fill_diagonal(w, 0.0)
    return sp.csr_matrix(w)


def _run_python_loop(network_set: NetworkSet, nstep: int, x0_list: list) -> list:
    """Run pure-Python NetworkSet loop and return observed, mode-collapsed states."""
    x = network_set.States(*[arr.copy() for arr in x0_list])
    network_set.init_projection_buffers(x)
    outputs = [[] for _ in network_set.subnets]
    for step in range(1, nstep + 1):
        x = network_set.step(step, x)
        for i, xi in enumerate(x):
            obs = network_set.subnets[i].model.observe(xi)
            # Match Numba / NetworkSet.observe mode collapse
            outputs[i].append(obs.sum(axis=-1)[..., np.newaxis])
    return [np.stack(o, axis=0) for o in outputs]


def _run_nb(network_set: NetworkSet, nstep: int, x0_list: list) -> list:
    """Run NbHybridBackend and return per-step state data."""
    backend = NbHybridBackend()
    results = backend.run_network(
        network_set,
        nstep=nstep,
        chunk_size=1,
        initial_states=x0_list,
    )
    return [data for _, data, _ in results]


class TestSweepCfunParamSlotMismatch:
    """Regression tests for the prange sweep cfun param index bug.

    _cfun_get_param / _cfun_set_param use a descriptor index space that does
    not match the compiled kernel's cfun_params layout for SigmoidalJansenRit
    classic and HyperbolicTangent.  As a result, sweeping certain parameters
    writes into the wrong kernel slot.
    """

    def _make_jr_net(self, n=5, cfun=None, source_cvar=(1,)):
        model = JansenRit()
        model.configure()
        scheme = HeunDeterministic(dt=DT)
        sn = Subnetwork(name="jr", model=model, scheme=scheme, nnodes=n)
        w = _sparse_weights(n, seed=1)
        intra = IntraProjection(
            source_cvar=np.array(source_cvar, dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w,
            lengths=sp.csr_matrix(w.toarray() * 0.0),
            cv=1.0,
            dt=DT,
            scale=1.0,
            cfun=cfun,
        )
        sn.projections = [intra]
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()
        return ns

    def _sweep_param(self, param_name, values, nstep=20):
        cfun = SigmoidalJansenRit(
            a=np.array([1.0]),
            cmin=np.array([0.0]),
            cmax=np.array([2.0]),
            r=np.array([1.0]),
            midpoint=np.array([0.5]),
            use_classic=1,
        )
        ns = self._make_jr_net(cfun=cfun, source_cvar=(1, 2))
        backend = NbHybridBackend()
        key = f"jr.intra.{param_name}"
        seq = backend.sweep(
            ns, params={key: np.array(values)},
            nstep=nstep, backend="cpu", n_workers=1,
        )
        par = backend.sweep(
            ns, params={key: np.array(values)},
            nstep=nstep, backend="cpu", n_workers=4,
        )
        return seq, par

    def test_sjr_sweep_cmin_matches_sequential(self):
        """Sweeping SJR cmin must change only cmin, not corrupt midpoint."""
        seq, par = self._sweep_param("cmin", [0.0, 1.0])
        # Sequential output should differ between the two cmin values.
        assert not np.allclose(seq.merged_tavg[0], seq.merged_tavg[1], atol=1e-6)
        # Prange output must match sequential (and must also differ).
        np.testing.assert_allclose(
            seq.merged_tavg,
            par.merged_tavg,
            atol=1e-5, rtol=1e-5,
            err_msg="prange SJR cmin sweep does not match sequential (slot corruption)",
        )

    def test_sjr_sweep_midpoint_matches_sequential(self):
        """Sweeping SJR midpoint must not be aliased to cmin slot."""
        seq, par = self._sweep_param("midpoint", [0.0, 1.0])
        np.testing.assert_allclose(
            seq.merged_tavg,
            par.merged_tavg,
            atol=1e-5, rtol=1e-5,
            err_msg="prange SJR midpoint sweep does not match sequential",
        )

    def test_tanh_sweep_b_matches_sequential(self):
        """Sweeping HyperbolicTangent b must land in the kernel b slot."""
        cfun = HyperbolicTangent(
            a=np.array([1.0]),
            b=np.array([0.0]),
            midpoint=np.array([0.0]),
            sigma=np.array([1.0]),
        )
        ns = self._make_jr_net(cfun=cfun)
        backend = NbHybridBackend()
        key = "jr.intra.b"
        seq = backend.sweep(
            ns, params={key: np.array([0.5, 1.0])},
            nstep=20, backend="cpu", n_workers=1,
        )
        par = backend.sweep(
            ns, params={key: np.array([0.5, 1.0])},
            nstep=20, backend="cpu", n_workers=4,
        )
        # The sequential result must actually change.
        assert not np.allclose(seq.merged_tavg[0], seq.merged_tavg[1], atol=1e-6)
        np.testing.assert_allclose(
            seq.merged_tavg,
            par.merged_tavg,
            atol=1e-5, rtol=1e-5,
            err_msg="prange tanh b sweep writes into wrong cfun_params slot",
        )


class TestNumbaCvarMapping:
    """Regression tests for the target_cvar_cpl / non-identity cvar mapping."""

    def test_non_identity_cvar_matches_python(self):
        """ReducedSetFitzHughNagumo has cvar=[0,2]; Numba must match Python."""
        n = 4
        model = ReducedSetFitzHughNagumo()
        model.configure()
        assert np.array_equal(model.cvar, [0, 2])
        scheme = HeunDeterministic(dt=DT)
        sn = Subnetwork(name="rsfn", model=model, scheme=scheme, nnodes=n)
        w = _sparse_weights(n, seed=2)
        intra = IntraProjection(
            source_cvar=np.array([0, 2], dtype=np.int_),
            target_cvar=np.array([0, 1], dtype=np.int_),
            weights=w,
            lengths=sp.csr_matrix(w.toarray() * 0.0),
            cv=1.0,
            dt=DT,
            scale=1.0,
            cfun=Linear(a=np.array([0.5]), b=np.array([0.1])),
        )
        sn.projections = [intra]
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        rng = np.random.RandomState(17)
        x0 = rng.randn(model.nvar, n, model.number_of_modes).astype(np.float64)

        py = _run_python_loop(ns, nstep=8, x0_list=[x0])[0]
        nb = _run_nb(ns, nstep=8, x0_list=[x0])[0]

        assert py.shape == nb.shape, f"shape mismatch: {py.shape} vs {nb.shape}"
        np.testing.assert_allclose(
            nb, py,
            rtol=1e-3, atol=1e-4,
            err_msg="Numba output differs from Python for non-identity model.cvar",
        )


class TestStimulusSlotTargeting:
    """Regression test for the Numba stimulus broadcast bug."""

    def test_stimulus_targets_only_requested_cvar_slot(self):
        """A stimulus targeting one coupling slot must not leak into others."""
        from tvb.simulator.hybrid.stimulus_utils import constant_stim

        n = 5
        # JansenRit has cvar=[1,2] so the coupling array has 2 slots.
        model = JansenRit()
        model.configure()
        scheme = HeunDeterministic(dt=DT)
        sn = Subnetwork(name="jr", model=model, scheme=scheme, nnodes=n)
        # No projections: coupling starts at zero; stimulus is the only input.
        sn.projections = []
        sn.configure()

        # Stimulus targets coupling slot 0 only.
        from tvb.datatypes.patterns import StimuliRegion
        from tvb.datatypes import equations as eqs
        temporal = eqs.Linear()
        temporal.parameters["a"] = 0.0
        temporal.parameters["b"] = 1.0
        weight = np.zeros(n)
        weight[0] = 1.0
        pattern = StimuliRegion.from_weights(weight=weight, temporal=temporal)
        stim = sn.add_stimulus(pattern, stimulus_cvar=0, projection_scale=1.0)
        stim.configure(simulation_length=10 * DT)

        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        rng = np.random.RandomState(3)
        x0 = rng.randn(model.nvar, n, 1).astype(np.float64)

        py = _run_python_loop(ns, nstep=8, x0_list=[x0])[0]
        nb = _run_nb(ns, nstep=8, x0_list=[x0])[0]

        # Slot 0 (Voi index depends on observe).  The Python reference only
        # writes into slot 0; the Numba backend used to broadcast into all
        # slots.  Compare full trajectories.
        np.testing.assert_allclose(
            nb, py,
            rtol=1e-3, atol=1e-4,
            err_msg="Numba stimulus leaked into non-target cvar slots",
        )


class TestSweepNoiseSharing:
    """Regression test for shared noise across prange sweep configurations."""

    def _make_stochastic_net(self, seed):
        from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
        model = MontbrioPazoRoxin()
        model.configure()
        noise = Additive(nsig=np.array([0.1]))
        noise.noise_seed = seed
        noise.random_stream = np.random.RandomState(seed)
        noise.configure_white(DT)
        scheme = HeunStochastic(dt=DT, noise=noise)
        scheme.configure_boundaries(model)
        sn = Subnetwork(name="mpr", model=model, scheme=scheme, nnodes=4)
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()
        return ns

    def test_prange_sweep_uses_independent_noise_per_config(self):
        """Sequential sweep differs with noise seed; prange does not (bug)."""
        ns1 = self._make_stochastic_net(seed=42)
        ns2 = self._make_stochastic_net(seed=43)
        backend = NbHybridBackend()

        # Add a dummy projection so "coupling_scale" resolves.
        sn1 = ns1.subnets[0]
        w = sp.csr_matrix(np.zeros((4, 4)))
        intra = IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w,
            lengths=w.copy(),
            cv=1.0,
            dt=DT,
            scale=1.0,
            cfun=Linear(a=np.array([1.0])),
        )
        sn1.projections = [intra]
        sn1.configure()
        ns1.configure()

        sn2 = ns2.subnets[0]
        sn2.projections = [intra]
        sn2.configure()
        ns2.configure()

        # Sequential path: each config uses its own noise seed → outputs differ.
        seq1 = backend.sweep(
            ns1, params={"coupling_scale": np.array([1.0])},
            nstep=30, backend="cpu", n_workers=1,
        )
        seq2 = backend.sweep(
            ns2, params={"coupling_scale": np.array([1.0])},
            nstep=30, backend="cpu", n_workers=1,
        )
        assert not np.allclose(
            seq1.merged_tavg.mean(axis=1), seq2.merged_tavg.mean(axis=1),
            atol=1e-6, rtol=1e-6,
        ), "sequential stochastic sweeps are unexpectedly identical"

        # Prange path currently shares one noise array across configurations,
        # so different seeds incorrectly produce identical output.
        par1 = backend.sweep(
            ns1, params={"coupling_scale": np.array([1.0])},
            nstep=30, backend="cpu", n_workers=4,
        )
        par2 = backend.sweep(
            ns2, params={"coupling_scale": np.array([1.0])},
            nstep=30, backend="cpu", n_workers=4,
        )
        assert not np.allclose(
            par1.merged_tavg, par2.merged_tavg,
            atol=1e-6, rtol=1e-6,
        ), "prange sweep shares the same noise across configurations (different seeds give identical output)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
