"""Public C++ execution must not load the Numba hybrid compiler backend."""

import json
import subprocess
import sys
import textwrap

import pytest


def test_public_delayed_monitored_run_without_nb_hybrid():
    pytest.importorskip("tvb.simulator.backend.cpp_hybrid")
    script = textwrap.dedent(r'''
        import importlib.abc
        import json
        import sys

        blocked = []

        class BlockNumbaHybrid(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "tvb.simulator.backend.nb_hybrid" or fullname.startswith(
                    "tvb.simulator.backend.nb_hybrid."
                ):
                    blocked.append(fullname)
                    raise ImportError("nb_hybrid deliberately blocked")

        assert "tvb.simulator.backend.nb_hybrid" not in sys.modules
        sys.meta_path.insert(0, BlockNumbaHybrid())

        import numba  # The package itself must remain available.
        import numpy as np
        import scipy.sparse as sp
        from tvb.simulator.backend.cpp_hybrid import CppHybridBackend
        from tvb.simulator.backend.cpp_hybrid import backend, notebook
        from tvb.simulator.hybrid.coupling import Linear as LinearCoupling
        from tvb.simulator.hybrid.inter_projection import InterProjection
        from tvb.simulator.hybrid.network import NetworkSet
        from tvb.simulator.hybrid.simulator import Simulator
        from tvb.simulator.hybrid.subnetwork import Subnetwork
        from tvb.simulator.integrators import HeunDeterministic
        from tvb.simulator.models.linear import Linear
        from tvb.simulator.monitors import GlobalAverage, Raw, TemporalAverage

        def network(delayed=True):
            subnets = []
            for name, indices in [("source", [1]), ("target", [0])]:
                model = Linear(gamma=np.array([-0.1]))
                model.configure()
                sn = Subnetwork(name=name, model=model, nnodes=1,
                                node_indices=np.array(indices),
                                scheme=HeunDeterministic(dt=0.1))
                sn.configure()
                subnets.append(sn)
            projection = InterProjection(
                source=subnets[0], target=subnets[1],
                source_cvar=np.array([0]), target_cvar=np.array([0]),
                weights=sp.csr_matrix([[0.5]]),
                lengths=sp.csr_matrix([[0.2 if delayed else 0.0]]),
                cfun=LinearCoupling(), cv=1.0, dt=0.1,
            )
            ns = NetworkSet(subnets=subnets, projections=[projection])
            ns.configure()
            if delayed:
                assert np.any(projection.idelays > 0)
            return ns

        def run(kind, delayed=True):
            sim = Simulator(
                nets=network(delayed), simulation_length=1.2, backend=kind,
                monitors=[Raw(), TemporalAverage(period=0.2),
                          GlobalAverage(period=0.1)],
            )
            sim.configure()
            return sim.run(initial_conditions=[
                np.array([[[1.0]]]), np.array([[[0.0]]]),
            ])

        expected = run("python")
        actual = run("cpp")
        for (t, d), (reference_t, reference_d) in zip(actual, expected):
            np.testing.assert_allclose(t, reference_t, atol=1e-7)
            np.testing.assert_allclose(d, reference_d, rtol=1e-5, atol=1e-6)
            assert np.isfinite(d).all()
        assert [d.shape for _, d in actual] == [
            (12, 1, 2, 1), (6, 1, 2, 1), (12, 1, 1, 1),
        ]
        # Ensure this fixture observes a real delayed trajectory, not a
        # zero-weight or constant-state projection that masks delay bugs.
        assert not np.allclose(actual[0][1], run("cpp", delayed=False)[0][1])
        ns = network()
        descriptor, values = CppHybridBackend()._resolve_named_params(
            ns, {"coupling_scale": [0.5, 1.0]})
        assert descriptor[0]["projection"] == "source_to_target"
        assert values.shape == (2, 1)
        assert "coupling_scale" in notebook.available_sweep_params(ns)["coupling"]
        CppHybridBackend.clear_cache()
        assert not blocked, blocked  # includes best-effort imports that got caught
        assert "tvb.simulator.backend.nb_hybrid" not in sys.modules
        assert "numba" in sys.modules
        print(json.dumps({"backend_file": backend.__file__,
                          "shapes": [list(d.shape) for _, d in actual],
                          "blocked_attempts": blocked}))
    ''')
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True,
        check=False, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evidence = json.loads(completed.stdout.splitlines()[-1])
    assert evidence["blocked_attempts"] == []
    print(json.dumps(evidence))
