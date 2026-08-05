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
# PARTICULAR PURPOSE.  See the GNU General Public License for more details. You should
# have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#

"""
Regression tests for the hybrid-simulator coupling pipeline.

The original ``BaseProjection.apply()`` (and the Numba backend) applied the
coupling ``pre()`` transform *after* the weighted sum over source nodes:

    post(scale * pre(Σ_j w_ij · x_j))

instead of the correct per-edge order:

    scale * post(Σ_j w_ij · pre(x_j))

This bug is invisible when ``pre()`` is the identity (Linear, Scaling,
Sigmoidal) but produces wrong results for nonlinear ``pre()`` functions such as
HyperbolicTangent, SigmoidalJansenRit, and PreSigmoidal.

These tests verify that the corrected pipeline is in place and will **fail**
if the pre/post ordering is ever reverted.

See Also
--------
FIX_COUPLING_PIPELINE.md — design doc with quantitative examples.
"""

import numpy as np
import pytest
from scipy import sparse as sp

from tvb.simulator.hybrid.base_projection import BaseProjection
from tvb.simulator.hybrid.coupling import HyperbolicTangent, Linear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ring_weights(n_nodes: int, weight: float = 1.0, dtype=np.float32):
    """Return a sparse CSR weight matrix for a directed n-node ring where
    each node receives from every other node with the given weight.
    """
    rows, cols, data = [], [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
                data.append(weight)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=dtype)


class TestCouplingPipelineCorrectness:
    """NumPy-backend tests that ``BaseProjection.apply()`` applies ``pre()``
    per-edge *before* the weighted sum.
    """

    def test_tanh_pre_is_per_edge(self):
        """HyperbolicTangent ``pre()`` must be evaluated on each individual
        delayed source state *before* weighting and summation.

        Setup: 3-node ring, unit weights, states [1.0, 2.0, -1.0].
        With ``pre(x) = 1 + tanh(x)``, the correct coupling at each target is
        ``Σ_j w_ij · (1 + tanh(x_j))``.  If ``pre()`` were applied after the
        sum (the old bug), the result would be ``1 + tanh(Σ_j w_ij · x_j)``,
        which differs by ~0.44 for these values.
        """
        n_nodes = 3
        weights = _make_ring_weights(n_nodes, weight=1.0, dtype=np.float32)
        lengths = sp.csr_matrix((n_nodes, n_nodes), dtype=np.float32)

        cfun = HyperbolicTangent(
            a=np.array([1.0]), midpoint=np.array([0.0]), sigma=np.array([1.0])
        )

        proj = BaseProjection(
            weights=weights,
            lengths=lengths,
            source_cvar=np.array([0]),
            target_cvar=np.array([0]),
            scale=1.0,
            cfun=cfun,
            cv=1.0,
            dt=0.1,
        )
        proj.configure_buffer(n_vars_src=2, n_nodes_src=n_nodes, n_modes_src=1)

        state = np.zeros((2, n_nodes, 1), dtype=np.float32)
        state[0, :, 0] = [1.0, 2.0, -1.0]
        proj.update_buffer(state, t=0)

        tgt = np.zeros((2, n_nodes, 1), dtype=np.float32)
        mode_map = np.eye(1, dtype=np.int_)
        proj.apply(tgt, t=1, mode_map=mode_map)

        # Compute the correct reference using the *already-eps-trick* weights
        W = proj.weights.toarray().astype(np.float32)
        x = state[0, :, 0]
        pre_x = cfun.pre(x).ravel()  # per-edge pre-transform
        correct = np.dot(W, pre_x)   # Σ w · pre(x_j)
        actual = tgt[0, :, 0]

        np.testing.assert_allclose(
            actual, correct, rtol=1e-4, atol=1e-5,
            err_msg=(
                "BaseProjection.apply() does not compute scale*post(Σ w*pre(x)). "
                "The pre() function is likely being applied after the weighted sum "
                "instead of per-edge before it."
            ),
        )

    def test_linear_coupling_unaffected(self):
        """Linear coupling with default ``a=1, b=0`` has ``pre()`` = identity,
        so the pipeline order does not matter.  This control test should pass
        both before and after any fix.
        """
        n_nodes = 3
        weights = _make_ring_weights(n_nodes, weight=1.0, dtype=np.float32)
        lengths = sp.csr_matrix((n_nodes, n_nodes), dtype=np.float32)

        cfun = Linear()  # defaults: a=1.0, b=0.0 → identity

        proj = BaseProjection(
            weights=weights,
            lengths=lengths,
            source_cvar=np.array([0]),
            target_cvar=np.array([0]),
            scale=1.0,
            cfun=cfun,
            cv=1.0,
            dt=0.1,
        )
        proj.configure_buffer(n_vars_src=2, n_nodes_src=n_nodes, n_modes_src=1)

        state = np.zeros((2, n_nodes, 1), dtype=np.float32)
        state[0, :, 0] = [1.0, 2.0, -1.0]
        proj.update_buffer(state, t=0)

        tgt = np.zeros((2, n_nodes, 1), dtype=np.float32)
        mode_map = np.eye(1, dtype=np.int_)
        proj.apply(tgt, t=1, mode_map=mode_map)

        W = proj.weights.toarray().astype(np.float32)
        x = state[0, :, 0]
        correct = np.dot(W, x)   # identity → just weighted sum
        actual = tgt[0, :, 0]

        np.testing.assert_allclose(actual, correct, rtol=1e-4, atol=1e-5)


class TestNumbaCouplingPipeline:
    """Verify that the Numba backend also applies ``pre()`` per-edge.

    The Numba simulation kernel (``nb-hybrid-sim.py.mako``) was updated to
    evaluate ``pre()`` inside the per-edge inner loop before weighting and
    accumulation.  This test runs a minimal one-step simulation and checks
    the first-step coupling values against a hand-computed reference.
    """

    @pytest.mark.skipif(
        pytest.importorskip("numba", reason="numba not installed") is None,
        reason="numba not available",
    )
    def test_numba_tanh_pre_per_edge(self):
        """Run a 1-step NbHybridBackend simulation with MontbrioPazoRoxin
        and HyperbolicTangent intra-coupling.  ``ctavg`` must match the
        per-edge-pre reference.
        """
        from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
        from tvb.simulator.integrators import HeunDeterministic
        from tvb.simulator.hybrid import Subnetwork, NetworkSet
        from tvb.simulator.hybrid.intra_projection import IntraProjection
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        model = MontbrioPazoRoxin()
        model.configure()
        scheme = HeunDeterministic(dt=0.1)
        subnet = Subnetwork(name="mpr", model=model, scheme=scheme, nnodes=3)

        w = sp.csr_matrix(
            np.array(
                [[0.0, 0.5, 0.5],
                 [0.5, 0.0, 0.5],
                 [0.5, 0.5, 0.0]],
                dtype=np.float64,
            )
        )
        l = sp.csr_matrix((3, 3), dtype=np.float64)

        cfun = HyperbolicTangent(
            a=np.array([1.0]), midpoint=np.array([0.0]), sigma=np.array([1.0])
        )
        proj = IntraProjection(
            source_cvar=model.cvar[:1],
            target_cvar=np.array([0], dtype=np.int32),
            weights=w,
            lengths=l,
            cv=1.0,
            dt=0.1,
            scale=1.0,
            cfun=cfun,
        )
        subnet.projections = [proj]
        subnet.configure()

        nets = NetworkSet(subnets=[subnet], projections=[])
        nets.configure()

        # Deterministic ICs so we can compute the reference by hand
        ic = np.zeros((model.nvar, 3, 1), dtype=np.float64)
        ic[0, :, 0] = [0.1, 0.2, 0.3]   # r (cvar)
        ic[1, :, 0] = [0.0, 0.0, 0.0]   # V

        be = NbHybridBackend()
        result = be.run_network(nets, nstep=1, initial_states=[ic])
        times, data, ctavg = result[0]

        # Reference using the CORRECT pipeline: scale * post(Σ w · pre(x_j))
        r = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        pre_r = 1.0 + np.tanh(r)
        W = w.toarray()
        correct = W @ pre_r

        actual = ctavg[0, 0, :, 0]
        np.testing.assert_allclose(
            actual, correct, rtol=1e-4, atol=1e-5,
            err_msg=(
                "Numba backend coupling does not match the per-edge-pre reference. "
                "The JIT-compiled kernel may be applying pre() after the weighted sum."
            ),
        )


class TestQuantitativeDiscrepancy:
    """Pure-numpy test (no TVB imports) demonstrating mathematically that
    ``Σ w·pre(x_j) ≠ pre(Σ w·x_j)`` for a nonlinear ``pre``.

    Uses the 4-node example from FIX_COUPLING_PIPELINE.md.
    """

    def test_nonlinear_pre_order_matters(self):
        """For a nonlinear ``pre`` (here ``1 + tanh(x)``), applying it before
        vs. after the weighted sum yields different results.  If the two
        quantities were equal, the pre/post ordering bug would be invisible.
        """
        W = np.array(
            [
                [0.0, 0.2, 0.3, 0.1],
                [0.2, 0.0, 0.1, 0.3],
                [0.3, 0.1, 0.0, 0.2],
                [0.1, 0.3, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        x = np.array([1.0, -0.5, 2.0, -1.0], dtype=np.float32)

        def pre(x):
            return 1.0 * (1.0 + np.tanh(x))

        correct = W @ pre(x)   # Σ w · pre(x_j)  — correct order
        buggy = pre(W @ x)     # pre(Σ w · x_j)  — old buggy order

        # The discrepancy must be non-negligible; otherwise the test is moot
        assert not np.allclose(correct, buggy, atol=1e-3), (
            "For a linear pre, pre(sum) == sum(pre).  With tanh they must differ. "
            "If they don't, the chosen weights/states are accidentally degenerate."
        )

        # Match the expected values from FIX_COUPLING_PIPELINE.md
        np.testing.assert_allclose(correct, [0.721, 0.620, 0.630, 0.730], atol=1e-3)
        np.testing.assert_allclose(buggy, [1.380, 1.100, 1.050, 1.336], atol=1e-3)
