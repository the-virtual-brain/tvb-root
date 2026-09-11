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
Tests for the per-target-cvar ``target_scales`` feature of
:class:`~tvb.simulator.hybrid.BaseProjection`.

Four cases are covered:

A. ``target_scales=None`` (default) — output must be identical to a projection
   that uses only the global ``scale``.
B. **Broadcast topology** (1 source → N targets) — each target slot receives a
   distinct fraction of the signal proportional to its ``target_scales[i]``.
C. **Element-wise topology** (N sources → N targets) — each (src, tgt) pair is
   independently scaled.
D. **Wrong-length ``target_scales``** — ``configure_buffer()`` must raise
   ``ValueError``.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.models import JansenRit
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.hybrid import Subnetwork, NetworkSet


# ---------------------------------------------------------------------------
# Helper: build a minimal two-subnetwork setup
# ---------------------------------------------------------------------------

def _make_subnets(n_src=4, n_tgt=3):
    """Return a configured (source, target) pair of JansenRit subnetworks."""
    scheme = HeunDeterministic(dt=0.1)
    src = Subnetwork(
        name="src", model=JansenRit(), scheme=scheme, nnodes=n_src
    ).configure()
    tgt = Subnetwork(
        name="tgt", model=JansenRit(), scheme=scheme, nnodes=n_tgt
    ).configure()
    return src, tgt


def _make_weights(n_tgt, n_src, seed=7):
    """Return a dense-to-sparse random weight matrix."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.1, 1.0, size=(n_tgt, n_src))
    return sp.csr_matrix(W)


def _projection_output(proj, src_subnet, n_target_cvars, n_tgt, n_modes=1, steps=5):
    """
    Run *steps* buffer updates and one apply(); return the target coupling array.
    """
    proj.configure_buffer(
        src_subnet.model.nvar, src_subnet.nnodes, src_subnet.model.number_of_modes
    )
    rng = np.random.default_rng(0)
    state = rng.standard_normal(
        (src_subnet.model.nvar, src_subnet.nnodes, src_subnet.model.number_of_modes)
    ).astype("f")
    for i in range(steps):
        proj.update_buffer(state, i)
    tgt_arr = np.zeros((n_target_cvars, n_tgt, n_modes))
    proj.apply(tgt_arr, steps - 1)
    return tgt_arr


# ---------------------------------------------------------------------------
# Case A: target_scales=None is a no-op
# ---------------------------------------------------------------------------

class TestTargetScalesNone:
    """target_scales=None leaves the projection behaviour unchanged."""

    def test_none_is_regression_safe(self):
        """With target_scales=None output equals a projection without the field."""
        src, tgt = _make_subnets()
        W = _make_weights(tgt.nnodes, src.nnodes)
        lengths = sp.csr_matrix((tgt.nnodes, src.nnodes))

        def _make(ts):
            kwargs = dict(
                source=src, target=tgt,
                source_cvar=np.r_[0], target_cvar=np.r_[0, 1],
                weights=W.copy(), lengths=lengths.copy(), cv=3.0, dt=0.1, scale=0.5,
            )
            if ts is not None:
                kwargs["target_scales"] = ts
            return InterProjection(**kwargs)

        proj_no_ts = _make(None)
        # target_scales=[1, 1] should reproduce the same numbers as no target_scales
        proj_ones = _make(np.array([1.0, 1.0]))

        n_cvar = len(tgt.model.cvar)
        out_no_ts = _projection_output(proj_no_ts, src, n_cvar, tgt.nnodes)
        out_ones  = _projection_output(proj_ones,  src, n_cvar, tgt.nnodes)

        np.testing.assert_allclose(
            out_no_ts, out_ones, rtol=1e-6,
            err_msg="target_scales=[1,1] must produce identical output to no target_scales",
        )


# ---------------------------------------------------------------------------
# Case B: broadcast (1 src → N targets)
# ---------------------------------------------------------------------------

class TestTargetScalesBroadcast:
    """
    Broadcast topology: source_cvar.size == 1, multiple target cvars.

    Each target slot should receive a signal scaled by its own
    ``target_scales[i]``, with no cross-contamination.
    """

    def test_broadcast_scales_are_independent(self):
        """
        With source_cvar=[0] (single) → target_cvar=[0,1], each target slot
        receives a distinct fraction of the input proportional to its scale.

        Concretely: with target_scales=[2.0, 0.5] the first slot output must
        be exactly 4× the second slot output, matching the ratio 2.0/0.5.
        """
        src, tgt = _make_subnets()
        W = _make_weights(tgt.nnodes, src.nnodes)
        lengths = sp.csr_matrix((tgt.nnodes, src.nnodes))
        n_cvar = len(tgt.model.cvar)

        ts = np.array([2.0, 0.5])
        proj_scaled = InterProjection(
            source=src, target=tgt,
            source_cvar=np.r_[0], target_cvar=np.r_[0, 1],
            weights=W.copy(), lengths=lengths.copy(), cv=3.0, dt=0.1, scale=1.0,
            target_scales=ts,
        )
        # Reference projection without target_scales (scale=1.0)
        proj_ref = InterProjection(
            source=src, target=tgt,
            source_cvar=np.r_[0], target_cvar=np.r_[0, 1],
            weights=W.copy(), lengths=lengths.copy(), cv=3.0, dt=0.1, scale=1.0,
        )

        out_scaled = _projection_output(proj_scaled, src, n_cvar, tgt.nnodes)
        out_ref    = _projection_output(proj_ref,    src, n_cvar, tgt.nnodes)

        # Slot 0 should be 2.0× reference slot 0
        np.testing.assert_allclose(
            out_scaled[0], out_ref[0] * ts[0], rtol=1e-5,
            err_msg="Broadcast slot 0 should be target_scales[0] * reference",
        )
        # Slot 1 should be 0.5× reference slot 1
        np.testing.assert_allclose(
            out_scaled[1], out_ref[1] * ts[1], rtol=1e-5,
            err_msg="Broadcast slot 1 should be target_scales[1] * reference",
        )

    def test_broadcast_zero_scale_silences_slot(self):
        """target_scales[0]=0.0 must produce zero contribution on that target slot."""
        src, tgt = _make_subnets()
        W = _make_weights(tgt.nnodes, src.nnodes)
        lengths = sp.csr_matrix((tgt.nnodes, src.nnodes))
        n_cvar = len(tgt.model.cvar)

        proj = InterProjection(
            source=src, target=tgt,
            source_cvar=np.r_[0], target_cvar=np.r_[0, 1],
            weights=W.copy(), lengths=lengths.copy(), cv=3.0, dt=0.1, scale=1.0,
            target_scales=np.array([0.0, 1.0]),
        )
        out = _projection_output(proj, src, n_cvar, tgt.nnodes)
        np.testing.assert_array_equal(
            out[0], np.zeros_like(out[0]),
            err_msg="Slot 0 with target_scales=0.0 must be zero",
        )
        assert np.any(out[1] != 0), "Slot 1 with target_scales=1.0 must be non-zero"


# ---------------------------------------------------------------------------
# Case C: element-wise (M src → M targets)
# ---------------------------------------------------------------------------

class TestTargetScalesElementWise:
    """
    Element-wise topology: source_cvar.size == target_cvar.size > 1.

    Each (src[i], tgt[i]) pair is independently scaled by target_scales[i].
    """

    def test_elementwise_independent_scaling(self):
        """
        With source_cvar=[0,0], target_cvar=[0,1] and target_scales=[3.0,0.0]
        only the first target slot receives non-zero input.
        """
        src, tgt = _make_subnets()
        W = _make_weights(tgt.nnodes, src.nnodes)
        lengths = sp.csr_matrix((tgt.nnodes, src.nnodes))
        n_cvar = len(tgt.model.cvar)

        ts = np.array([3.0, 0.0])
        proj = InterProjection(
            source=src, target=tgt,
            source_cvar=np.r_[0, 0], target_cvar=np.r_[0, 1],
            weights=W.copy(), lengths=lengths.copy(), cv=3.0, dt=0.1, scale=1.0,
            target_scales=ts,
        )
        proj_ref = InterProjection(
            source=src, target=tgt,
            source_cvar=np.r_[0, 0], target_cvar=np.r_[0, 1],
            weights=W.copy(), lengths=lengths.copy(), cv=3.0, dt=0.1, scale=1.0,
        )

        out = _projection_output(proj, src, n_cvar, tgt.nnodes)
        out_ref = _projection_output(proj_ref, src, n_cvar, tgt.nnodes)

        # Slot 0: should be 3× the reference value
        np.testing.assert_allclose(
            out[0], out_ref[0] * ts[0], rtol=1e-5,
            err_msg="Element-wise slot 0 should be target_scales[0] * reference",
        )
        # Slot 1: should be zero (target_scales[1] = 0)
        np.testing.assert_array_equal(
            out[1], np.zeros_like(out[1]),
            err_msg="Element-wise slot 1 with target_scales=0 should be zero",
        )

    def test_elementwise_multiplicative_with_global_scale(self):
        """
        Effective per-target scale is global ``scale`` × ``target_scales[i]``.

        A projection with scale=2.0 and target_scales=[1.0, 3.0] must
        produce slot 0 = 2× reference and slot 1 = 6× reference.
        """
        src, tgt = _make_subnets()
        W = _make_weights(tgt.nnodes, src.nnodes)
        lengths = sp.csr_matrix((tgt.nnodes, src.nnodes))
        n_cvar = len(tgt.model.cvar)

        proj_scaled = InterProjection(
            source=src, target=tgt,
            source_cvar=np.r_[0, 0], target_cvar=np.r_[0, 1],
            weights=W.copy(), lengths=lengths.copy(), cv=3.0, dt=0.1,
            scale=2.0, target_scales=np.array([1.0, 3.0]),
        )
        proj_ref = InterProjection(
            source=src, target=tgt,
            source_cvar=np.r_[0, 0], target_cvar=np.r_[0, 1],
            weights=W.copy(), lengths=lengths.copy(), cv=3.0, dt=0.1, scale=1.0,
        )

        out_scaled = _projection_output(proj_scaled, src, n_cvar, tgt.nnodes)
        out_ref    = _projection_output(proj_ref,    src, n_cvar, tgt.nnodes)

        np.testing.assert_allclose(out_scaled[0], out_ref[0] * 2.0, rtol=1e-5,
                                   err_msg="slot 0: effective scale should be 2.0×1.0=2.0")
        np.testing.assert_allclose(out_scaled[1], out_ref[1] * 6.0, rtol=1e-5,
                                   err_msg="slot 1: effective scale should be 2.0×3.0=6.0")


# ---------------------------------------------------------------------------
# Case D: wrong-length target_scales raises ValueError
# ---------------------------------------------------------------------------

class TestTargetScalesValidation:
    """configure_buffer() must reject target_scales with the wrong length."""

    def test_wrong_length_raises(self):
        """target_scales with len != len(target_cvar) must raise ValueError."""
        src, tgt = _make_subnets()
        W = _make_weights(tgt.nnodes, src.nnodes)
        lengths = sp.csr_matrix((tgt.nnodes, src.nnodes))

        proj = InterProjection(
            source=src, target=tgt,
            source_cvar=np.r_[0], target_cvar=np.r_[0, 1],  # 2 targets
            weights=W.copy(), lengths=lengths.copy(), cv=3.0, dt=0.1, scale=1.0,
            target_scales=np.array([1.0, 2.0, 3.0]),  # 3 scales — wrong!
        )
        with pytest.raises(ValueError, match="target_scales length"):
            proj.configure_buffer(
                src.model.nvar, src.nnodes, src.model.number_of_modes
            )

    def test_correct_length_does_not_raise(self):
        """target_scales matching target_cvar length must not raise."""
        src, tgt = _make_subnets()
        W = _make_weights(tgt.nnodes, src.nnodes)
        lengths = sp.csr_matrix((tgt.nnodes, src.nnodes))

        proj = InterProjection(
            source=src, target=tgt,
            source_cvar=np.r_[0], target_cvar=np.r_[0, 1],
            weights=W.copy(), lengths=lengths.copy(), cv=3.0, dt=0.1, scale=1.0,
            target_scales=np.array([1.0, 2.0]),  # correct length
        )
        proj.configure_buffer(src.model.nvar, src.nnodes, src.model.number_of_modes)
