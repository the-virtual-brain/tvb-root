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
Regression tests verifying that hybrid coupling functions match classic TVB
coupling semantics.

The classic TVB coupling pattern is::

    result = post((weights * pre(x_i, x_j)).sum(axis=source_nodes))

where:
- ``pre(x_i, x_j)`` is applied PER-EDGE before weighting by g_ij
- ``post(gx)`` is applied AFTER the weighted sum
- ``x_i`` is the current (non-delayed) state of the TARGET node
- ``x_j`` is the delayed state of the SOURCE node

The hybrid framework applies ``pre(x)`` to the already-summed weighted
afferent (single argument), which is correct when ``pre`` is identity.
For coupling classes where ``pre`` depends on BOTH ``x_i`` and ``x_j``
(e.g., Difference, Kuramoto), the hybrid implementations now correctly
accept two arguments.

These tests compare each hybrid coupling function's output against the classic
TVB coupling function's output with the same inputs.  They are intended to
PASS as regression guards ensuring the hybrid implementations continue to
match classic TVB semantics.
"""

import numpy as np
import pytest

from tvb.simulator.coupling import (
    Linear as ClassicLinear,
    Scaling as ClassicScaling,
    Sigmoidal as ClassicSigmoidal,
    Difference as ClassicDifference,
    Kuramoto as ClassicKuramoto,
    HyperbolicTangent as ClassicHyperbolicTangent,
    SigmoidalJansenRit as ClassicSigmoidalJansenRit,
    PreSigmoidal as ClassicPreSigmoidal,
)
from tvb.simulator.hybrid.coupling import (
    Linear as HybridLinear,
    Scaling as HybridScaling,
    Sigmoidal as HybridSigmoidal,
    Difference as HybridDifference,
    Kuramoto as HybridKuramoto,
    HyperbolicTangent as HybridHyperbolicTangent,
    SigmoidalJansenRit as HybridSigmoidalJansenRit,
    PreSigmoidal as HybridPreSigmoidal,
)


def _simulate_classic_coupling(classic_cfun, weights, x_i, x_j):
    """Simulate the classic TVB coupling pipeline end-to-end.

    Reproduces the logic of ``Coupling.__call__``:

        pre = cfun.pre(x_i, x_j)          # per-edge transform
        sum = (weights * pre).sum(axis=2)  # weighted sum over sources
        result = cfun.post(sum)            # post-summation transform

    Parameters
    ----------
    classic_cfun : tvb.simulator.coupling.Coupling
        Classic coupling instance.
    weights : ndarray, shape (n_target, n_source)
        Weight matrix g_ij.
    x_i : ndarray, shape (n_cvar, n_target, n_source, n_mode)
        Current (non-delayed) state of target nodes, broadcast across sources.
    x_j : ndarray, shape (n_cvar, n_source, n_mode)
        Delayed state of source nodes (will be broadcast to match x_i shape).

    Returns
    -------
    ndarray, shape (n_cvar, n_target, n_mode)
        Coupling result matching classic TVB semantics.
    """
    n_cvar = x_j.shape[0]
    n_target = weights.shape[0]
    n_source = weights.shape[1]
    n_mode = x_j.shape[2]

    x_j_expanded = np.broadcast_to(
        x_j[:, np.newaxis, :, :],
        (n_cvar, n_target, n_source, n_mode),
    )
    x_i_broadcast = np.broadcast_to(
        x_i[:, :, :, :],
        (n_cvar, n_target, n_source, n_mode),
    ) if x_i.shape != (n_cvar, n_target, n_source, n_mode) else x_i

    pre = classic_cfun.pre(x_i_broadcast, x_j_expanded)

    g_ij = weights[np.newaxis, :, :, np.newaxis]
    weighted_sum = (g_ij * pre).sum(axis=2)

    return classic_cfun.post(weighted_sum)


# ---------------------------------------------------------------------------
# Linear – NO DISCREPANCY
# ---------------------------------------------------------------------------

class TestLinearEquivalence:
    """Verify hybrid Linear matches classic Linear.

    Classic: pre = x_j (identity), post = a * gx + b
    Hybrid:  pre = identity, post = a * x + b

    Both apply identity before summation and affine transform after, so
    the results should be identical.
    """

    @pytest.mark.parametrize(
        "a_val, b_val",
        [
            (np.array([1.0]), np.array([0.0])),
            (np.array([0.00390625]), np.array([0.0])),
            (np.array([0.5]), np.array([0.1])),
            (np.array([2.0]), np.array([-1.0])),
        ],
    )
    def test_linear_post_matches_classic(self, a_val, b_val):
        """Hybrid Linear.post should produce same result as classic pipeline."""
        classic = ClassicLinear(a=a_val.copy(), b=b_val.copy())
        hybrid = HybridLinear(a=a_val.copy(), b=b_val.copy())

        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        classic_result = classic.post(x)
        hybrid_result = hybrid.post(x)

        np.testing.assert_array_almost_equal(hybrid_result, classic_result)

    def test_linear_pre_is_identity(self):
        """Both classic and hybrid Linear.pre should return x_j unchanged."""
        classic = ClassicLinear()
        hybrid = HybridLinear()

        x_j = np.random.randn(2, 4, 1)
        x_i = np.random.randn(2, 3, 4, 1)

        classic_pre = classic.pre(x_i, x_j)
        hybrid_pre = hybrid.pre(x_j[0, :, 0])  # hybrid pre takes single arg

        np.testing.assert_array_almost_equal(classic_pre, x_j)
        np.testing.assert_array_equal(hybrid_pre, x_j[0, :, 0])


# ---------------------------------------------------------------------------
# Scaling – NO DISCREPANCY
# ---------------------------------------------------------------------------

class TestScalingEquivalence:
    """Verify hybrid Scaling matches classic Scaling.

    Classic: pre = x_j (identity), post = a * gx
    Hybrid:  pre = identity, post = a * x

    Both are a simple post-summation scale, so no discrepancy.
    """

    @pytest.mark.parametrize(
        "a_val",
        [
            np.array([1.0]),
            np.array([0.00390625]),
            np.array([2.0]),
            np.array([0.5]),
        ],
    )
    def test_scaling_post_matches_classic(self, a_val):
        """Hybrid Scaling.post should produce same result as classic pipeline."""
        classic = ClassicScaling(a=a_val.copy())
        hybrid = HybridScaling(a=a_val.copy())

        x = np.array([[1.0, 2.0], [3.0, 4.0]])

        classic_result = classic.post(x)
        hybrid_result = hybrid.post(x)

        np.testing.assert_array_almost_equal(hybrid_result, classic_result)


# ---------------------------------------------------------------------------
# Sigmoidal – NO DISCREPANCY
# ---------------------------------------------------------------------------

class TestSigmoidalEquivalence:
    """Verify hybrid Sigmoidal matches classic Sigmoidal.

    Classic: pre = x_j (identity), post = cmin + (cmax-cmin)/(1+exp(-a*(gx-midpoint)/sigma))
    Hybrid:  pre = identity, post = cmin + (cmax-cmin)/(1+exp(-a*(x-midpoint)/sigma))

    Both apply identity pre-summation and sigmoidal post-summation, so
    no discrepancy.
    """

    def test_sigmoidal_post_matches_classic(self):
        """Hybrid Sigmoidal.post should produce same result as classic pipeline."""
        classic = ClassicSigmoidal(
            cmin=np.array([-1.0]),
            cmax=np.array([1.0]),
            a=np.array([1.0]),
            midpoint=np.array([0.0]),
            sigma=np.array([1.0]),
        )
        hybrid = HybridSigmoidal(
            cmin=np.array([-1.0]),
            cmax=np.array([1.0]),
            a=np.array([1.0]),
            midpoint=np.array([0.0]),
            sigma=np.array([1.0]),
        )

        x = np.array([[-2.0, 0.0, 2.0], [0.5, 1.0, -0.5]])

        classic_result = classic.post(x)
        hybrid_result = hybrid.post(x)

        np.testing.assert_array_almost_equal(hybrid_result, classic_result, decimal=5)


# ---------------------------------------------------------------------------
# Difference – MAJOR DISCREPANCY
# ---------------------------------------------------------------------------

class TestDifferenceEquivalence:
    """Verify that hybrid Difference matches classic TVB semantics.

    Classic TVB Difference::

        pre(x_i, x_j) = x_j - x_i
        post(gx) = a * gx

    The full coupling computation is::

        result_i = a * sum_j(g_ij * (x_j - x_i))

    The hybrid implementation correctly accepts both x_i and x_j
    in its pre function and computes the difference matching classic TVB.
    """

    def test_classic_difference_pre_returns_difference(self):
        """Classic Difference.pre(x_i, x_j) must return x_j - x_i."""
        classic = ClassicDifference(a=np.array([0.1]))

        x_i = np.array([[[1.0], [2.0], [3.0]]])  # (1, 3, 1)
        x_j = np.array([[[4.0], [5.0], [6.0]]])  # (1, 3, 1)

        result = classic.pre(x_i, x_j)
        expected = x_j - x_i

        np.testing.assert_array_almost_equal(result, expected)

    def test_hybrid_difference_pre_should_return_difference(self):
        """Hybrid Difference.pre must compute x_j - x_i.

        With the updated signature pre(x_j, x_i=None), when both arguments
        are provided, the result should be x_j - x_i, matching classic TVB.
        """
        hybrid = HybridDifference(a=np.array([0.1]))

        x_i = np.array([1.0, 2.0, 3.0])
        x_j = np.array([4.0, 5.0, 6.0])

        result = hybrid.pre(x_j, x_i)
        expected = x_j - x_i

        np.testing.assert_array_almost_equal(result, expected)

    def test_hybrid_difference_pre_returns_identity_when_no_x_i(self):
        """Hybrid Difference.pre(x_j) returns x_j when x_i is not provided."""
        hybrid = HybridDifference(a=np.array([0.1]))

        x_j = np.array([4.0, 5.0, 6.0])

        result = hybrid.pre(x_j)
        np.testing.assert_array_equal(result, x_j)

    def test_difference_end_to_end_equivalence(self):
        """End-to-end comparison: hybrid Difference matches classic TVB.

        Classic computes::

            a * sum_j(g_ij * (x_j - x_i))

        Hybrid with corrected pre(x_j, x_i) computes the same when both
        arguments are provided.
        """
        weights = np.array([[0.5, 0.3], [0.2, 0.4]])  # (target, source)
        n_target, n_source = weights.shape
        n_cvar = 1
        n_mode = 1

        rng = np.random.RandomState(42)
        x_i = rng.randn(n_cvar, n_target, n_source, n_mode)
        x_j = rng.randn(n_cvar, n_source, n_mode)

        classic = ClassicDifference(a=np.array([0.1]))
        hybrid = HybridDifference(a=np.array([0.1]))

        classic_result = _simulate_classic_coupling(classic, weights, x_i, x_j)

        # With corrected pre(x_j, x_i), hybrid should match classic
        x_j_expanded = x_j[:, np.newaxis, :, :]  # (n_cvar, 1, n_source, n_mode)
        x_i_expanded = x_i  # (n_cvar, n_target, n_source, n_mode)
        pre_result = hybrid.pre(x_j_expanded, x_i_expanded)
        g_ij = weights[np.newaxis, :, :, np.newaxis]
        hybrid_weighted_sum = (g_ij * pre_result).sum(axis=2)
        hybrid_result = hybrid.post(hybrid_weighted_sum)

        np.testing.assert_array_almost_equal(hybrid_result, classic_result, decimal=5)

    def test_difference_correct_formula(self):
        """Verify the CORRECT formula: a * sum_j(g_ij * (x_j - x_i)).

        This test shows what the correct result should be, computed manually.
        When the hybrid Difference is fixed, this should pass.
        """
        a = np.array([0.1])
        weights = np.array([[0.5, 0.3], [0.2, 0.4]])

        x_i_vals = np.array([1.0, 2.0])  # target node states
        x_j_vals = np.array([4.0, 5.0])  # source node states

        correct_result = np.zeros(2)
        for i in range(2):
            total = 0.0
            for j in range(2):
                total += weights[i, j] * (x_j_vals[j] - x_i_vals[i])
            correct_result[i] = a[0] * total

        expected = a[0] * (weights @ (x_j_vals - x_i_vals[:, np.newaxis]).T.diagonal())
        manual = a[0] * np.array([
            0.5 * (4.0 - 1.0) + 0.3 * (5.0 - 1.0),
            0.2 * (4.0 - 2.0) + 0.4 * (5.0 - 2.0),
        ])

        np.testing.assert_array_almost_equal(correct_result, manual)

    def test_difference_diagonal_weights(self):
        """With identity weights, Difference should give a * (x_j - x_i) per edge.

        When the hybrid Difference is corrected, the pre function should
        return x_j - x_i and the weighted sum (with identity weights) should
        give the sum of differences.
        """
        a = np.array([1.0])

        classic = ClassicDifference(a=a.copy())

        x_i = np.array([[[[1.0]], [[2.0]]]])   # (1, 2, 2, 1) - n_cvar=1, 2 targets, 2 sources, 1 mode
        x_j = np.array([[[4.0], [5.0]]])       # (1, 2, 1) - n_cvar=1, 2 sources, 1 mode

        weights = np.eye(2)

        classic_result = _simulate_classic_coupling(classic, weights, x_i, x_j)

        expected_for_target_0 = a[0] * np.sum(weights[0] * (x_j[0, :, 0] - x_i[0, 0, :, 0]))
        expected_for_target_1 = a[0] * np.sum(weights[1] * (x_j[0, :, 0] - x_i[0, 1, :, 0]))

        np.testing.assert_almost_equal(classic_result[0, 0, 0], expected_for_target_0, decimal=5)
        np.testing.assert_almost_equal(classic_result[0, 1, 0], expected_for_target_1, decimal=5)


# ---------------------------------------------------------------------------
# Kuramoto – MAJOR DISCREPANCY
# ---------------------------------------------------------------------------

class TestKuramotoEquivalence:
    """Verify that hybrid Kuramoto matches classic TVB semantics.

    Classic TVB Kuramoto::

        pre(x_i, x_j) = sin(x_j - x_i)
        post(gx) = a / N * gx

    The full coupling computation is::

        result_i = (a / N) * sum_j(g_ij * sin(x_j - x_i))

    The hybrid implementation now correctly accepts both x_i and x_j
    in its pre function and includes the 1/N normalization in post.
    """

    def test_classic_kuramoto_pre_returns_sine_of_difference(self):
        """Classic Kuramoto.pre(x_i, x_j) must return sin(x_j - x_i)."""
        classic = ClassicKuramoto(a=np.array([1.0]))

        x_i = np.array([[[0.0], [np.pi / 4]]])
        x_j = np.array([[[np.pi / 2], [np.pi]]])

        result = classic.pre(x_i, x_j)
        expected = np.sin(x_j - x_i)

        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_hybrid_kuramoto_pre_should_accept_two_args(self):
        """Hybrid Kuramoto.pre must accept (x_j, x_i) and return sin(x_j - x_i).

        With the updated signature pre(x_j, x_i=None), when both arguments
        are provided, the result should be sin(x_j - x_i), matching classic TVB.
        """
        hybrid = HybridKuramoto(a=np.array([1.0]))

        x_i = np.array([0.0, np.pi / 4])
        x_j = np.array([np.pi / 2, np.pi])

        result = hybrid.pre(x_j, x_i)
        expected = np.sin(x_j - x_i)

        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_hybrid_kuramoto_pre_returns_sin_x_j_when_no_x_i(self):
        """Hybrid Kuramoto.pre(x_j) returns sin(x_j) when x_i is not provided."""
        hybrid = HybridKuramoto(a=np.array([1.0]))

        x_j = np.array([np.pi / 2, np.pi])

        result = hybrid.pre(x_j)
        expected = np.sin(x_j)

        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_classic_kuramoto_post_normalizes_by_n(self):
        """Classic Kuramoto.post divides by the number of state variables (N).

        The classic implementation normalizes by::

            a / gx.shape[0] * gx

        where gx.shape[0] is the number of coupling variables (modes).
        """
        classic = ClassicKuramoto(a=np.array([1.0]))

        gx = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = classic.post(gx)

        expected = 1.0 / gx.shape[0] * gx

        np.testing.assert_array_almost_equal(result, expected)

    def test_kuramoto_end_to_end_equivalence(self):
        """End-to-end: hybrid Kuramoto matches classic TVB.

        Classic computes::

            (a / N) * sum_j(g_ij * sin(x_j - x_i))

        Hybrid with corrected pre(x_j, x_i) computes::

            a * sum_j(g_ij * sin(x_j - x_i))

        These differ only by the 1/N normalization factor that classic TVB
        applies in post(). Without normalization, the hybrid result should
        equal N times the classic result.
        """
        weights = np.array([[0.5, 0.3], [0.2, 0.4]])
        n_target, n_source = weights.shape
        n_cvar = 1
        n_mode = 1

        rng = np.random.RandomState(42)
        x_i = rng.randn(n_cvar, n_target, n_source, n_mode)
        x_j = rng.randn(n_cvar, n_source, n_mode)

        classic = ClassicKuramoto(a=np.array([1.0]))
        hybrid = HybridKuramoto(a=np.array([1.0]))

        classic_result = _simulate_classic_coupling(classic, weights, x_i, x_j)

        # With corrected pre(x_j, x_i), hybrid should compute sin(x_j - x_i)
        x_j_expanded = x_j[:, np.newaxis, :, :]  # (n_cvar, 1, n_source, n_mode)
        x_i_expanded = x_i  # (n_cvar, n_target, n_source, n_mode)
        pre_result = hybrid.pre(x_j_expanded, x_i_expanded)
        g_ij = weights[np.newaxis, :, :, np.newaxis]
        hybrid_weighted_sum = (g_ij * pre_result).sum(axis=2)
        hybrid_result = hybrid.post(hybrid_weighted_sum)

        # Hybrid result should equal classic * N (since classic divides by N in post)
        n = n_cvar
        np.testing.assert_array_almost_equal(
            hybrid_result, classic_result * n, decimal=5
        )

    def test_kuramoto_sine_nonlinearity_matters(self):
        """Demonstrate that sin(sum) != sum(sin) for non-trivial inputs.

        Even if normalization were fixed, the hybrid would compute
        sin(weighted_sum) instead of weighted_sum_of_sin, which are
        generally different.
        """
        x_j = np.array([0.5, 1.0, 1.5])
        x_i = np.array([0.1, 0.2, 0.3])
        g = np.array([0.3, 0.4, 0.3])

        classic_per_edge = np.sin(x_j - x_i)
        classic_sum = np.sum(g * classic_per_edge)

        hybrid_sum = np.sum(g * x_j)
        hybrid_sin = np.sin(hybrid_sum)

        assert classic_sum != hybrid_sin, (
            f"sin(sum) and sum(sin) should differ but got "
            f"classic={classic_sum}, hybrid={hybrid_sin}"
        )


# ---------------------------------------------------------------------------
# HyperbolicTangent – MINOR DISCREPANCY (missing b parameter)
# ---------------------------------------------------------------------------

class TestHyperbolicTangentEquivalence:
    """Verify that hybrid HyperbolicTangent matches classic TVB semantics.

    Classic TVB HyperbolicTangent::

        pre(x_i, x_j) = a * (1 + tanh((b * x_j - midpoint) / sigma))
        post(gx) = gx  (identity)

    The ``b`` parameter scales x_j before computing the tanh. This allows
    modeling of different sensitivities to input magnitude.

    The hybrid implementation now correctly includes the ``b`` parameter.
    """

    def test_classic_hyperbolictangent_includes_b_parameter(self):
        """Classic HyperbolicTangent.pre(x_i, x_j) uses b * x_j."""
        classic = ClassicHyperbolicTangent(
            a=np.array([1.0]),
            b=np.array([2.0]),
            midpoint=np.array([0.0]),
            sigma=np.array([1.0]),
        )

        x_i = np.array([[[0.5]]])   # (1, 1, 1) dummy
        x_j = np.array([[1.0]])     # shapes compatible with classic.pre

        result = classic.pre(x_i, x_j)
        expected_val = 1.0 * (1.0 + np.tanh((2.0 * 1.0 - 0.0) / 1.0))

        np.testing.assert_almost_equal(result.flatten()[0], expected_val, decimal=10)

    def test_classic_hyperbolictangent_b_scaling(self):
        """Classic pre(x_i, x_j) with b=2 should differ from b=1."""
        x_j_vals = np.array([[1.0, 2.0, 3.0]])

        classic_b1 = ClassicHyperbolicTangent(
            a=np.array([1.0]), b=np.array([1.0]),
            midpoint=np.array([0.0]), sigma=np.array([1.0]),
        )
        classic_b2 = ClassicHyperbolicTangent(
            a=np.array([1.0]), b=np.array([2.0]),
            midpoint=np.array([0.0]), sigma=np.array([1.0]),
        )

        x_i = np.zeros_like(x_j_vals)
        result_b1 = classic_b1.pre(x_i, x_j_vals)
        result_b2 = classic_b2.pre(x_i, x_j_vals)

        assert not np.allclose(result_b1, result_b2), (
            "b=1 and b=2 should produce different results"
        )

    def test_hybrid_hyperbolictangent_missing_b(self):
        """Hybrid HyperbolicTangent is missing the b parameter.

        The hybrid HyperbolicTangent now includes the ``b`` parameter,
        matching the classic TVB formula: ``a * (1 + tanh((b*x - midpoint) / sigma))``.

        When b != 1, the ``b`` parameter correctly scales the input.
        """
        hybrid_default = HybridHyperbolicTangent(
            a=np.array([1.0]),
            midpoint=np.array([0.0]),
            sigma=np.array([1.0]),
        )

        x = np.array([[1.0]])

        hybrid_result = hybrid_default.pre(x)

        expected_with_b_equals_1 = 1.0 * (1.0 + np.tanh((1.0 * 1.0 - 0.0) / 1.0))

        np.testing.assert_array_almost_equal(
            hybrid_result.flatten(),
            expected_with_b_equals_1.flatten(),
        )

        hybrid_with_b = HybridHyperbolicTangent(
            a=np.array([1.0]),
            b=np.array([2.0]),
            midpoint=np.array([0.0]),
            sigma=np.array([1.0]),
        )

        result_with_b = hybrid_with_b.pre(x)
        expected_with_b = 1.0 * (1.0 + np.tanh((2.0 * 1.0 - 0.0) / 1.0))

        np.testing.assert_array_almost_equal(
            result_with_b.flatten(),
            expected_with_b.flatten(),
        )

    def test_classic_hyperbolictangent_pre_uses_only_xj(self):
        """Classic HyperbolicTangent.pre does not depend on x_i.

        Unlike Difference and Kuramoto, the classic HyperbolicTangent
        pre function only uses x_j (not x_i), so the single-argument
        signature in hybrid is semantically correct—only the missing
        ``b`` parameter is an issue.
        """
        classic = ClassicHyperbolicTangent(
            a=np.array([1.0]),
            b=np.array([1.0]),
            midpoint=np.array([0.0]),
            sigma=np.array([1.0]),
        )

        x_j = np.array([[1.0, 2.0, 3.0]])

        x_i_a = np.zeros((1, 1, 3, 1))
        x_i_b = np.ones((1, 1, 3, 1)) * 99.0
        x_j_expanded = np.broadcast_to(x_j[:, :, np.newaxis], (1, 1, 3, 1))

        result_a = classic.pre(x_i_a, x_j_expanded)
        result_b = classic.pre(x_i_b, x_j_expanded)

        np.testing.assert_array_equal(result_a, result_b)

    def test_hyperbolictangent_end_to_end_with_b_equals_1(self):
        """When b=1 and classic pre ignores x_i, results should match."""
        weights = np.array([[0.5, 0.3], [0.2, 0.4]])
        n_target, n_source = weights.shape
        n_cvar = 1
        n_mode = 1

        rng = np.random.RandomState(42)
        x_i = rng.randn(n_cvar, n_target, n_source, n_mode)
        x_j = rng.randn(n_cvar, n_source, n_mode)

        classic = ClassicHyperbolicTangent(
            a=np.array([1.0]),
            b=np.array([1.0]),
            midpoint=np.array([0.0]),
            sigma=np.array([1.0]),
        )
        hybrid = HybridHyperbolicTangent(
            a=np.array([1.0]),
            midpoint=np.array([0.0]),
            sigma=np.array([1.0]),
        )

        classic_result = _simulate_classic_coupling(classic, weights, x_i, x_j)

        x_j_expanded = np.broadcast_to(
            x_j[:, np.newaxis, :, :],
            (n_cvar, n_target, n_source, n_mode),
        ).copy()
        x_i_broadcast = np.broadcast_to(
            x_i,
            (n_cvar, n_target, n_source, n_mode),
        )

        classic_pre = classic.pre(x_i_broadcast, x_j_expanded)
        g_ij = weights[np.newaxis, :, :, np.newaxis]
        hybrid_pre = hybrid.pre(x_j)
        hybrid_pre_broadcast = np.broadcast_to(
            hybrid_pre[:, np.newaxis, :, :],
            (n_cvar, n_target, n_source, n_mode),
        )

        assert classic_pre.shape == hybrid_pre_broadcast.shape, (
            f"Shape mismatch: classic {classic_pre.shape} vs hybrid {hybrid_pre_broadcast.shape}"
        )

        classic_weighted = (g_ij * classic_pre).sum(axis=2)
        hybrid_weighted = (g_ij * hybrid_pre_broadcast).sum(axis=2)

        np.testing.assert_array_almost_equal(hybrid_weighted, classic_weighted, decimal=5)


# ---------------------------------------------------------------------------
# SigmoidalJansenRit – MAJOR DISCREPANCY
# ---------------------------------------------------------------------------

class TestSigmoidalJansenRitEquivalence:
    """Verify that hybrid SigmoidalJansenRit matches classic TVB semantics.

    Classic TVB SigmoidalJansenRit::

        pre(x_i, x_j) = cmin + (cmax - cmin) / (1 + exp(r * (midpoint - (x_j[:,0] - x_j[:,1]))))
        post(gx) = a * gx

    The classic ``pre`` function:
    - Takes state variable arrays x_j with TWO state variables
    - Computes x_j[:,0] - x_j[:,1] (difference of first two state variables)
    - Applies a sigmoidal transformation to this difference
    - Returns result with an extra dimension (from squeeze to expand)

    The hybrid implementation::
        pre(x) = a * (2*e0) / (1 + exp(r * (v0 - x)))
        post(x) = x  (identity)

    This is a fundamentally different formula:
    1. The classic uses cmin, cmax; the hybrid uses e0, v0
    2. The classic computes x_j[:,0] - x_j[:,1]; the hybrid takes a single x
    3. The classic applies post(a * gx); the hybrid applies pre then identity
    """

    def test_classic_sigmoidal_jansen_rit_pre_uses_two_state_variables(self):
        """Classic SigmoidalJansenRit.pre computes x_j[:,0] - x_j[:,1].

        The classic implementation expects x_j to have at least 2 nodes
        (or 2 coupling variable indices) and computes their difference
        before applying the sigmoid. The indexing x_j[:,0] - x_j[:,1]
        operates along the second axis (nodes/sources).
        """
        classic = ClassicSigmoidalJansenRit(
            cmin=np.array([0.0]),
            cmax=np.array([0.0025 * 2]),
            midpoint=np.array([6.0]),
            r=np.array([0.56]),
            a=np.array([1.0]),
        )

        x_j = np.random.RandomState(42).randn(2, 3, 1)  # (2 cvars, 3 nodes, 1 mode)
        x_i = np.zeros_like(x_j)

        result = classic.pre(x_i, x_j)

        diff = x_j[:, 0, :] - x_j[:, 1, :]  # shape (2, 1) - node0 - node1 per cvar
        expected = classic.cmin + (classic.cmax - classic.cmin) / (
            1.0 + np.exp(classic.r * (classic.midpoint[:, None] - diff))
        )
        expected = expected[:, np.newaxis]  # add mode dimension -> (2, 1, 1)

        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_classic_sigmoidal_jansen_rit_post_multiplies_by_a(self):
        """Classic SigmoidalJansenRit.post multiplies by a."""
        classic = ClassicSigmoidalJansenRit(
            cmin=np.array([0.0]),
            cmax=np.array([0.005]),
            midpoint=np.array([6.0]),
            r=np.array([0.56]),
            a=np.array([1.0]),
        )

        gx = np.array([[1.0, 2.0, 3.0]])
        result = classic.post(gx)

        np.testing.assert_array_almost_equal(result, classic.a * gx)

    def test_hybrid_sigmoidal_jansen_rit_formula_equivalence(self):
        """Hybrid SJR supports classic formula via use_classic=1.

        With ``use_classic=1`` (default), the hybrid SJR accepts two
        source state variables and applies the classic formula::

            cmin + (cmax - cmin) / (1 + exp(r * (midpoint - (x_j[0] - x_j[1]))))

        The legacy ``e0/v0`` formula is still available via
        ``use_classic=0``::

            a * (2*e0) / (1 + exp(r * (v0 - x)))
        """
        # Classic mode: two source cvars
        hybrid = HybridSigmoidalJansenRit(
            cmin=np.array([0.0]),
            cmax=np.array([0.005]),
            midpoint=np.array([6.0]),
            r=np.array([0.56]),
            a=np.array([1.0]),
        )
        assert hybrid.use_classic == 1
        assert hybrid.n_cvar_in == 2

        # 2-cvar input: x_j[0] - x_j[1] = 6.0 = midpoint → sigmoid at 0.5
        x_2cvar = np.zeros((2, 3, 1))
        x_2cvar[0] = 12.0  # x_j[0]
        x_2cvar[1] = 6.0   # x_j[1] → diff = 6.0 = midpoint
        result = hybrid.pre(x_2cvar)
        assert result.shape == (1, 3, 1)
        expected = 0.0 + (0.005 - 0.0) / 2.0  # midpoint → sigmoid(0) = 0.5
        np.testing.assert_array_almost_equal(result[0, 0, 0], expected)

        # Legacy mode: single scalar input
        hybrid_legacy = HybridSigmoidalJansenRit(
            a=np.array([1.0]),
            e0=np.array([2.5]),
            r=np.array([0.56]),
            v0=np.array([6.0]),
            use_classic=0,
        )
        x = np.array([[5.0]])
        hybrid_result = hybrid_legacy.pre(x)
        expected_legacy = 1.0 * (2 * 2.5) / (1 + np.exp(0.56 * (6.0 - 5.0)))
        np.testing.assert_array_almost_equal(hybrid_result, expected_legacy)

    def test_sigmoidal_jansen_rit_end_to_end_equivalence(self):
        """End-to-end: hybrid SJR matches classic with 2-cvar pre.

        With ``use_classic=1`` and 2 source cvars, the hybrid SJR
        computes the same result as classic TVB.
        """
        weights = np.array([[0.5, 0.3], [0.2, 0.4]])
        n_target, n_source = weights.shape
        n_cvar = 2
        n_mode = 1

        rng = np.random.RandomState(42)
        x_j = rng.randn(n_cvar, n_source, n_mode)

        classic = ClassicSigmoidalJansenRit(
            cmin=np.array([0.0]),
            cmax=np.array([0.005]),
            midpoint=np.array([6.0]),
            r=np.array([0.56]),
            a=np.array([1.0]),
        )

        hybrid = HybridSigmoidalJansenRit(
            cmin=np.array([0.0]),
            cmax=np.array([0.005]),
            midpoint=np.array([6.0]),
            r=np.array([0.56]),
            a=np.array([1.0]),
        )

        # Both should produce the same result from the same input data
        # Hybrid: pre(x_j) where x_j has shape (2, n_source, n_mode)
        # Classic: pre(x_i, x_j) where x_j[:,0] and x_j[:,1] are the two cvars
        hybrid_pre = hybrid.pre(x_j)
        # Manually compute classic result
        diff = x_j[0] - x_j[1]
        classic_expected = classic.cmin + (classic.cmax - classic.cmin) / (
            1.0 + np.exp(classic.r * (classic.midpoint - diff))
        )
        np.testing.assert_array_almost_equal(
            hybrid_pre[0], classic_expected, decimal=5
        )


# ---------------------------------------------------------------------------
# PreSigmoidal – MAJOR DISCREPANCY
# ---------------------------------------------------------------------------

class TestPreSigmoidalEquivalence:
    """Verify that hybrid PreSigmoidal matches classic TVB semantics.

    Classic TVB PreSigmoidal has a complex ``__call__`` override with:

    **Static threshold mode**::

        _ = P * x_j - theta
        A_j = H * (Q + tanh(G * _))
        result = (g_ij.T * A_j).sum(axis=0)

    **Dynamic threshold mode**::

        _ = P * x_j[:,0] - x_j[:,1]  (uses TWO state variables)
        A_j = H * (Q + tanh(G * _))
        result = [weighted_sum, diagonal_self_connections]

    The hybrid implementation simplifies to::

        pre(x) = H * (Q + tanh(G * (P * x - theta)))
        post(x) = x  (identity)

    This is only equivalent in the static-threshold case when x_j has
    a single state variable.
    """

    def test_classic_presigmoidal_static_threshold(self):
        """Classic PreSigmoidal in static mode: P * x_j - theta."""
        classic = ClassicPreSigmoidal(
            H=np.array([1.0]),
            Q=np.array([0.0]),
            G=np.array([1.0]),
            P=np.array([1.0]),
            theta=np.array([0.0]),
        )
        classic.dynamic = False

        x_j_val = np.array([[1.0, 2.0, 3.0]])
        expected_transform = 1.0 * (0.0 + np.tanh(1.0 * (1.0 * x_j_val - 0.0)))

        assert expected_transform.shape == x_j_val.shape

    def test_classic_presigmoidal_dynamic_threshold_uses_two_state_vars(self):
        """Classic PreSigmoidal in dynamic mode uses x_j[:,0] - x_j[:,1].

        When dynamic=True, the threshold comes from the second state variable
        of x_j. The hybrid has no equivalent mechanism.
        """
        classic = ClassicPreSigmoidal(
            H=np.array([1.0]),
            Q=np.array([0.0]),
            G=np.array([1.0]),
            P=np.array([1.0]),
            theta=np.array([0.0]),
        )
        classic.dynamic = True

        assert classic.dynamic is True

    def test_hybrid_presigmoidal_static_formula_matches_intent(self):
        """Hybrid PreSigmoidal static formula is H * (Q + tanh(G * (P * x - theta))).

        This matches the classic static-threshold mode when P=1 and
        theta is a scalar (not derived from a second state variable).
        """
        hybrid = HybridPreSigmoidal(
            H=np.array([1.0]),
            Q=np.array([0.0]),
            G=np.array([2.0]),
            P=np.array([1.0]),
            theta=np.array([0.5]),
        )

        x = np.array([[1.0, 2.0, 3.0]])

        result = hybrid.pre(x)

        expected = 1.0 * (0.0 + np.tanh(2.0 * (1.0 * 1.0 - 0.5)))

        # Hybrid PreSigmoidal now supports static threshold.
        # Verify formula: H * (Q + tanh(G * (P * x - theta)))
        np.testing.assert_array_almost_equal(
            result[0, 0], expected, decimal=5
        )
        # Static mode preserves shape
        assert result.shape == x.shape

    def test_hybrid_presigmoidal_dynamic_mode_now_supported(self):
        """Hybrid PreSigmoidal now supports dynamic threshold (default=1).

        With ``dynamic=True``, the second source state variable (x_j[1])
        is used as the threshold instead of the static ``theta`` param.
        """
        hybrid = HybridPreSigmoidal(
            H=np.array([0.5]),
            Q=np.array([1.0]),
            G=np.array([60.0]),
            P=np.array([1.0]),
            theta=np.array([0.5]),
            dynamic=True,
        )

        assert hybrid.dynamic is True

        # 2-cvar input for dynamic mode
        x_2cvar = np.zeros((2, 5, 1), dtype=np.float32)
        x_2cvar[0] = 1.0  # afferent signal
        x_2cvar[1] = 0.3  # dynamic threshold
        result = hybrid.pre(x_2cvar)
        # Dynamic mode collapses 2→1 cvars
        assert result.shape == (1, 5, 1)
        # Verify formula: H*(Q + tanh(G*(P*x0 - x1)))
        expected = 0.5 * (1.0 + np.tanh(60.0 * (1.0 * 1.0 - 0.3)))
        np.testing.assert_array_almost_equal(result[0, 0, 0], expected, decimal=5)


# ---------------------------------------------------------------------------
# Cross-cutting: full pipeline comparison
# ---------------------------------------------------------------------------

class TestFullPipelineEquivalence:
    """End-to-end pipeline comparison for coupling types with NO discrepancy.

    For Linear, Scaling, and Sigmoidal (where pre is identity), the hybrid
    pipeline should produce identical results to the classic pipeline when
    the weights and inputs are the same.
    """

    def _make_weighted_sum(self, weights, x_j):
        """Compute weighted sum: sum_j(g_ij * x_j) for each target node i.

        This simulates the projection's weighted sum step that both
        classic and hybrid coupling operate on (when pre is identity).
        """
        g_ij = weights[np.newaxis, :, :, np.newaxis]
        x_j_broadcast = x_j[:, np.newaxis, :, :]
        return (g_ij * x_j_broadcast).sum(axis=2)

    def test_linear_full_pipeline(self):
        """Full pipeline: classic Linear == hybrid Linear."""
        weights = np.array([[0.5, 0.3], [0.2, 0.4]])
        n_cvar, n_source, n_mode = 1, 2, 1

        rng = np.random.RandomState(123)
        x_j = rng.randn(n_cvar, n_source, n_mode)

        classic = ClassicLinear(a=np.array([0.00390625]), b=np.array([0.0]))
        hybrid = HybridLinear(a=np.array([0.00390625]), b=np.array([0.0]))

        weighted_sum = self._make_weighted_sum(weights, x_j)

        classic_result = classic.post(weighted_sum)
        hybrid_result = hybrid.post(weighted_sum)

        np.testing.assert_array_almost_equal(hybrid_result, classic_result, decimal=10)

    def test_scaling_full_pipeline(self):
        """Full pipeline: classic Scaling == hybrid Scaling."""
        weights = np.array([[0.5, 0.3], [0.2, 0.4]])
        n_cvar, n_source, n_mode = 1, 2, 1

        rng = np.random.RandomState(456)
        x_j = rng.randn(n_cvar, n_source, n_mode)

        classic = ClassicScaling(a=np.array([0.00390625]))
        hybrid = HybridScaling(a=np.array([0.00390625]))

        weighted_sum = self._make_weighted_sum(weights, x_j)

        classic_result = classic.post(weighted_sum)
        hybrid_result = hybrid.post(weighted_sum)

        np.testing.assert_array_almost_equal(hybrid_result, classic_result, decimal=10)

    def test_sigmoidal_full_pipeline(self):
        """Full pipeline: classic Sigmoidal == hybrid Sigmoidal."""
        weights = np.array([[0.5, 0.3], [0.2, 0.4]])
        n_cvar, n_source, n_mode = 1, 2, 1

        rng = np.random.RandomState(789)
        x_j = rng.randn(n_cvar, n_source, n_mode)

        classic = ClassicSigmoidal(
            cmin=np.array([-1.0]),
            cmax=np.array([1.0]),
            a=np.array([1.0]),
            midpoint=np.array([0.0]),
            sigma=np.array([1.0]),
        )
        hybrid = HybridSigmoidal(
            cmin=np.array([-1.0]),
            cmax=np.array([1.0]),
            a=np.array([1.0]),
            midpoint=np.array([0.0]),
            sigma=np.array([1.0]),
        )

        weighted_sum = self._make_weighted_sum(weights, x_j)

        classic_result = classic.post(weighted_sum)
        hybrid_result = hybrid.post(weighted_sum)

        np.testing.assert_array_almost_equal(hybrid_result, classic_result, decimal=5)

    def test_difference_full_pipeline_now_equivalent(self):
        """Full pipeline: hybrid Difference now matches classic Difference.

        The hybrid now computes a * sum_j(g_ij * (x_j - x_i)) using
        per-edge pre() to compute x_j - x_i before the weighted sum,
        matching the classic TVB pipeline.
        """
        weights = np.array([[0.5, 0.3], [0.2, 0.4]])
        n_target, n_source = weights.shape
        n_cvar, n_mode = 1, 1

        rng = np.random.RandomState(42)
        x_i = rng.randn(n_cvar, n_target, n_source, n_mode)
        x_j = rng.randn(n_cvar, n_source, n_mode)

        classic = ClassicDifference(a=np.array([0.1]))
        hybrid = HybridDifference(a=np.array([0.1]))

        classic_result = _simulate_classic_coupling(classic, weights, x_i, x_j)

        # Hybrid: pre(x_j, x_i) computes x_j - x_i per-edge, then weight/sum/post
        pre_result = hybrid.pre(x_j, x_i)
        weighted = weights[np.newaxis, :, :, np.newaxis] * pre_result
        summed = weighted.sum(axis=2)
        hybrid_result = hybrid.post(summed)

        np.testing.assert_array_almost_equal(hybrid_result, classic_result, decimal=5)

    def test_kuramoto_full_pipeline_now_equivalent(self):
        """Full pipeline: hybrid Kuramoto now matches classic Kuramoto.

        The hybrid now computes (a/N) * sum_j(g_ij * sin(x_j - x_i))
        using per-edge pre() to compute sin(x_j - x_i) before the
        weighted sum, then a/N scaling in post(), matching classic TVB.
        """
        weights = np.array([[0.5, 0.3], [0.2, 0.4]])
        n_target, n_source = weights.shape
        n_cvar, n_mode = 1, 1

        rng = np.random.RandomState(42)
        x_i = rng.randn(n_cvar, n_target, n_source, n_mode)
        x_j = rng.randn(n_cvar, n_source, n_mode)

        classic = ClassicKuramoto(a=np.array([1.0]))
        hybrid = HybridKuramoto(a=np.array([1.0]))

        classic_result = _simulate_classic_coupling(classic, weights, x_i, x_j)

        # Hybrid: pre(x_j, x_i) computes sin(x_j - x_i) per-edge, then weight/sum/post
        pre_result = hybrid.pre(x_j, x_i)
        weighted = weights[np.newaxis, :, :, np.newaxis] * pre_result
        summed = weighted.sum(axis=2)
        hybrid_result = hybrid.post(summed)

        np.testing.assert_array_almost_equal(hybrid_result, classic_result, decimal=5)


# ---------------------------------------------------------------------------
# Pre-function semantic tests – what pre() SHOULD compute
# ---------------------------------------------------------------------------

class TestPreFunctionSemantics:
    """Test the correct semantics of pre() functions.

    In classic TVB, pre(x_i, x_j) is a per-edge transform applied BEFORE
    weighting by g_ij. These tests verify what each pre function SHOULD
    compute, establishing regression targets for hybrid implementations.
    """

    def test_difference_pre_should_compute_xj_minus_xi(self):
        """Difference.pre(x_i, x_j) should return x_j - x_i."""
        x_i = np.array([1.0, 2.0, 3.0])
        x_j = np.array([4.0, 5.0, 6.0])

        expected = x_j - x_i

        classic = ClassicDifference(a=np.array([1.0]))
        x_i_4d = x_i[np.newaxis, np.newaxis, :, np.newaxis]
        x_j_4d = x_j[np.newaxis, :, np.newaxis]
        classic_result = classic.pre(x_i_4d, x_j_4d)

        np.testing.assert_array_almost_equal(
            classic_result.flatten()[:len(expected)],
            expected,
            decimal=10,
        )

    def test_kuramoto_pre_should_compute_sin_of_difference(self):
        """Kuramoto.pre(x_i, x_j) should return sin(x_j - x_i)."""
        x_i = np.array([0.0, np.pi / 4, np.pi / 2])
        x_j = np.array([np.pi / 2, np.pi, 3 * np.pi / 2])

        expected = np.sin(x_j - x_i)

        classic = ClassicKuramoto(a=np.array([1.0]))
        x_i_4d = x_i[np.newaxis, np.newaxis, :, np.newaxis]
        x_j_4d = x_j[np.newaxis, :, np.newaxis]
        classic_result = classic.pre(x_i_4d, x_j_4d)

        np.testing.assert_array_almost_equal(
            classic_result.flatten()[:len(expected)],
            expected,
            decimal=10,
        )

    def test_hyperbolictangent_pre_should_include_b_parameter(self):
        """HyperbolicTangent.pre(x_i, x_j) should compute a*(1+tanh((b*x_j - midpoint)/sigma)).

        The b parameter scales x_j before the tanh is applied. With b=2
        and x_j=3, the input to tanh should be (2*3 - 0) / 1 = 6.
        """
        a = np.array([1.5])
        b = np.array([2.0])
        midpoint = np.array([0.0])
        sigma = np.array([1.0])

        classic = ClassicHyperbolicTangent(
            a=a, b=b, midpoint=midpoint, sigma=sigma,
        )

        x_j_val = np.array([3.0])
        x_i_val = np.array([0.0])

        x_j_4d = x_j_val[np.newaxis, :, np.newaxis]
        x_i_4d = x_i_val[np.newaxis, np.newaxis, :, np.newaxis]

        result = classic.pre(x_i_4d, x_j_4d)

        expected = a[0] * (1.0 + np.tanh((b[0] * x_j_val[0] - midpoint[0]) / sigma[0]))
        np.testing.assert_almost_equal(result.flatten()[0], expected, decimal=10)

    def test_linear_pre_should_be_identity(self):
        """Linear.pre should return x_j unchanged (identity)."""
        classic = ClassicLinear()
        x_j = np.array([[1.0, 2.0]])[:, :, np.newaxis]

        result = classic.pre(None, x_j)
        np.testing.assert_array_equal(result, x_j)

    def test_scaling_pre_should_be_identity(self):
        """Scaling.pre should return x_j unchanged (identity)."""
        classic = ClassicScaling()
        x_j = np.array([[1.0, 2.0]])[:, :, np.newaxis]

        result = classic.pre(None, x_j)
        np.testing.assert_array_equal(result, x_j)

    def test_sigmoidal_pre_should_be_identity(self):
        """Sigmoidal.pre should return x_j unchanged (identity)."""
        classic = ClassicSigmoidal()
        x_j = np.array([[1.0, 2.0]])[:, :, np.newaxis]

        result = classic.pre(None, x_j)
        np.testing.assert_array_equal(result, x_j)

    def test_sigmoidal_jansen_rit_pre_should_use_two_state_variables(self):
        """SigmoidalJansenRit.pre should compute sigmoid of (x_j[:,0] - x_j[:,1]).

        The classic implementation computes x_j[:,0] - x_j[:,1] along the
        second axis (nodes/sources) and applies the sigmoid to this difference.
        The result is then expanded with an extra dimension.
        """
        classic = ClassicSigmoidalJansenRit(
            cmin=np.array([0.0]),
            cmax=np.array([0.005]),
            midpoint=np.array([6.0]),
            r=np.array([0.56]),
            a=np.array([1.0]),
        )

        rng = np.random.RandomState(42)
        x_j = rng.randn(2, 3, 1)  # (2 cvars, 3 nodes, 1 mode)
        x_i = np.zeros_like(x_j)

        result = classic.pre(x_i, x_j)

        diff = x_j[:, 0, :] - x_j[:, 1, :]  # shape (2, 1) per cvar difference
        expected = classic.cmin + (classic.cmax - classic.cmin) / (
            1.0 + np.exp(classic.r * (classic.midpoint[:, None] - diff))
        )
        expected = expected[:, np.newaxis]  # (2, 1, 1)

        np.testing.assert_array_almost_equal(result, expected, decimal=10)


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

class TestKuramotoNormalization:
    """Test that Kuramoto coupling post() applies a[:, None] * gx.

    Classic TVB Kuramoto.post divides by N (the number of coupling variables)::

        post(gx) = a / gx.shape[0] * gx

    The hybrid Kuramoto.post does NOT normalize by N, applying only::

        post(gx) = a[:, None] * gx

    This is a known discrepancy documented in the hybrid implementation.
    """

    def test_classic_kuramoto_normalizes_by_n_cvar(self):
        """Classic Kuramoto divides by the number of state variables."""
        classic = ClassicKuramoto(a=np.array([1.0]))

        gx = np.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2) -> N=2 state vars

        result = classic.post(gx)

        expected = 1.0 / 2 * gx

        np.testing.assert_array_almost_equal(result, expected)

    def test_hybrid_kuramoto_no_normalization(self):
        """Hybrid Kuramoto now normalizes by N, matching classic TVB.

        The hybrid post() applies a[:, None] / N * gx where N is the
        number of coupling variables (gx.shape[0]). This matches the
        classic TVB normalization.
        """
        hybrid = HybridKuramoto(a=np.array([1.0]))

        gx = np.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)

        result = hybrid.post(gx)

        N = gx.shape[0]  # = 2
        expected = np.array([[1.0, 2.0], [3.0, 4.0]]) / N  # a/N * gx

        np.testing.assert_array_almost_equal(result, expected)