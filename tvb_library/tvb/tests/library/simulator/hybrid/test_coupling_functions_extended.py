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
Tests for extended coupling functions in the hybrid simulator.

Covers :class:`~tvb.simulator.hybrid.coupling.Kuramoto` (sinusoidal
phase-difference coupling: ``(a/N)*sin(x)``),
:class:`~tvb.simulator.hybrid.coupling.Difference` (diffusive: ``a*x``),
:class:`~tvb.simulator.hybrid.coupling.HyperbolicTangent`
(``a*(1+tanh((x-midpoint)/sigma))``),
:class:`~tvb.simulator.hybrid.coupling.SigmoidalJansenRit`
(Jansen–Rit sigmoid: ``a*(2*e0)/(1+exp(r*(v0-x)))``), and
:class:`~tvb.simulator.hybrid.coupling.PreSigmoidal`
(``H*(Q+tanh(G*(P*x-theta)))``).
"""

import numpy as np
import pytest
from scipy import sparse as sp

from tvb.simulator.hybrid.coupling import (
    Kuramoto,
    Difference,
    HyperbolicTangent,
    SigmoidalJansenRit,
    PreSigmoidal,
)
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.datatypes.connectivity import Connectivity


class TestKuramotoCoupling:
    """Tests for :class:`~tvb.simulator.hybrid.coupling.Kuramoto` (sinusoidal phase coupling: ``(a/N)*sin(x)``).

    Verifies sine-function behaviour, amplitude scaling, per-mode
    normalisation (``1/N`` factor), 2π periodicity, and that ``pre()`` is
    the identity.
    """

    def test_kuramoto_default_parameters(self):
        """Test Kuramoto with default parameters."""
        kuramoto = Kuramoto()
        x = np.array([[0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]])
        y = kuramoto.post(x, mode=0)

        expected = np.array([[0.0, 1.0, 0.0, -1.0, 0.0]])
        np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-8)

    def test_kuramoto_amplitude_scaling(self):
        """Test Kuramoto amplitude parameter."""
        a_values = [0.5, 1.0, 2.0]
        x = np.array([[np.pi / 2]])

        for a_val in a_values:
            kuramoto = Kuramoto(a=np.array([a_val]))
            y = kuramoto.post(x, mode=0)
            expected = np.array([[a_val * 1.0]])
            np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-8)

    def test_kuramoto_mode_normalization(self):
        """Test Kuramoto normalization with multiple modes."""
        kuramoto = Kuramoto(a=np.array([1.0]))
        x = np.array([[np.pi / 2]])

        y1 = kuramoto.post(x, mode=1)
        y2 = kuramoto.post(x, mode=2)
        y5 = kuramoto.post(x, mode=5)

        expected1 = np.array([[1.0]])
        expected2 = np.array([[0.5]])
        expected5 = np.array([[0.2]])

        np.testing.assert_allclose(y1, expected1, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(y2, expected2, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(y5, expected5, rtol=1e-6, atol=1e-8)

    def test_kuramoto_periodicity(self):
        """Test Kuramoto periodicity (2π period)."""
        kuramoto = Kuramoto(a=np.array([1.0]))
        x1 = np.array([[0.0]])
        x2 = np.array([[2 * np.pi]])
        x3 = np.array([[4 * np.pi]])

        y1 = kuramoto.post(x1, mode=0)
        y2 = kuramoto.post(x2, mode=0)
        y3 = kuramoto.post(x3, mode=0)

        np.testing.assert_allclose(y1, y2, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(y2, y3, rtol=1e-6, atol=1e-8)

    def test_kuramoto_multidimensional(self):
        """Test Kuramoto handles multidimensional arrays."""
        kuramoto = Kuramoto(a=np.array([0.5]))
        x = np.random.randn(2, 5, 3)
        y = kuramoto.post(x, mode=0)
        expected = 0.5 * np.sin(x)
        np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-8)

    def test_kuramoto_pre_identity(self):
        """Test Kuramoto.pre() returns input unchanged."""
        kuramoto = Kuramoto(a=np.array([2.0]))
        x = np.array([1.0, 2.0, 3.0])
        result = kuramoto.pre(x)
        np.testing.assert_array_equal(result, x)

    def test_kuramoto_formula(self):
        """Verify Kuramoto implements correct formula: (a/N)*sin(x)."""
        kuramoto = Kuramoto(a=np.array([2.0]))
        x = np.array([[0.0, np.pi / 2, np.pi]])
        result = kuramoto.post(x, mode=1)
        expected = 2.0 * np.sin(x)
        np.testing.assert_array_almost_equal(result, expected)


class TestDifferenceCoupling:
    """Tests for :class:`~tvb.simulator.hybrid.coupling.Difference` (diffusive coupling: ``a*x``).

    Verifies various coupling strengths including zero and negative values,
    and that ``pre()`` is the identity.
    """

    def test_difference_default_parameters(self):
        """Test Difference with default parameters (identity)."""
        diff = Difference()
        x = np.array([[1.0, 2.0, 3.0]])
        y = diff.post(x, mode=0)
        expected = x
        np.testing.assert_array_equal(y, expected)

    def test_difference_coupling_strength(self):
        """Test Difference with different coupling strengths."""
        a_values = [0.5, 1.0, 2.0]
        x = np.array([[1.0, 2.0, 3.0]])

        for a_val in a_values:
            diff = Difference(a=np.array([a_val]))
            y = diff.post(x, mode=0)
            expected = a_val * x
            np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-8)

    def test_difference_zero_coupling(self):
        """Test Difference with a=0 (should produce zeros)."""
        diff = Difference(a=np.array([0.0]))
        x = np.array([[1.0, 2.0, 3.0]])
        y = diff.post(x, mode=0)
        expected = np.zeros_like(x)
        np.testing.assert_array_equal(y, expected)

    def test_difference_negative_coupling(self):
        """Test Difference with negative coupling strength."""
        diff = Difference(a=np.array([-1.5]))
        x = np.array([[1.0, 2.0, 3.0]])
        y = diff.post(x, mode=0)
        expected = -1.5 * x
        np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-8)

    def test_difference_multidimensional(self):
        """Test Difference handles multidimensional arrays."""
        diff = Difference(a=np.array([2.0]))
        x = np.random.randn(3, 4, 5)
        y = diff.post(x, mode=0)
        expected = 2.0 * x
        np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-8)

    def test_difference_pre_identity(self):
        """Test Difference.pre() returns input unchanged."""
        diff = Difference(a=np.array([2.0]))
        x = np.array([1.0, 2.0, 3.0])
        result = diff.pre(x)
        np.testing.assert_array_equal(result, x)

    def test_difference_formula(self):
        """Verify Difference implements correct formula: a*x."""
        diff = Difference(a=np.array([3.0]))
        x = np.array([[1.0, 2.0, 3.0]])
        result = diff.post(x, mode=0)
        expected = 3.0 * x
        np.testing.assert_array_almost_equal(result, expected)


class TestHyperbolicTangentCoupling:
    """Tests for :class:`~tvb.simulator.hybrid.coupling.HyperbolicTangent` (``a*(1+tanh((x-m)/σ))``).

    Verifies saturation at large positive/negative inputs, custom midpoint,
    amplitude and width parameters, output range [0, 2a], and that ``post()``
    is the identity.
    """

    def test_hyperbolic_tangent_default_parameters(self):
        """Test HyperbolicTangent with default parameters."""
        tanh_cfun = HyperbolicTangent()
        x = np.array([[-10.0, 0.0, 10.0]])
        y = tanh_cfun.pre(x, mode=0)

        assert y[0, 0] < 0.01  # tanh(-inf) = -1, so a*(1-1) = 0
        assert abs(y[0, 1] - 1.0) < 0.01  # tanh(0) = 0, so a*(1+0) = 1
        assert y[0, 2] > 1.99  # tanh(+inf) = 1, so a*(1+1) = 2

    def test_hyperbolic_tangent_at_midpoint(self):
        """Test HyperbolicTangent at midpoint."""
        tanh_cfun = HyperbolicTangent(
            a=np.array([1.0]), midpoint=np.array([0.0]), sigma=np.array([1.0])
        )
        x = np.array([[0.0]])
        y = tanh_cfun.pre(x, mode=0)
        expected = np.array([[1.0]])  # tanh(0) = 0, so a*(1+0) = 1
        np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-8)

    def test_hyperbolic_tangent_large_positive(self):
        """Test HyperbolicTangent at large positive values."""
        tanh_cfun = HyperbolicTangent(
            a=np.array([1.0]), midpoint=np.array([0.0]), sigma=np.array([1.0])
        )
        x = np.array([[10.0]])
        y = tanh_cfun.pre(x, mode=0)
        expected = np.array([[2.0]])  # tanh(+inf) = 1, so a*(1+1) = 2
        np.testing.assert_allclose(y, expected, rtol=1e-2, atol=1e-2)

    def test_hyperbolic_tangent_large_negative(self):
        """Test HyperbolicTangent at large negative values."""
        tanh_cfun = HyperbolicTangent(
            a=np.array([1.0]), midpoint=np.array([0.0]), sigma=np.array([1.0])
        )
        x = np.array([[-10.0]])
        y = tanh_cfun.pre(x, mode=0)
        expected = np.array([[0.0]])  # tanh(-inf) = -1, so a*(1-1) = 0
        np.testing.assert_allclose(y, expected, rtol=1e-2, atol=1e-2)

    def test_hyperbolic_tangent_custom_midpoint(self):
        """Test HyperbolicTangent with custom midpoint."""
        tanh_cfun = HyperbolicTangent(
            a=np.array([1.0]), midpoint=np.array([5.0]), sigma=np.array([1.0])
        )
        x = np.array([[0.0, 5.0, 10.0]])
        y = tanh_cfun.pre(x, mode=0)

        assert y[0, 0] < 0.01  # Far left of sigmoid
        assert abs(y[0, 1] - 1.0) < 0.01  # At midpoint
        assert y[0, 2] > 1.99  # Far right of sigmoid

    def test_hyperbolic_tangent_amplitude(self):
        """Test HyperbolicTangent amplitude parameter."""
        a_values = [0.5, 1.0, 2.0]
        x = np.array([[10.0]])

        for a_val in a_values:
            tanh_cfun = HyperbolicTangent(a=np.array([a_val]))
            y = tanh_cfun.pre(x, mode=0)
            expected = np.array([[2.0 * a_val]])  # tanh(+inf) = 1, so a*(1+1) = 2a
            np.testing.assert_allclose(y, expected, rtol=1e-2, atol=1e-2)

    def test_hyperbolic_tangent_width(self):
        """Test HyperbolicTangent width parameter (sigma)."""
        x = np.linspace(-5, 5, 100)
        tanh_narrow = HyperbolicTangent(sigma=np.array([0.5]))
        tanh_wide = HyperbolicTangent(sigma=np.array([2.0]))
        result_narrow = tanh_narrow.pre(x.reshape(1, -1), mode=0)[0, :]
        result_wide = tanh_wide.pre(x.reshape(1, -1), mode=0)[0, :]

        assert np.abs(result_narrow[40] - result_narrow[60]) > np.abs(
            result_wide[40] - result_wide[60]
        )

    def test_hyperbolic_tangent_multidimensional(self):
        """Test HyperbolicTangent handles multidimensional arrays."""
        tanh_cfun = HyperbolicTangent()
        x = np.random.randn(2, 3, 4)
        y = tanh_cfun.pre(x, mode=0)
        assert y.shape == x.shape
        assert np.all(y >= 0.0)  # Output ranges from 0 to 2
        assert np.all(y <= 2.0)

    def test_hyperbolic_tangent_post_identity(self):
        """Test HyperbolicTangent.post() returns input unchanged."""
        tanh_cfun = HyperbolicTangent()
        x = np.array([1.0, 2.0, 3.0])
        result = tanh_cfun.post(x)
        np.testing.assert_array_equal(result, x)

    def test_hyperbolic_tangent_formula(self):
        """Verify HyperbolicTangent implements correct formula."""
        tanh_cfun = HyperbolicTangent(
            a=np.array([2.0]), midpoint=np.array([1.0]), sigma=np.array([0.5])
        )
        x = np.array([0.0, 1.0, 2.0])
        result = tanh_cfun.pre(x.reshape(1, -1), mode=0)[0, :]
        expected = 2.0 * (1.0 + np.tanh((x - 1.0) / 0.5))
        np.testing.assert_array_almost_equal(result, expected)


class TestSigmoidalJansenRitCoupling:
    """Tests for :class:`~tvb.simulator.hybrid.coupling.SigmoidalJansenRit` (Jansen–Rit sigmoid).

    Verifies that output equals ``e0`` at the threshold ``x=v0``, saturates
    near ``2*e0`` for large inputs, and that the ``a``, ``e0``, ``r``, and
    ``v0`` parameters control amplitude, maximum firing rate, slope, and
    threshold respectively.  ``post()`` must be the identity.
    """

    def test_jansen_rit_default_parameters(self):
        """Test SigmoidalJansenRit with default parameters."""
        jr_sigmoid = SigmoidalJansenRit()
        x = np.array([[-10.0, 6.0, 20.0]])
        y = jr_sigmoid.pre(x, mode=0)

        assert y[0, 0] < 0.1  # Near zero for low input
        assert abs(y[0, 1] - 2.5) < 0.01  # At threshold, output = e0 = 2.5
        assert y[0, 2] > 4.9  # Near 2*e0 = 5 for high input

    def test_jansen_rit_at_threshold(self):
        """Test SigmoidalJansenRit at threshold (x = v0)."""
        jr_sigmoid = SigmoidalJansenRit(
            a=np.array([1.0]),
            e0=np.array([2.5]),
            r=np.array([0.56]),
            v0=np.array([6.0]),
        )
        x = np.array([[6.0]])
        y = jr_sigmoid.pre(x, mode=0)
        expected = np.array([[2.5]])  # At threshold: a * (2*e0) / 2 = a * e0
        np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-8)

    def test_jansen_rit_amplitude(self):
        """Test SigmoidalJansenRit amplitude parameter."""
        a_values = [0.5, 1.0, 2.0]
        x = np.array([[20.0]])

        for a_val in a_values:
            jr_sigmoid = SigmoidalJansenRit(a=np.array([a_val]))
            y = jr_sigmoid.pre(x, mode=0)
            expected = np.array([[5.0 * a_val]])  # Near 2*a*e0
            np.testing.assert_allclose(y, expected, rtol=1e-2, atol=1e-2)

    def test_jansen_rit_e0_parameter(self):
        """Test SigmoidalJansenRit e0 parameter (max firing rate)."""
        e0_values = [1.0, 2.5, 5.0]
        x = np.array([[20.0]])

        for e0_val in e0_values:
            jr_sigmoid = SigmoidalJansenRit(e0=np.array([e0_val]))
            y = jr_sigmoid.pre(x, mode=0)
            expected = np.array([[2.0 * e0_val]])  # Near 2*e0
            np.testing.assert_allclose(y, expected, rtol=1e-2, atol=1e-2)

    def test_jansen_rit_r_parameter(self):
        """Test SigmoidalJansenRit r parameter (slope)."""
        r_values = [0.1, 0.56, 2.0]
        x = np.array([[6.0, 7.0]])

        for r_val in r_values:
            jr_sigmoid = SigmoidalJansenRit(r=np.array([r_val]))
            y = jr_sigmoid.pre(x, mode=0)
            assert y.shape == x.shape

    def test_jansen_rit_v0_parameter(self):
        """Test SigmoidalJansenRit v0 parameter (threshold)."""
        v0_values = [0.0, 6.0, 10.0]

        for v0_val in v0_values:
            jr_sigmoid = SigmoidalJansenRit(v0=np.array([v0_val]))
            x = np.array([[v0_val]])
            y = jr_sigmoid.pre(x, mode=0)
            expected = np.array([[2.5]])  # At threshold, output = e0
            np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-8)

    def test_jansen_rit_multidimensional(self):
        """Test SigmoidalJansenRit handles multidimensional arrays."""
        jr_sigmoid = SigmoidalJansenRit()
        x = np.random.randn(2, 3, 4)
        y = jr_sigmoid.pre(x, mode=0)
        assert y.shape == x.shape
        assert np.all(y >= 0.0)  # Output is non-negative
        assert np.all(y <= 5.0)  # Output <= 2*e0

    def test_jansen_rit_post_identity(self):
        """Test SigmoidalJansenRit.post() returns input unchanged."""
        jr_sigmoid = SigmoidalJansenRit()
        x = np.array([1.0, 2.0, 3.0])
        result = jr_sigmoid.post(x)
        np.testing.assert_array_equal(result, x)

    def test_jansen_rit_formula(self):
        """Verify SigmoidalJansenRit implements correct formula."""
        jr_sigmoid = SigmoidalJansenRit(
            a=np.array([2.0]), e0=np.array([3.0]), r=np.array([0.5]), v0=np.array([5.0])
        )
        x = np.array([5.0])
        result = jr_sigmoid.pre(x.reshape(1, -1), mode=0)[0, :]
        expected = 2.0 * (2 * 3.0) / (1.0 + np.exp(0.5 * (5.0 - 5.0)))
        np.testing.assert_array_almost_equal(result, expected)


class TestPreSigmoidalCoupling:
    """Tests for :class:`~tvb.simulator.hybrid.coupling.PreSigmoidal` (``H*(Q+tanh(G*(P*x-θ)))``).

    Verifies boundary values at large inputs, the zero-crossing at the
    threshold ``x=θ/P``, and each of the five parameters (``H``, ``Q``,
    ``G``, ``P``, ``theta``).  ``post()`` must be the identity.
    """

    def test_pre_sigmoidal_default_parameters(self):
        """Test PreSigmoidal with default parameters."""
        pre_sigmoid = PreSigmoidal()
        x = np.array([[-10.0, 0.0, 10.0]])
        y = pre_sigmoid.pre(x, mode=0)

        assert y[0, 0] < -0.99  # tanh(-inf) = -1, so H*(Q-1) = -1
        assert abs(y[0, 1]) < 0.01  # tanh(0) = 0, so H*(Q+0) = 0
        assert y[0, 2] > 0.99  # tanh(+inf) = 1, so H*(Q+1) = 1

    def test_pre_sigmoidal_at_threshold(self):
        """Test PreSigmoidal at threshold (x = theta)."""
        pre_sigmoid = PreSigmoidal(
            H=np.array([1.0]),
            Q=np.array([0.0]),
            G=np.array([1.0]),
            P=np.array([1.0]),
            theta=np.array([0.0]),
        )
        x = np.array([[0.0]])
        y = pre_sigmoid.pre(x, mode=0)
        expected = np.array([[0.0]])  # H*(Q+tanh(0)) = 1*(0+0) = 0
        np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-8)

    def test_pre_sigmoidal_large_positive(self):
        """Test PreSigmoidal at large positive values."""
        pre_sigmoid = PreSigmoidal(
            H=np.array([1.0]),
            Q=np.array([0.0]),
            G=np.array([1.0]),
            P=np.array([1.0]),
            theta=np.array([0.0]),
        )
        x = np.array([[10.0]])
        y = pre_sigmoid.pre(x, mode=0)
        expected = np.array([[1.0]])  # H*(Q+tanh(+inf)) = 1*(0+1) = 1
        np.testing.assert_allclose(y, expected, rtol=1e-2, atol=1e-2)

    def test_pre_sigmoidal_baseline_q(self):
        """Test PreSigmoidal baseline parameter (Q)."""
        q_values = [0.0, 1.0, -1.0]
        x = np.array([[10.0]])

        for q_val in q_values:
            pre_sigmoid = PreSigmoidal(Q=np.array([q_val]))
            y = pre_sigmoid.pre(x, mode=0)
            expected = np.array([[1.0 + q_val]])  # tanh(+inf) = 1
            np.testing.assert_allclose(y, expected, rtol=1e-2, atol=1e-2)

    def test_pre_sigmoidal_amplitude_h(self):
        """Test PreSigmoidal amplitude parameter (H)."""
        h_values = [0.5, 1.0, 2.0]
        x = np.array([[10.0]])

        for h_val in h_values:
            pre_sigmoid = PreSigmoidal(H=np.array([h_val]))
            y = pre_sigmoid.pre(x, mode=0)
            expected = np.array([[h_val]])  # tanh(+inf) = 1, Q=0
            np.testing.assert_allclose(y, expected, rtol=1e-2, atol=1e-2)

    def test_pre_sigmoidal_gain_g(self):
        """Test PreSigmoidal gain parameter (G)."""
        g_values = [0.5, 1.0, 2.0]
        x = np.array([[5.0]])

        for g_val in g_values:
            pre_sigmoid = PreSigmoidal(G=np.array([g_val]))
            y = pre_sigmoid.pre(x, mode=0)
            assert y.shape == x.shape

    def test_pre_sigmoidal_projection_p(self):
        """Test PreSigmoidal projection parameter (P)."""
        p_values = [-1.0, 0.5, 1.0]
        x = np.array([[5.0]])

        for p_val in p_values:
            pre_sigmoid = PreSigmoidal(P=np.array([p_val]))
            y = pre_sigmoid.pre(x, mode=0)
            assert y.shape == x.shape

    def test_pre_sigmoidal_threshold_theta(self):
        """Test PreSigmoidal threshold parameter (theta)."""
        theta_values = [0.0, 5.0, 10.0]

        for theta_val in theta_values:
            pre_sigmoid = PreSigmoidal(theta=np.array([theta_val]))
            x = np.array([[theta_val]])
            y = pre_sigmoid.pre(x, mode=0)
            expected = np.array([[0.0]])  # At threshold, tanh(0) = 0
            np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-8)

    def test_pre_sigmoidal_multidimensional(self):
        """Test PreSigmoidal handles multidimensional arrays."""
        pre_sigmoid = PreSigmoidal()
        x = np.random.randn(2, 3, 4)
        y = pre_sigmoid.pre(x, mode=0)
        assert y.shape == x.shape

    def test_pre_sigmoidal_post_identity(self):
        """Test PreSigmoidal.post() returns input unchanged."""
        pre_sigmoid = PreSigmoidal()
        x = np.array([1.0, 2.0, 3.0])
        result = pre_sigmoid.post(x)
        np.testing.assert_array_equal(result, x)

    def test_pre_sigmoidal_formula(self):
        """Verify PreSigmoidal implements correct formula."""
        pre_sigmoid = PreSigmoidal(
            H=np.array([2.0]),
            Q=np.array([1.0]),
            G=np.array([0.5]),
            P=np.array([1.0]),
            theta=np.array([2.0]),
        )
        x = np.array([2.0])
        result = pre_sigmoid.pre(x.reshape(1, -1), mode=0)[0, :]
        expected = 2.0 * (1.0 + np.tanh(0.5 * (1.0 * 2.0 - 2.0)))
        np.testing.assert_array_almost_equal(result, expected)


class TestCouplingIntegrationExtended:
    """Test new coupling functions integrated with projections."""

    @pytest.fixture
    def simple_subnet(self):
        """Create a simple subnetwork."""
        from tvb.simulator.models import Generic2dOscillator

        subnet = Subnetwork(
            name="test_subnet",
            model=Generic2dOscillator(),
            scheme=HeunDeterministic(dt=0.1),
            nnodes=2,
        )
        subnet.connectivity = Connectivity(
            weights=np.eye(2),
            tract_lengths=np.zeros((2, 2)),
            centres=np.ones((2, 3)),
            region_labels=np.array(["region_0", "region_1"]),
            speed=np.array([3.0]),
            number_of_regions=2,
        )
        subnet.configure()
        return subnet

    def test_projection_with_kuramoto_coupling(self, simple_subnet):
        """Test projection with Kuramoto coupling."""
        from tvb.simulator.hybrid.projection_utils import create_intra_projection

        weights = sp.csr_matrix(np.eye(2))
        coupling = Kuramoto(a=np.array([1.0]))
        proj = create_intra_projection(
            subnet=simple_subnet,
            source_cvar=[0],
            target_cvar=[0],
            weights=weights,
            coupling=coupling,
        )
        assert proj.cfun is coupling
        assert isinstance(proj.cfun, Kuramoto)

    def test_projection_with_difference_coupling(self, simple_subnet):
        """Test projection with Difference coupling."""
        from tvb.simulator.hybrid.projection_utils import create_intra_projection

        weights = sp.csr_matrix(np.eye(2))
        coupling = Difference(a=np.array([1.0]))
        proj = create_intra_projection(
            subnet=simple_subnet,
            source_cvar=[0],
            target_cvar=[0],
            weights=weights,
            coupling=coupling,
        )
        assert proj.cfun is coupling
        assert isinstance(proj.cfun, Difference)

    def test_projection_with_hyperbolic_tangent_coupling(self, simple_subnet):
        """Test projection with HyperbolicTangent coupling."""
        from tvb.simulator.hybrid.projection_utils import create_intra_projection

        weights = sp.csr_matrix(np.eye(2))
        coupling = HyperbolicTangent()
        proj = create_intra_projection(
            subnet=simple_subnet,
            source_cvar=[0],
            target_cvar=[0],
            weights=weights,
            coupling=coupling,
        )
        assert proj.cfun is coupling
        assert isinstance(proj.cfun, HyperbolicTangent)

    def test_projection_with_jansen_rit_coupling(self, simple_subnet):
        """Test projection with SigmoidalJansenRit coupling."""
        from tvb.simulator.hybrid.projection_utils import create_intra_projection

        weights = sp.csr_matrix(np.eye(2))
        coupling = SigmoidalJansenRit()
        proj = create_intra_projection(
            subnet=simple_subnet,
            source_cvar=[0],
            target_cvar=[0],
            weights=weights,
            coupling=coupling,
        )
        assert proj.cfun is coupling
        assert isinstance(proj.cfun, SigmoidalJansenRit)

    def test_projection_with_pre_sigmoidal_coupling(self, simple_subnet):
        """Test projection with PreSigmoidal coupling."""
        from tvb.simulator.hybrid.projection_utils import create_intra_projection

        weights = sp.csr_matrix(np.eye(2))
        coupling = PreSigmoidal()
        proj = create_intra_projection(
            subnet=simple_subnet,
            source_cvar=[0],
            target_cvar=[0],
            weights=weights,
            coupling=coupling,
        )
        assert proj.cfun is coupling
        assert isinstance(proj.cfun, PreSigmoidal)

    def test_projection_apply_with_kuramoto_coupling(self, simple_subnet):
        """Test that projection.apply() correctly uses Kuramoto coupling."""
        from tvb.simulator.hybrid.projection_utils import create_intra_projection

        weights = sp.csr_matrix(np.eye(2))
        coupling = Kuramoto(a=np.array([1.0]))
        proj = create_intra_projection(
            subnet=simple_subnet,
            source_cvar=[0],
            target_cvar=[0],
            weights=weights,
            coupling=coupling,
        )

        proj.configure_buffer(
            n_vars_src=simple_subnet.model.nvar, n_nodes_src=2, n_modes_src=1
        )

        tgt = np.zeros((simple_subnet.model.nvar, 2, 1))

        source_state = np.ones((simple_subnet.model.nvar, 2, 1))
        source_state[0, :, 0] = [np.pi / 2, 3 * np.pi / 2]
        proj.update_buffer(source_state, t=10)
        proj.apply(tgt, t=10, n_modes=1)

        expected = np.array([1.0, -1.0])  # sin(pi/2) = 1, sin(3pi/2) = -1
        np.testing.assert_array_almost_equal(tgt[0, :, 0], expected, decimal=5)

    def test_projection_apply_with_difference_coupling(self, simple_subnet):
        """Test that projection.apply() correctly uses Difference coupling."""
        from tvb.simulator.hybrid.projection_utils import create_intra_projection

        weights = sp.csr_matrix(np.eye(2))
        coupling = Difference(a=np.array([2.0]))
        proj = create_intra_projection(
            subnet=simple_subnet,
            source_cvar=[0],
            target_cvar=[0],
            weights=weights,
            coupling=coupling,
        )

        proj.configure_buffer(
            n_vars_src=simple_subnet.model.nvar, n_nodes_src=2, n_modes_src=1
        )

        tgt = np.zeros((simple_subnet.model.nvar, 2, 1))

        source_state = np.ones((simple_subnet.model.nvar, 2, 1))
        source_state[0, :, 0] = [3.0, 5.0]
        proj.update_buffer(source_state, t=10)
        proj.apply(tgt, t=10, n_modes=1)

        expected = 2.0 * np.array([3.0, 5.0])
        np.testing.assert_array_almost_equal(tgt[0, :, 0], expected)


class TestCouplingEdgeCases:
    """Test edge cases for coupling functions."""

    def test_kuramoto_zero_strength(self):
        """Test Kuramoto with zero strength."""
        kuramoto = Kuramoto(a=np.array([0.0]))
        x = np.array([[1.0, 2.0, 3.0]])
        y = kuramoto.post(x, mode=0)
        expected = np.zeros_like(x)
        np.testing.assert_array_equal(y, expected)

    def test_difference_zero_strength(self):
        """Test Difference with zero strength."""
        diff = Difference(a=np.array([0.0]))
        x = np.array([[1.0, 2.0, 3.0]])
        y = diff.post(x, mode=0)
        expected = np.zeros_like(x)
        np.testing.assert_array_equal(y, expected)

    def test_hyperbolic_tangent_zero_amplitude(self):
        """Test HyperbolicTangent with zero amplitude."""
        tanh_cfun = HyperbolicTangent(a=np.array([0.0]))
        x = np.array([[1.0, 2.0, 3.0]])
        y = tanh_cfun.pre(x, mode=0)
        expected = np.zeros_like(x)
        np.testing.assert_array_equal(y, expected)

    def test_jansen_rit_zero_amplitude(self):
        """Test SigmoidalJansenRit with zero amplitude."""
        jr_sigmoid = SigmoidalJansenRit(a=np.array([0.0]))
        x = np.array([[1.0, 2.0, 3.0]])
        y = jr_sigmoid.pre(x, mode=0)
        expected = np.zeros_like(x)
        np.testing.assert_array_equal(y, expected)

    def test_pre_sigmoidal_zero_amplitude(self):
        """Test PreSigmoidal with zero amplitude."""
        pre_sigmoid = PreSigmoidal(H=np.array([0.0]))
        x = np.array([[1.0, 2.0, 3.0]])
        y = pre_sigmoid.pre(x, mode=0)
        expected = np.zeros_like(x)
        np.testing.assert_array_equal(y, expected)

    def test_hyperbolic_tangent_zero_sigma(self):
        """Test HyperbolicTangent with very small sigma."""
        tanh_cfun = HyperbolicTangent(sigma=np.array([1e-10]))
        x = np.array([[0.0]])
        y = tanh_cfun.pre(x, mode=0)
        assert y.shape == x.shape

    def test_pre_sigmoidal_zero_gain(self):
        """Test PreSigmoidal with zero gain."""
        pre_sigmoid = PreSigmoidal(G=np.array([0.0]))
        x = np.array([[1.0, 2.0, 3.0]])
        y = pre_sigmoid.pre(x, mode=0)
        expected = np.array([[0.0, 0.0, 0.0]])  # Q=0, tanh(-theta) ~ 0 for theta>0
        np.testing.assert_array_almost_equal(y, expected, decimal=5)
