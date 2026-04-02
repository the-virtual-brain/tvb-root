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
Tests for coupling functions in the hybrid simulator.

Covers the base pass-through :class:`~tvb.simulator.hybrid.coupling.Coupling`,
:class:`~tvb.simulator.hybrid.coupling.Linear` (affine: ``a*x + b``),
:class:`~tvb.simulator.hybrid.coupling.Scaling` (pure scale: ``a*x``), and
:class:`~tvb.simulator.hybrid.coupling.Sigmoidal` (smooth nonlinearity).
The integration class ``TestCouplingIntegration`` verifies that these
functions can be attached to projections created with
:func:`~tvb.simulator.hybrid.projection_utils.create_intra_projection`.
"""

import numpy as np
import pytest
from scipy import sparse as sp

from tvb.simulator.hybrid.coupling import Coupling, Linear, Scaling, Sigmoidal
from tvb.simulator.hybrid.base_projection import BaseProjection
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.models import JansenRit, ReducedSetFitzHughNagumo
from tvb.simulator.integrators import HeunDeterministic
from tvb.datatypes.connectivity import Connectivity


class TestCouplingBase:
    """Tests for the pass-through behaviour of the base :class:`~tvb.simulator.hybrid.coupling.Coupling` class.

    Verifies that the unspecialised base class returns its input unchanged
    from both ``pre()`` and ``post()``, for 1-D and multi-dimensional arrays.
    """

    def test_coupling_base_identity_pre(self):
        """Test that base Coupling applies identity in pre()."""
        coupling = Coupling()
        x = np.array([1.0, 2.0, 3.0])
        result = coupling.pre(x)
        np.testing.assert_array_equal(result, x)

    def test_coupling_base_identity_post(self):
        """Test that base Coupling applies identity in post()."""
        coupling = Coupling()
        x = np.array([1.0, 2.0, 3.0])
        result = coupling.post(x)
        np.testing.assert_array_equal(result, x)

    def test_coupling_base_multidimensional(self):
        """Test that base Coupling handles multidimensional arrays."""
        coupling = Coupling()
        x = np.random.randn(3, 5, 2)
        result = coupling.post(x)
        np.testing.assert_array_equal(result, x)


class TestLinearCoupling:
    """Tests for :class:`~tvb.simulator.hybrid.coupling.Linear` (affine: ``a*x + b``).

    Verifies scaling, offset, sign handling, and that ``pre()`` is always
    the identity (transformation is applied only in ``post()``).
    """

    def test_linear_default_parameters(self):
        """Test Linear with default parameters (identity: a=1, b=0)."""
        linear = Linear()
        x = np.array([1.0, 2.0, 3.0])
        result = linear.post(x)
        np.testing.assert_array_equal(result, x)

    def test_linear_scaling_only(self):
        """Test Linear with only scaling (b=0)."""
        linear = Linear(a=np.array([0.5]), b=np.array([0.0]))
        x = np.array([1.0, 2.0, 3.0])
        result = linear.post(x)
        expected = np.array([0.5, 1.0, 1.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_linear_scaling_and_offset(self):
        """Test Linear with both scaling and offset."""
        linear = Linear(a=np.array([0.5]), b=np.array([0.1]))
        x = np.array([1.0, 2.0, 3.0])
        result = linear.post(x)
        expected = np.array([0.6, 1.1, 1.6])
        np.testing.assert_array_almost_equal(result, expected)

    def test_linear_negative_values(self):
        """Test Linear handles negative input values."""
        linear = Linear(a=np.array([2.0]), b=np.array([-1.0]))
        x = np.array([-1.0, 0.0, 1.0])
        result = linear.post(x)
        expected = np.array([-3.0, -1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_linear_multidimensional(self):
        """Test Linear handles multidimensional arrays."""
        linear = Linear(a=np.array([0.5]), b=np.array([1.0]))
        x = np.random.randn(2, 3, 4)
        result = linear.post(x)
        expected = 0.5 * x + 1.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_linear_pre_identity(self):
        """Test Linear.pre() returns input unchanged."""
        linear = Linear(a=np.array([2.0]), b=np.array([1.0]))
        x = np.array([1.0, 2.0, 3.0])
        result = linear.pre(x)
        np.testing.assert_array_equal(result, x)


class TestScalingCoupling:
    """Tests for :class:`~tvb.simulator.hybrid.coupling.Scaling` (pure scale: ``a*x``).

    Verifies scale-up, scale-down, zero, and negative scale factors, and
    that ``pre()`` is the identity.
    """

    def test_scaling_default_parameters(self):
        """Test Scaling with default parameters (identity: a=1)."""
        scaling = Scaling()
        x = np.array([1.0, 2.0, 3.0])
        result = scaling.post(x)
        np.testing.assert_array_equal(result, x)

    def test_scaling_scale_down(self):
        """Test Scaling with a < 1."""
        scaling = Scaling(a=np.array([0.5]))
        x = np.array([1.0, 2.0, 3.0])
        result = scaling.post(x)
        expected = np.array([0.5, 1.0, 1.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_scaling_scale_up(self):
        """Test Scaling with a > 1."""
        scaling = Scaling(a=np.array([2.0]))
        x = np.array([1.0, 2.0, 3.0])
        result = scaling.post(x)
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_scaling_negative_scale(self):
        """Test Scaling with negative a."""
        scaling = Scaling(a=np.array([-1.5]))
        x = np.array([1.0, 2.0, 3.0])
        result = scaling.post(x)
        expected = np.array([-1.5, -3.0, -4.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_scaling_zero_scale(self):
        """Test Scaling with a=0 (should produce zeros)."""
        scaling = Scaling(a=np.array([0.0]))
        x = np.array([1.0, 2.0, 3.0])
        result = scaling.post(x)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_scaling_multidimensional(self):
        """Test Scaling handles multidimensional arrays."""
        scaling = Scaling(a=np.array([2.0]))
        x = np.random.randn(3, 4, 5)
        result = scaling.post(x)
        expected = 2.0 * x
        np.testing.assert_array_almost_equal(result, expected)

    def test_scaling_pre_identity(self):
        """Test Scaling.pre() returns input unchanged."""
        scaling = Scaling(a=np.array([2.0]))
        x = np.array([1.0, 2.0, 3.0])
        result = scaling.pre(x)
        np.testing.assert_array_equal(result, x)


class TestSigmoidalCoupling:
    """Tests for :class:`~tvb.simulator.hybrid.coupling.Sigmoidal` (smooth nonlinearity).

    Verifies boundary values, custom midpoint/steepness/width parameters,
    anti-symmetry around the midpoint, and that ``pre()`` is the identity.
    """

    def test_sigmoidal_default_parameters(self):
        """Test Sigmoidal with default parameters."""
        sigmoid = Sigmoidal()
        x = np.array([-10.0, 0.0, 10.0])
        result = sigmoid.post(x)
        assert result[0] < -0.99  # Near lower bound
        assert result[1] < 0.01 and result[1] > -0.01  # Near zero
        assert result[2] > 0.99  # Near upper bound

    def test_sigmoidal_custom_bounds(self):
        """Test Sigmoidal with custom bounds."""
        sigmoid = Sigmoidal(cmin=np.array([0.0]), cmax=np.array([2.0]))
        x = np.array([-10.0, 0.0, 10.0])
        result = sigmoid.post(x)
        assert result[0] < 0.01  # Near lower bound (0)
        assert result[1] > 0.99 and result[1] < 1.01  # Near midpoint (1)
        assert result[2] > 1.99  # Near upper bound (2)

    def test_sigmoidal_custom_midpoint(self):
        """Test Sigmoidal with custom midpoint."""
        sigmoid = Sigmoidal(midpoint=np.array([5.0]))
        x = np.array([0.0, 5.0, 10.0])
        result = sigmoid.post(x)
        assert result[0] < -0.98  # At 0, far left of sigmoid
        assert result[1] > -0.01 and result[1] < 0.01  # At 5, center of sigmoid
        assert result[2] > 0.98  # At 10, far right of sigmoid

    def test_sigmoidal_steepness(self):
        """Test Sigmoidal with different steepness (a)."""
        x = np.linspace(-5, 5, 100)
        sigmoid_gentle = Sigmoidal(a=np.array([1.0]))
        sigmoid_steep = Sigmoidal(a=np.array([5.0]))
        result_gentle = sigmoid_gentle.post(x)
        result_steep = sigmoid_steep.post(x)
        assert np.abs(result_gentle[25] - result_gentle[75]) > 0.5  # Gentle transition
        assert np.abs(result_steep[45] - result_steep[55]) > 0.5  # Steep transition

    def test_sigmoidal_width(self):
        """Test Sigmoidal with different width (sigma)."""
        x = np.linspace(-5, 5, 100)
        sigmoid_narrow = Sigmoidal(sigma=np.array([0.5]))
        sigmoid_wide = Sigmoidal(sigma=np.array([2.0]))
        result_narrow = sigmoid_narrow.post(x)
        result_wide = sigmoid_wide.post(x)
        assert np.abs(result_narrow[45] - result_narrow[55]) > np.abs(
            result_wide[45] - result_wide[55]
        )

    def test_sigmoidal_symmetry(self):
        """Test Sigmoidal is symmetric around midpoint."""
        sigmoid = Sigmoidal()
        x_left = np.array([-1.0])
        x_right = np.array([1.0])
        result_left = sigmoid.post(x_left)
        result_right = sigmoid.post(x_right)
        np.testing.assert_array_almost_equal(result_left, -result_right, decimal=5)

    def test_sigmoidal_multidimensional(self):
        """Test Sigmoidal handles multidimensional arrays."""
        sigmoid = Sigmoidal()
        x = np.random.randn(2, 3, 4)
        result = sigmoid.post(x)
        assert result.shape == x.shape
        assert np.all(result > -1.0)  # All values above cmin
        assert np.all(result < 1.0)  # All values below cmax

    def test_sigmoidal_pre_identity(self):
        """Test Sigmoidal.pre() returns input unchanged."""
        sigmoid = Sigmoidal()
        x = np.array([1.0, 2.0, 3.0])
        result = sigmoid.pre(x)
        np.testing.assert_array_equal(result, x)


class TestCouplingIntegration:
    """Tests that coupling functions can be attached to intra-projections.

    Verifies that :func:`~tvb.simulator.hybrid.projection_utils.create_intra_projection`
    correctly stores ``None``, :class:`~tvb.simulator.hybrid.coupling.Linear`,
    :class:`~tvb.simulator.hybrid.coupling.Scaling`, and
    :class:`~tvb.simulator.hybrid.coupling.Sigmoidal` instances on the
    ``cfun`` attribute of the created projection.
    """

    @pytest.fixture
    def simple_connectivity(self):
        """Create a simple 2x2 connectivity matrix."""
        conn = Connectivity()
        conn.weights = np.array([[1.0, 0.5], [0.5, 1.0]])
        conn.tract_lengths = np.array([[0.0, 10.0], [10.0, 0.0]])
        conn.speed = np.array([3.0])
        conn.number_of_regions = 2
        conn.configure()
        return conn

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

    def test_projection_without_coupling(self, simple_subnet):
        """Test that projection without coupling (coupling=None) works."""
        from tvb.simulator.hybrid.projection_utils import create_intra_projection

        weights = sp.csr_matrix(np.eye(2))
        proj = create_intra_projection(
            subnet=simple_subnet,
            source_cvar=[0],
            target_cvar=[0],
            weights=weights,
            coupling=None,
        )
        assert proj.cfun is None

    def test_projection_with_linear_coupling(self, simple_subnet):
        """Test projection with Linear coupling."""
        from tvb.simulator.hybrid.projection_utils import create_intra_projection

        weights = sp.csr_matrix(np.eye(2))
        coupling = Linear(a=np.array([0.5]), b=np.array([0.1]))
        proj = create_intra_projection(
            subnet=simple_subnet,
            source_cvar=[0],
            target_cvar=[0],
            weights=weights,
            coupling=coupling,
        )
        assert proj.cfun is coupling
        assert isinstance(proj.cfun, Linear)

    def test_projection_with_scaling_coupling(self, simple_subnet):
        """Test projection with Scaling coupling."""
        from tvb.simulator.hybrid.projection_utils import create_intra_projection

        weights = sp.csr_matrix(np.eye(2))
        coupling = Scaling(a=np.array([2.0]))
        proj = create_intra_projection(
            subnet=simple_subnet,
            source_cvar=[0],
            target_cvar=[0],
            weights=weights,
            coupling=coupling,
        )
        assert proj.cfun is coupling
        assert isinstance(proj.cfun, Scaling)

    def test_projection_with_sigmoidal_coupling(self, simple_subnet):
        """Test projection with Sigmoidal coupling."""
        from tvb.simulator.hybrid.projection_utils import create_intra_projection

        weights = sp.csr_matrix(np.eye(2))
        coupling = Sigmoidal()
        proj = create_intra_projection(
            subnet=simple_subnet,
            source_cvar=[0],
            target_cvar=[0],
            weights=weights,
            coupling=coupling,
        )
        assert proj.cfun is coupling
        assert isinstance(proj.cfun, Sigmoidal)

    def test_projection_apply_with_linear_coupling(self, simple_subnet):
        """Test that projection.apply() correctly uses Linear coupling."""
        from tvb.simulator.hybrid.projection_utils import create_intra_projection

        weights = sp.csr_matrix(np.eye(2))
        coupling = Linear(a=np.array([2.0]), b=np.array([1.0]))
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

        expected = 2.0 * np.array([3.0, 5.0]) + 1.0
        np.testing.assert_array_almost_equal(tgt[0, :, 0], expected)

    def test_projection_apply_with_sigmoidal_coupling(self, simple_subnet):
        """Test that projection.apply() correctly uses Sigmoidal coupling."""
        from tvb.simulator.hybrid.projection_utils import create_intra_projection

        weights = sp.csr_matrix(np.eye(2))
        coupling = Sigmoidal(cmin=np.array([0.0]), cmax=np.array([1.0]))
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
        source_state[0, :, 0] = [5.0, -5.0]
        proj.update_buffer(source_state, t=10)
        proj.apply(tgt, t=10, n_modes=1)

        assert tgt[0, 0, 0] > 0.99  # Large positive -> near upper bound
        assert tgt[0, 1, 0] < 0.01  # Large negative -> near lower bound


class TestCouplingMathematicalCorrectness:
    """Test mathematical correctness of coupling functions."""

    def test_linear_formula(self):
        """Verify Linear implements correct formula: a*x + b."""
        linear = Linear(a=np.array([2.0]), b=np.array([3.0]))
        x = np.array([1.0, 2.0, 3.0])
        result = linear.post(x)
        expected = 2.0 * x + 3.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_scaling_formula(self):
        """Verify Scaling implements correct formula: a*x."""
        scaling = Scaling(a=np.array([3.0]))
        x = np.array([1.0, 2.0, 3.0])
        result = scaling.post(x)
        expected = 3.0 * x
        np.testing.assert_array_almost_equal(result, expected)

    def test_sigmoidal_formula(self):
        """Verify Sigmoidal implements correct formula."""
        sigmoid = Sigmoidal(
            cmin=np.array([0.0]),
            cmax=np.array([2.0]),
            a=np.array([1.0]),
            midpoint=np.array([1.0]),
            sigma=np.array([1.0]),
        )
        x = np.array([0.0, 1.0, 2.0])
        result = sigmoid.post(x)
        expected = 0.0 + (2.0 - 0.0) / (1.0 + np.exp(-1.0 * ((x - 1.0) / 1.0)))
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_sigmoidal_bounds(self):
        """Verify Sigmoidal respects bounds."""
        sigmoid = Sigmoidal(cmin=np.array([5.0]), cmax=np.array([10.0]))
        x = np.array([-100.0, 100.0])
        result = sigmoid.post(x)
        assert result[0] >= 5.0 and result[0] < 5.1  # Near lower bound
        assert result[1] <= 10.0 and result[1] > 9.9  # Near upper bound
