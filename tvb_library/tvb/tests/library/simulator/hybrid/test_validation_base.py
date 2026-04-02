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
Shared assertion helpers for hybrid-vs-classic equivalence validation tests.

This module provides :class:`ValidationTestBase`, which extends
:class:`BaseHybridTest` with higher-level helpers for comparing the numerical
output of a classic TVB simulation against its hybrid-simulator equivalent:

* :meth:`~ValidationTestBase.assert_simulation_equivalent` — element-wise
  numerical equivalence with configurable tolerances.
* :meth:`~ValidationTestBase.assert_statistics_equivalent` — mean/std/min/max
  agreement when exact equality is not expected (e.g., stochastic runs).
* :meth:`~ValidationTestBase.compute_order_parameter` — Kuramoto *R* for
  synchronisation checks in coupled-oscillator validation tests.
* :meth:`~ValidationTestBase.phase_difference` — wrapped phase difference for
  pairwise synchronisation analysis.
"""

import numpy as np
from tvb.tests.library.simulator.hybrid.test_base import BaseHybridTest


class ValidationTestBase(BaseHybridTest):
    """
    Assertion helpers for hybrid-vs-classic equivalence validation tests.

    Inherits from :class:`BaseHybridTest` so that all concrete subclasses
    have access to the standard cortex-thalamus setup fixtures as well as
    these higher-level comparison utilities.
    """

    def assert_simulation_equivalent(self, classic_y, hybrid_y, rtol=1e-4, atol=1e-5):
        """
        Assert element-wise numerical equivalence of classic and hybrid outputs.

        Both arrays must have identical shapes.  Each state variable slice
        ``[:, i, :, :]`` is compared independently so that per-variable
        failures are reported with an informative ``err_msg``.

        Parameters
        ----------
        classic_y : ndarray, shape (T, nvars, nnodes, nmodes)
            Time-series output from a classic TVB simulation.
        hybrid_y : ndarray, shape (T, nvars, nnodes, nmodes)
            Time-series output from the equivalent hybrid TVB simulation.
        rtol : float, default=1e-4
            Relative tolerance passed to :func:`numpy.testing.assert_allclose`.
        atol : float, default=1e-5
            Absolute tolerance passed to :func:`numpy.testing.assert_allclose`.

        Raises
        ------
        AssertionError
            If shapes differ or any element exceeds the specified tolerances.
        """
        assert classic_y.shape == hybrid_y.shape, "Output shapes do not match"

        for i in range(classic_y.shape[1]):
            np.testing.assert_allclose(
                classic_y[:, i, :, :],
                hybrid_y[:, i, :, :],
                rtol=rtol,
                atol=atol,
                err_msg=f"Mismatch in state variable {i}",
            )

    def assert_statistics_equivalent(self, classic_y, hybrid_y, places=4):
        """
        Assert that summary statistics of classic and hybrid outputs agree.

        Compares global mean, standard deviation, maximum, and minimum of the
        two arrays.  Useful when exact element-wise equality is not expected
        (e.g., due to floating-point non-associativity in stochastic runs) but
        the overall distribution should be the same.

        Parameters
        ----------
        classic_y : ndarray
            Time-series output from a classic TVB simulation.
        hybrid_y : ndarray
            Time-series output from the equivalent hybrid TVB simulation.
        places : int, default=4
            Decimal places used by :func:`numpy.testing.assert_almost_equal`
            for each statistic.

        Raises
        ------
        AssertionError
            If any of the four summary statistics differ beyond *places*
            decimal places.
        """
        classic_mean, classic_std = classic_y.mean(), classic_y.std()
        classic_max, classic_min = classic_y.max(), classic_y.min()

        hybrid_mean, hybrid_std = hybrid_y.mean(), hybrid_y.std()
        hybrid_max, hybrid_min = hybrid_y.max(), hybrid_y.min()

        np.testing.assert_almost_equal(classic_mean, hybrid_mean, decimal=places)
        np.testing.assert_almost_equal(classic_std, hybrid_std, decimal=places)
        np.testing.assert_almost_equal(classic_max, hybrid_max, decimal=places)
        np.testing.assert_almost_equal(classic_min, hybrid_min, decimal=places)

    def compute_order_parameter(self, phases):
        """
        Compute the Kuramoto order parameter *R* for a set of oscillator phases.

        The order parameter is defined as

        .. math::
            R = \\left|\\frac{1}{N} \\sum_{j=1}^{N} e^{i\\theta_j}\\right|

        and quantifies the degree of phase synchronisation: *R* = 1 means
        perfect synchronisation; *R* ≈ 0 indicates incoherence.

        Parameters
        ----------
        phases : ndarray, shape (..., n_nodes)
            Phase angles in radians.  The order parameter is computed over the
            last axis (nodes).

        Returns
        -------
        ndarray, shape (...)
            Order parameter values in ``[0, 1]``.
        """
        return np.abs(np.mean(np.exp(1j * phases), axis=-1))

    def phase_difference(self, theta1, theta2):
        """
        Compute the signed phase difference *theta2 - theta1* wrapped to
        ``[-pi, pi]``.

        Uses the complex-exponential trick
        ``angle(exp(i*(theta2 - theta1)))`` to avoid branch-cut artefacts
        in direct subtraction.

        Parameters
        ----------
        theta1 : ndarray
            Reference phase angles in radians.
        theta2 : ndarray
            Comparison phase angles in radians.

        Returns
        -------
        ndarray
            Wrapped phase differences in ``(-pi, pi]``.
        """
        diff = theta2 - theta1
        return np.angle(np.exp(1j * diff))
