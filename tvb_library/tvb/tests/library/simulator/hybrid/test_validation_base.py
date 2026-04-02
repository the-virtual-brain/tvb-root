"""
Base class and helper functions for validation tests.
"""

import numpy as np
from tvb.tests.library.simulator.hybrid.test_base import BaseHybridTest


class ValidationTestBase(BaseHybridTest):
    """Base class for validation tests with helper functions."""

    def assert_simulation_equivalent(self, classic_y, hybrid_y, rtol=1e-4, atol=1e-5):
        """Assert that classic and hybrid outputs are equivalent.

        Parameters
        ----------
        classic_y : ndarray
            Output from classic TVB simulation
        hybrid_y : ndarray
            Output from hybrid TVB simulation
        rtol : float, default=1e-4
            Relative tolerance for comparison
        atol : float, default=1e-5
            Absolute tolerance for comparison
        """
        self.assertEqual(classic_y.shape, hybrid_y.shape, "Output shapes do not match")

        for i in range(classic_y.shape[1]):
            np.testing.assert_allclose(
                classic_y[:, i, :, :],
                hybrid_y[:, i, :, :],
                rtol=rtol,
                atol=atol,
                err_msg=f"Mismatch in state variable {i}",
            )

    def assert_statistics_equivalent(self, classic_y, hybrid_y, places=4):
        """Assert that statistical measures are equivalent.

        Parameters
        ----------
        classic_y : ndarray
            Output from classic TVB simulation
        hybrid_y : ndarray
            Output from hybrid TVB simulation
        places : int, default=4
            Number of decimal places for comparison
        """
        classic_mean, classic_std = classic_y.mean(), classic_y.std()
        classic_max, classic_min = classic_y.max(), classic_y.min()

        hybrid_mean, hybrid_std = hybrid_y.mean(), hybrid_y.std()
        hybrid_max, hybrid_min = hybrid_y.max(), hybrid_y.min()

        self.assertAlmostEqual(classic_mean, hybrid_mean, places=places)
        self.assertAlmostEqual(classic_std, hybrid_std, places=places)
        self.assertAlmostEqual(classic_max, hybrid_max, places=places)
        self.assertAlmostEqual(classic_min, hybrid_min, places=places)

    def compute_order_parameter(self, phases):
        """Compute Kuramoto order parameter for synchronization.

        Parameters
        ----------
        phases : ndarray
            Phase values, shape (..., n_nodes)

        Returns
        -------
        ndarray
            Order parameter values, ranging from 0 (unsynchronized) to 1 (fully synchronized)
        """
        return np.abs(np.mean(np.exp(1j * phases), axis=-1))

    def phase_difference(self, theta1, theta2):
        """Compute phase difference wrapped to [-pi, pi].

        Parameters
        ----------
        theta1 : ndarray
            First phase angle
        theta2 : ndarray
            Second phase angle

        Returns
        -------
        ndarray
            Wrapped phase difference in range [-pi, pi]
        """
        diff = theta2 - theta1
        return np.angle(np.exp(1j * diff))
