"""
Tests for cvar_utils module.
"""

import numpy as np
import pytest
from tvb.simulator.models import JansenRit, ReducedSetFitzHughNagumo
from tvb.simulator.hybrid.cvar_utils import resolve_cvar_names, validate_cvar_indices


class TestCvarUtils:
    """Tests for cvar name resolution utilities."""

    def test_resolve_single_string(self):
        """Test resolving a single cvar name."""
        model = JansenRit()
        model.configure()

        indices = resolve_cvar_names(model, "y0")
        np.testing.assert_array_equal(indices, np.array([0]))

    def test_resolve_list_of_strings(self):
        """Test resolving multiple cvar names."""
        model = JansenRit()
        model.configure()

        indices = resolve_cvar_names(model, ["y0", "y1"])
        np.testing.assert_array_equal(indices, np.array([0, 1]))

    def test_resolve_repeated_names(self):
        """Test resolving repeated cvar names (more cvars than svars)."""
        model = JansenRit()
        model.configure()

        # Can have more cvars than state variables
        indices = resolve_cvar_names(model, ["y0", "y0", "y1"])
        np.testing.assert_array_equal(indices, np.array([0, 0, 1]))

    def test_resolve_integer_indices(self):
        """Test that integer indices pass through."""
        model = JansenRit()
        model.configure()

        indices = resolve_cvar_names(model, np.array([0, 1]))
        np.testing.assert_array_equal(indices, np.array([0, 1]))

    def test_resolve_list_of_integers(self):
        """Test that list of integers pass through."""
        model = JansenRit()
        model.configure()

        indices = resolve_cvar_names(model, [0, 1])
        np.testing.assert_array_equal(indices, np.array([0, 1]))

    def test_invalid_cvar_name(self):
        """Test that invalid cvar name raises ValueError."""
        model = JansenRit()
        model.configure()

        with pytest.raises(ValueError, match="Unknown state variable"):
            resolve_cvar_names(model, "invalid_name")

    def test_invalid_cvar_index(self):
        """Test that out-of-range cvar index raises ValueError."""
        model = JansenRit()
        model.configure()

        # JansenRit has 6 state variables (indices 0-5)
        with pytest.raises(ValueError, match="Invalid cvar indices"):
            validate_cvar_indices(model, np.array([10]))

    def test_fitzhugh_nagumo_cvars(self):
        """Test with ReducedSetFitzHughNagumo model."""
        model = ReducedSetFitzHughNagumo()
        model.configure()

        # xi, eta, alpha, beta are the state variables
        indices = resolve_cvar_names(model, "xi")
        np.testing.assert_array_equal(indices, np.array([0]))

        indices = resolve_cvar_names(model, ["xi", "eta", "alpha"])
        np.testing.assert_array_equal(indices, np.array([0, 1, 2]))
