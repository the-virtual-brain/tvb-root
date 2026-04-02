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
Unit tests for :mod:`tvb.simulator.hybrid.cvar_utils`.

Covers the two public helpers that translate coupling-variable (cvar)
specifications into integer index arrays:

* :func:`~tvb.simulator.hybrid.cvar_utils.resolve_cvar_names` — accepts a
  string, list of strings, integer array, or list of integers and resolves
  them to a NumPy index array referencing state-variable positions in the
  model.
* :func:`~tvb.simulator.hybrid.cvar_utils.validate_cvar_indices` — raises
  ``ValueError`` when any index exceeds the model's state-variable count.

Test models used: ``JansenRit`` (6 state variables, 2 coupling variables) and
``ReducedSetFitzHughNagumo`` (4 state variables).
"""

import numpy as np
import pytest
from tvb.simulator.models import JansenRit, ReducedSetFitzHughNagumo
from tvb.simulator.hybrid.cvar_utils import resolve_cvar_names, validate_cvar_indices


class TestCvarUtils:
    """
    Tests for :func:`~tvb.simulator.hybrid.cvar_utils.resolve_cvar_names` and
    :func:`~tvb.simulator.hybrid.cvar_utils.validate_cvar_indices`.

    Each test targets a distinct input type or error case to ensure the
    helpers accept all documented input forms and reject invalid ones.
    """

    def test_resolve_single_string(self):
        """
        A single state-variable name resolves to a length-1 index array.

        ``resolve_cvar_names(model, "y0")`` should return ``np.array([0])``
        for JansenRit, where ``y0`` is the first state variable.
        """
        model = JansenRit()
        model.configure()

        indices = resolve_cvar_names(model, "y0")
        np.testing.assert_array_equal(indices, np.array([0]))

    def test_resolve_list_of_strings(self):
        """
        A list of state-variable names resolves to a corresponding index array.

        Verifies that the order of names is preserved: ``["y0", "y1"]``
        must map to ``[0, 1]``, not ``[1, 0]``.
        """
        model = JansenRit()
        model.configure()

        indices = resolve_cvar_names(model, ["y0", "y1"])
        np.testing.assert_array_equal(indices, np.array([0, 1]))

    def test_resolve_repeated_names(self):
        """
        The same variable name may appear more than once in the input list.

        Projections sometimes map a single source variable to several target
        coupling indices; this edge case ensures that ``resolve_cvar_names``
        does not deduplicate the input.
        """
        model = JansenRit()
        model.configure()

        # Can have more cvars than state variables
        indices = resolve_cvar_names(model, ["y0", "y0", "y1"])
        np.testing.assert_array_equal(indices, np.array([0, 0, 1]))

    def test_resolve_integer_indices(self):
        """
        A NumPy integer array passes through unchanged.

        When the caller already has numeric indices, ``resolve_cvar_names``
        should be a no-op (pass-through), returning an equal array.
        """
        model = JansenRit()
        model.configure()

        indices = resolve_cvar_names(model, np.array([0, 1]))
        np.testing.assert_array_equal(indices, np.array([0, 1]))

    def test_resolve_list_of_integers(self):
        """
        A plain Python list of integers is accepted and converted to an array.

        This tests the common user-facing pattern of passing ``[0, 1]``
        rather than ``np.array([0, 1])``.
        """
        model = JansenRit()
        model.configure()

        indices = resolve_cvar_names(model, [0, 1])
        np.testing.assert_array_equal(indices, np.array([0, 1]))

    def test_invalid_cvar_name(self):
        """
        An unrecognised state-variable name raises ``ValueError``.

        The error message must mention ``"Unknown state variable"`` so that
        downstream callers can provide a meaningful diagnostic.
        """
        model = JansenRit()
        model.configure()

        with pytest.raises(ValueError, match="Unknown state variable"):
            resolve_cvar_names(model, "invalid_name")

    def test_invalid_cvar_index(self):
        """
        An out-of-range index raises ``ValueError`` via ``validate_cvar_indices``.

        JansenRit has 6 state variables (valid indices 0–5).  Passing index 10
        should be rejected immediately to prevent silent wrong-dimension
        indexing during the simulation loop.
        """
        model = JansenRit()
        model.configure()

        # JansenRit has 6 state variables (indices 0-5)
        with pytest.raises(ValueError, match="Invalid cvar indices"):
            validate_cvar_indices(model, np.array([10]))

    def test_fitzhugh_nagumo_cvars(self):
        """
        Cvar name resolution works correctly for ``ReducedSetFitzHughNagumo``.

        Verifies both single-name and multi-name resolution against the
        variable ordering ``xi`` (0), ``eta`` (1), ``alpha`` (2), ``beta`` (3)
        defined by that model, confirming the utility is model-agnostic.
        """
        model = ReducedSetFitzHughNagumo()
        model.configure()

        # xi, eta, alpha, beta are the state variables
        indices = resolve_cvar_names(model, "xi")
        np.testing.assert_array_equal(indices, np.array([0]))

        indices = resolve_cvar_names(model, ["xi", "eta", "alpha"])
        np.testing.assert_array_equal(indices, np.array([0, 1, 2]))
