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
from tvb.simulator.hybrid.cvar_utils import (
    resolve_cvar_names,
    resolve_source_cvar,
    resolve_target_cvar,
    validate_cvar_indices,
    validate_target_cvar_indices,
)


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


class TestSourceTargetCvarResolution:
    """Tests for :func:`resolve_source_cvar` and :func:`resolve_target_cvar`.

    Both helpers enforce that string names must identify a *coupling variable*
    of the model (i.e. a name in ``model.state_variables`` at a position
    listed in ``model.cvar``), not an arbitrary state variable.

    JansenRit: ``cvar=[1, 2]``, coupling variable names = ``['y1', 'y2']``.
    ``source_cvar`` should return the state-variable index stored in
    ``model.cvar`` (e.g. ``'y1'`` → index 1); ``target_cvar`` should return
    the coupling-slot position in ``model.cvar`` (e.g. ``'y1'`` → slot 0).
    """

    def _jr(self):
        m = JansenRit()
        m.configure()
        return m

    # ------------------------------------------------------------------ #
    # resolve_source_cvar                                                  #
    # ------------------------------------------------------------------ #

    def test_source_cvar_string_returns_state_var_index(self):
        """``resolve_source_cvar('y1')`` for JR returns state-variable index 1.

        JR has ``cvar=[1, 2]`` so 'y1' is coupling slot 0, but the value
        stored in the history buffer is state variable 1 (``model.cvar[0]``).
        """
        m = self._jr()
        result = resolve_source_cvar(m, "y1")
        np.testing.assert_array_equal(result, np.array([1]))

    def test_source_cvar_list_returns_state_var_indices(self):
        """``resolve_source_cvar(['y1', 'y2'])`` for JR returns ``[1, 2]``."""
        m = self._jr()
        result = resolve_source_cvar(m, ["y1", "y2"])
        np.testing.assert_array_equal(result, np.array([1, 2]))

    def test_source_cvar_integer_passthrough(self):
        """Integer arrays pass through ``resolve_source_cvar`` unchanged."""
        m = self._jr()
        result = resolve_source_cvar(m, np.array([1, 2]))
        np.testing.assert_array_equal(result, np.array([1, 2]))

    def test_source_cvar_non_coupling_name_raises(self):
        """A state variable that is NOT a coupling variable raises ValueError.

        For JR, 'y0' is state variable 0 but is not in ``model.cvar=[1,2]``,
        so passing it as a source cvar name must raise.
        """
        m = self._jr()
        with pytest.raises(ValueError, match="not a coupling variable"):
            resolve_source_cvar(m, "y0")

    def test_source_cvar_unknown_name_raises(self):
        """A name not in state_variables at all raises ValueError."""
        m = self._jr()
        with pytest.raises(ValueError, match="not a coupling variable"):
            resolve_source_cvar(m, "nonexistent")

    # ------------------------------------------------------------------ #
    # resolve_target_cvar                                                  #
    # ------------------------------------------------------------------ #

    def test_target_cvar_string_returns_slot_index(self):
        """``resolve_target_cvar('y1')`` for JR returns coupling slot 0.

        'y1' is the first coupling variable (slot 0 in the coupling array),
        even though its state-variable index is 1.
        """
        m = self._jr()
        result = resolve_target_cvar(m, "y1")
        np.testing.assert_array_equal(result, np.array([0]))

    def test_target_cvar_second_slot(self):
        """``resolve_target_cvar('y2')`` for JR returns slot 1."""
        m = self._jr()
        result = resolve_target_cvar(m, "y2")
        np.testing.assert_array_equal(result, np.array([1]))

    def test_target_cvar_list(self):
        """``resolve_target_cvar(['y1', 'y2'])`` for JR returns ``[0, 1]``."""
        m = self._jr()
        result = resolve_target_cvar(m, ["y1", "y2"])
        np.testing.assert_array_equal(result, np.array([0, 1]))

    def test_target_cvar_integer_passthrough(self):
        """Integer arrays pass through ``resolve_target_cvar`` unchanged."""
        m = self._jr()
        result = resolve_target_cvar(m, np.array([0, 1]))
        np.testing.assert_array_equal(result, np.array([0, 1]))

    def test_target_cvar_non_coupling_name_raises(self):
        """A state variable that is NOT a coupling variable raises ValueError."""
        m = self._jr()
        with pytest.raises(ValueError, match="not a coupling variable"):
            resolve_target_cvar(m, "y0")

    # ------------------------------------------------------------------ #
    # source vs target distinction (key correctness property)             #
    # ------------------------------------------------------------------ #

    def test_source_target_differ_for_non_zero_based_cvar(self):
        """source and target resolution give different results for JR.

        For JR ``cvar=[1,2]``, ``'y1'`` must resolve to index 1 as a source
        (state-variable position in the buffer) but to slot 0 as a target
        (coupling-array position).  If they were the same the coupling would
        silently write to the wrong slot.
        """
        m = self._jr()
        src = resolve_source_cvar(m, "y1")
        tgt = resolve_target_cvar(m, "y1")
        assert int(src[0]) != int(tgt[0]), (
            "source and target indices must differ for JR 'y1': "
            f"got src={src}, tgt={tgt}"
        )
        np.testing.assert_array_equal(src, [1])
        np.testing.assert_array_equal(tgt, [0])

    # ------------------------------------------------------------------ #
    # validate_target_cvar_indices                                        #
    # ------------------------------------------------------------------ #

    def test_validate_target_cvar_valid(self):
        """Valid slot indices 0 and 1 pass validation for JR (2 coupling vars)."""
        m = self._jr()
        validate_target_cvar_indices(m, np.array([0, 1]))  # should not raise

    def test_validate_target_cvar_out_of_range_raises(self):
        """Slot index 2 is out of range for JR (only slots 0 and 1 exist)."""
        m = self._jr()
        with pytest.raises(ValueError, match="Invalid target cvar indices"):
            validate_target_cvar_indices(m, np.array([2]))
