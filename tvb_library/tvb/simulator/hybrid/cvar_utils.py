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
Utility functions for resolving coupling variable names to indices.

This module provides functions to convert human-readable state variable names
into integer indices for use in projections.
"""

import numpy as np


def resolve_cvar_names(model, cvar_spec):
    """Resolve coupling variable names to integer indices.

    Supports string names, lists of names, or existing integer indices.
    Validates that names exist in the model's state_variables.

    Parameters
    ----------
    model : Model
        The model whose state_variables will be used for resolution.
    cvar_spec : str, list of str, or np.ndarray of int
        Coupling variable specification as:
        - Single string: 'y0' → [0]
        - List of strings: ['y0', 'y1'] → [0, 1]
        - List of ints: [0, 1] → [0, 1] (for more cvars than svars)
        - Single int: 0 → [0]

    Returns
    -------
    np.ndarray
        Array of integer indices for coupling variables.

    Raises
    ------
    ValueError
        If cvar_spec contains unknown state variable names.

    Examples
    --------
    >>> from tvb.simulator.models import JansenRit
    >>> model = JansenRit()
    >>> model.configure()
    >>> resolve_cvar_names(model, 'y0')
    array([0])
    >>> resolve_cvar_names(model, ['y0', 'y1'])
    array([0, 1])
    >>> resolve_cvar_names(model, ['y0', 'y0', 'y1'])
    array([0, 0, 1])
    """
    # If already an integer array, return as-is
    if isinstance(cvar_spec, (np.ndarray, list, tuple)):
        # Check if it's already integer indices
        if all(isinstance(x, (int, np.integer)) for x in cvar_spec):
            return np.atleast_1d(np.array(cvar_spec, dtype=np.int_))

    # Convert string to list for uniform handling
    if isinstance(cvar_spec, str):
        cvar_list = [cvar_spec]
    elif isinstance(cvar_spec, list):
        cvar_list = cvar_spec
    elif isinstance(cvar_spec, np.ndarray):
        if cvar_spec.dtype.kind in ["U", "S", "O"]:  # String or object array
            cvar_list = list(cvar_spec)
        else:
            # Already numeric
            return np.atleast_1d(cvar_spec.astype(np.int_))
    else:
        raise TypeError(
            f"cvar_spec must be str, list, or np.ndarray, got {type(cvar_spec)}"
        )

    # Resolve each name to an index
    indices = []
    for cvar_name in cvar_list:
        if isinstance(cvar_name, (int, np.integer)):
            # Already an integer
            indices.append(int(cvar_name))
        elif isinstance(cvar_name, str):
            try:
                idx = model.state_variables.index(cvar_name)
                indices.append(idx)
            except ValueError:
                raise ValueError(
                    f"Unknown state variable '{cvar_name}' for model "
                    f"{model.__class__.__name__}. Available variables: "
                    f"{list(model.state_variables)}"
                )
        else:
            raise TypeError(f"cvar element must be str or int, got {type(cvar_name)}")

    return np.array(indices, dtype=np.int_)


def _cvar_names(model):
    """Return the list of coupling variable names for *model*.

    These are the entries of ``model.state_variables`` at positions given by
    ``model.cvar`` — i.e. the state variables that participate in coupling.
    """
    return [model.state_variables[int(i)] for i in model.cvar]


def _normalize_to_list(cvar_spec):
    """Coerce *cvar_spec* to a flat Python list (strings or ints)."""
    if isinstance(cvar_spec, str):
        return [cvar_spec]
    if isinstance(cvar_spec, (list, tuple)):
        return list(cvar_spec)
    if isinstance(cvar_spec, np.ndarray):
        return list(cvar_spec)
    raise TypeError(
        f"cvar_spec must be str, list, tuple, or ndarray, got {type(cvar_spec)}"
    )


def resolve_source_cvar(model, cvar_spec):
    """Resolve a coupling-variable specification to state-variable indices.

    Source cvars index the history buffer, which stores all state variables.
    When strings are given they must name a coupling variable of the model
    (i.e. a name reachable via ``model.cvar``); any other state variable is
    rejected.  Integer inputs are passed through unchanged.

    Parameters
    ----------
    model : Model
        The model whose ``cvar`` and ``state_variables`` define valid names.
    cvar_spec : str, list of str, int, list of int, or ndarray
        Coupling variable specification.

    Returns
    -------
    ndarray of int
        State-variable indices (``model.cvar[slot]`` values) suitable for
        indexing the history buffer.

    Raises
    ------
    ValueError
        If a string name is not a coupling variable of the model.
    """
    # Integer fast-path
    if isinstance(cvar_spec, (int, np.integer)):
        return np.atleast_1d(np.array([int(cvar_spec)], dtype=np.int_))
    if isinstance(cvar_spec, (np.ndarray, list, tuple)):
        flat = _normalize_to_list(cvar_spec)
        if all(isinstance(x, (int, np.integer)) for x in flat):
            return np.atleast_1d(np.array(flat, dtype=np.int_))

    names = _cvar_names(model)
    flat = _normalize_to_list(cvar_spec)
    indices = []
    for item in flat:
        if isinstance(item, (int, np.integer)):
            indices.append(int(item))
        elif isinstance(item, str):
            if item not in names:
                raise ValueError(
                    f"'{item}' is not a coupling variable of "
                    f"{model.__class__.__name__}. "
                    f"Coupling variables are: {names}. "
                    f"(All state variables: {list(model.state_variables)})"
                )
            slot = names.index(item)
            indices.append(int(model.cvar[slot]))
        else:
            raise TypeError(
                f"cvar element must be str or int, got {type(item)}"
            )
    return np.array(indices, dtype=np.int_)


def resolve_target_cvar(model, cvar_spec):
    """Resolve a coupling-variable specification to coupling-slot indices.

    Target cvars index the coupling array, which has shape
    ``(len(model.cvar), nnodes, modes)``.  Slot 0 corresponds to
    ``model.cvar[0]``, slot 1 to ``model.cvar[1]``, etc.  When strings are
    given they must name a coupling variable of the model; any other state
    variable is rejected.  Integer inputs are passed through unchanged.

    Parameters
    ----------
    model : Model
        The model whose ``cvar`` and ``state_variables`` define valid names.
    cvar_spec : str, list of str, int, list of int, or ndarray
        Coupling variable specification.

    Returns
    -------
    ndarray of int
        Coupling-slot indices (0-based position in ``model.cvar``) suitable
        for indexing the coupling array.

    Raises
    ------
    ValueError
        If a string name is not a coupling variable of the model.
    """
    # Integer fast-path
    if isinstance(cvar_spec, (int, np.integer)):
        return np.atleast_1d(np.array([int(cvar_spec)], dtype=np.int_))
    if isinstance(cvar_spec, (np.ndarray, list, tuple)):
        flat = _normalize_to_list(cvar_spec)
        if all(isinstance(x, (int, np.integer)) for x in flat):
            return np.atleast_1d(np.array(flat, dtype=np.int_))

    names = _cvar_names(model)
    flat = _normalize_to_list(cvar_spec)
    indices = []
    for item in flat:
        if isinstance(item, (int, np.integer)):
            indices.append(int(item))
        elif isinstance(item, str):
            if item not in names:
                raise ValueError(
                    f"'{item}' is not a coupling variable of "
                    f"{model.__class__.__name__}. "
                    f"Coupling variables are: {names}. "
                    f"(All state variables: {list(model.state_variables)})"
                )
            indices.append(names.index(item))
        else:
            raise TypeError(
                f"cvar element must be str or int, got {type(item)}"
            )
    return np.array(indices, dtype=np.int_)


def validate_target_cvar_indices(model, cvar_indices):
    """Validate that *cvar_indices* are valid coupling-slot indices.

    Parameters
    ----------
    model : Model
    cvar_indices : ndarray of int
        Must satisfy ``0 <= idx < len(model.cvar)`` for every element.

    Raises
    ------
    ValueError
        If any index is out of range.
    """
    n_slots = len(model.cvar)
    if np.any(cvar_indices < 0) or np.any(cvar_indices >= n_slots):
        invalid = cvar_indices[(cvar_indices < 0) | (cvar_indices >= n_slots)]
        raise ValueError(
            f"Invalid target cvar indices {invalid} for model "
            f"{model.__class__.__name__}. Model has {n_slots} coupling "
            f"variable(s) (slots 0-{n_slots - 1})."
        )
    return True


def validate_cvar_indices(model, cvar_indices):
    """Validate that cvar indices are within model's range.

    Parameters
    ----------
    model : Model
        The model to validate against.
    cvar_indices : np.ndarray
        Array of integer indices to validate.

    Returns
    -------
    bool
        True if all indices are valid.

    Raises
    ------
    ValueError
        If any index is out of range.
    """
    n_vars = len(model.state_variables)

    if np.any(cvar_indices < 0) or np.any(cvar_indices >= n_vars):
        invalid = cvar_indices[(cvar_indices < 0) | (cvar_indices >= n_vars)]
        raise ValueError(
            f"Invalid cvar indices {invalid} for model "
            f"{model.__class__.__name__}. Model has {n_vars} state "
            f"variables (indices 0-{n_vars - 1})."
        )

    return True
