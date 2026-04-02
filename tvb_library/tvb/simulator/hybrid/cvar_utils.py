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
