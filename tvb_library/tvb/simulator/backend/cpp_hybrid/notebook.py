# -*- coding: utf-8 -*-
"""Notebook-friendly sugar over the SIMD C++ hybrid parameter sweep.

Wraps :class:`CppHybridBackend.sweep` with ergonomics for interactive use:

* build the swept-parameter dict from bare names (``subnet.param``,
  ``proj.attr`` or the ``coupling_scale`` alias), with helpful errors that
  list valid choices;
* tqdm progress; auto SIMD lane width;
* results come back as a :class:`SweepResult` with pandas
  :meth:`to_dataframe` and a floating :meth:`features` table computed over
  the per-sweep monitor time series (see ``tvb.simulator.backend.cpp_hybrid.features``).

Example
-------
.. code-block:: python

    from tvb.simulator.backend.cpp_hybrid import notebook as nb

    res = nb.sweep(network, 2000,
                   coupling_scale=np.linspace(0.2, 2.0, 40),
                   progress=True)
    df  = res.to_dataframe(svar=0)                 # param + per-node columns
    f   = res.features(["mean", "std", "rms"], fs=1/dt)
    ax  = plt.plot(res.sweep_values[:, 0], f["rms"].mean(axis=1))
"""

from __future__ import annotations

import numpy as np

from .backend import CppHybridBackend, SweepResult

__all__ = ["sweep", "available_sweep_params", "default_width"]


def _default_workers():
    import os
    return max(1, (os.cpu_count() or 2) // 2)


def default_width(n_sweeps: int, ns: "NetworkSet" = None) -> int:
    """SIMD lane width.  The C++ core instantiates width 1 (scalar) and 8
    (8-wide SIMD) only; we default to 8 (SIMD) and fall back to 1 only for
    very large networks where an 8-wide block would be memory-heavy.
    Leftover lanes in a final block are safely padded, so a width of 8 is
    fine even when ``n_sweeps`` is small."""
    if ns is not None:
        nodes = sum(int(getattr(sn, "nnodes", 0)) for sn in ns.subnets)
        if nodes * 8 > 100_000:
            return 1
    return 8


def available_sweep_params(ns: "NetworkSet"):
    """List valid swept-parameter keys for *ns*.

    Returns a dict grouping ``coupling`` (aliases + projection attrs) and
    ``model`` (``subnet.param`` dotted names).
    """
    from tvb.simulator.backend.nb_hybrid import _NAMED_PARAM_ALIASES
    acc = {"coupling": sorted(_NAMED_PARAM_ALIASES or []),
           "projection": [], "model": []}
    for sn in ns.subnets:
        model = getattr(sn, "model", None)
        if model is not None:
            names = getattr(model, "global_parameter_names", None) \
                or getattr(model, "parameter_names", None) or []
            for p in names:
                acc["model"].append(f"{sn.name}.{p}")
    for proj in getattr(ns, "projections", []) or []:
        acc["projection"].append(
            f"{proj.source.name}_to_{proj.target.name}.<cfun_attr>")
    return acc


def sweep(
    network_set,
    nstep: int = 100,
    *,
    progress: bool = False,
    width=None,
    workers: int = None,
    monitor: str = "tavg",
    **sweeps,
) -> SweepResult:
    """Run a multithreaded SIMD parameter sweep on *network_set*.

    Parameters
    ----------
    network_set : tvb.simulator.hybrid.NetworkSet
        The (arbitrarily complex, multi-subnet / multi-mode) network to sweep.
    nstep : int
        Integration steps per sweep member.
    progress : bool
        Show a tqdm progress bar over the SIMD lane-groups.
    width : int or "auto"/None
        SIMD lane count per batch; defaults to :func:`default_width`.
    workers : int
        Parallel lane-group threads; defaults to half the CPU count.
    **sweeps : dict[str, array-like]
        Each key names a parameter to sweep and its value the 1-D values.
        Valid key forms (see :func:`available_sweep_params`):
        ``'coupling_scale'`` alias, ``'{proj_name}.{attr}'`` projection cfun
        attribute, or ``'{subnet}.{param}'`` model parameter.  All arrays must
        be the same length (the sweep is the Cartesian cross, one scalar per
        listed dimension).
    """
    params = {name: np.asarray(vals, dtype=np.float32) for name, vals in sweeps.items()}
    if not params:
        raise ValueError("sweep() needs at least one swept parameter, e.g. "
                         "sweep(ns, nstep, coupling_scale=[...])")
    n = len(next(iter(params.values())))
    if width is None or width == "auto":
        width = default_width(n, network_set)
    if width not in (1, 8):
        raise ValueError(
            f"width={width}: the C++ core only supports width 1 (scalar) or "
            f"8 (SIMD lanes). Pass width in (1, 8) or leave it auto."
        )
    if workers is None:
        workers = _default_workers()
    try:
        return CppHybridBackend().sweep(
            network_set, params, nstep=nstep, progress=progress,
            width=width, n_workers=workers, monitor=monitor,
        )
    except ValueError as e:
        opts = available_sweep_params(network_set)
        raise ValueError(
            f"{e}\nTry one of: coupling {opts['coupling']}; "
            f"model {opts['model'][:12]}...; "
            f"projection attrs e.g. {opts['projection'][:6]}"
        ) from e
