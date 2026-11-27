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

import os
import platform
import sys

import numpy as np

from .backend import CppHybridBackend, SweepResult

__all__ = ["sweep", "available_sweep_params", "default_width"]

_VALID_WIDTHS = (1, 4, 8, 16)


def _default_workers():
    if sys.platform == "emscripten":
        return 1
    return max(1, (os.cpu_count() or 2) // 2)


def default_width(n_sweeps: int, ns: "NetworkSet" = None) -> int:
    """SIMD lane width.  The C++ core instantiates widths ``(1, 4, 8, 16)``;
    we default to ``4`` on Emscripten/WASM (128-bit ``-msimd128``) and
    ARM64/AArch64 (128-bit NEON) and ``8`` on x86_64, falling back to ``1``
    only for very large networks where a SIMD block would be memory-heavy.
    Leftover lanes in a final block are safely padded, so a SIMD width is
    fine even when ``n_sweeps`` is small."""
    arch = platform.machine().lower()
    if sys.platform == "emscripten" or arch in ("arm64", "aarch64"):
        w = 4
    else:
        w = 8
    if ns is not None:
        nodes = sum(int(getattr(sn, "nnodes", 0)) for sn in ns.subnets)
        if nodes * w > 100_000:
            return 1
    return w


def available_sweep_params(ns: "NetworkSet"):
    """List valid swept-parameter keys for *ns*.

    Returns a dict grouping ``coupling`` (aliases + projection attrs) and
    ``model`` (``subnet.param`` dotted names).
    """
    from tvb.simulator.backend.hybrid_common import _NAMED_PARAM_ALIASES
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
    monitor_period: int = 1,
    bold_period: float = None,
    chunk_size: int = None,
    initial_states: list = None,
    features=None,
    grid: bool = False,
    bw: dict = None,
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
        SIMD lane count per batch in ``(1, 4, 8, 16)``; defaults to
        :func:`default_width`.
    workers : int
        Parallel lane-group threads; defaults to half the CPU count (1 on
        Emscripten).
    monitor : str
        Monitor mode forwarded to :meth:`CppHybridBackend.sweep`.
    monitor_period : int
        Sampling period in steps for temporal-average monitors.
    bold_period : float, optional
        BOLD repetition time in ms when requested.
    chunk_size : int, optional
        Step chunk size forwarded to :meth:`CppHybridBackend.sweep`.
    initial_states : list, optional
        Per-subnet initial state arrays forwarded to
        :meth:`CppHybridBackend.sweep`.
    features : sequence of str, optional
        Optional online streaming feature names to compute during the sweep.
    grid : bool
        If True, sweep the Cartesian product rather than pairing values.
        Parameters follow keyword insertion order, with the last varying
        fastest (equivalent to ``meshgrid(..., indexing="ij")`` flattened in
        C order). Each row of ``result.sweep_values`` aligns with the same
        sweep axis in monitor outputs and feature tables.
    bw : dict, optional
        Explicit BW placements forwarded to the backend: ``global`` is a
        BalloonWindkessel monitor and ``local`` maps subnetwork names to such
        monitors. Each result has its own midpoint grid in ``bw_global`` or
        ``bw_local``. BW is fractional BOLD change, not HRF ``monitor='bold'``.
    **sweeps : dict[str, array-like]
        Each key names a parameter to sweep and its value the 1-D values.
        Valid key forms (see :func:`available_sweep_params`):
        ``'coupling_scale'`` alias, ``'{proj_name}.{attr}'`` projection cfun
        attribute, or ``'{subnet}.{param}'`` model parameter.  All arrays must
        have the same length and are paired elementwise (lockstep zip) by
        default. With ``grid=True`` lengths may differ; each must be nonempty
        and one-dimensional.
    """
    params = {name: np.asarray(vals, dtype=np.float32) for name, vals in sweeps.items()}
    if not params:
        raise ValueError("sweep() needs at least one swept parameter, e.g. "
                         "sweep(ns, nstep, coupling_scale=[...])")
    if grid:
        for name, values in params.items():
            if values.ndim != 1 or values.size == 0:
                raise ValueError(f"grid parameter {name!r} must be a nonempty 1-D array")
        params = dict(zip(params, (a.ravel() for a in np.meshgrid(
            *params.values(), indexing="ij"))))
    n = len(next(iter(params.values())))
    if width is None or width == "auto":
        width = default_width(n, network_set)
    if isinstance(width, bool) or width not in _VALID_WIDTHS:
        raise ValueError(
            f"width={width}: supported SIMD widths are {_VALID_WIDTHS}. "
            f"Pass width in {_VALID_WIDTHS} or leave it auto."
        )
    width = int(width)
    if workers is None:
        workers = _default_workers()
    sweep_kwargs = dict(
        nstep=nstep,
        progress=progress,
        width=width,
        n_workers=workers,
        monitor=monitor,
        monitor_period=monitor_period,
        bold_period=bold_period,
        chunk_size=chunk_size,
        initial_states=initial_states,
    )
    if bw is not None:
        sweep_kwargs["bw"] = bw
    if features is not None:
        sweep_kwargs["features"] = features
    if network_set is not None:
        try:
            CppHybridBackend._resolve_sweep(network_set, params)
        except ValueError as e:
            opts = available_sweep_params(network_set)
            raise ValueError(
                f"{e}\nTry one of: coupling {opts['coupling']}; "
                f"model {opts['model'][:12]}...; "
                f"projection attrs e.g. {opts['projection'][:6]}"
            ) from e
    return CppHybridBackend().sweep(
        network_set, params, **sweep_kwargs,
    )
