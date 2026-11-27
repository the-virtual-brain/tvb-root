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
Numba backend for Hybrid Simulator.

Generates and JIT-compiles a simulation kernel for NetworkSet objects
whose subnetworks all use the MontbrioPazoRoxin model and deterministic
integrators (Heun or Euler).

Usage::

    from tvb.simulator.backend.nb_hybrid import NbHybridBackend

    backend = NbHybridBackend()
    results = backend.run_network(network_set, nstep=1000)
    # results: list of (times, data) tuples, one per subnetwork

.. moduleauthor:: TVB contributors

Design and implementation plan: ``nb_hybrid_plan.md`` (same directory).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import numpy as np
import autopep8
from typing import List, Optional

from tvb.simulator.monitors import BalloonWindkessel, _reject_balloon_windkessel_sweep

from .hybrid_common import (
    _CFUN_PARAM_ATTRS,
    _CFUN_ATTR_TO_IDX,
    _NAMED_PARAM_ALIASES,
    _compute_chunk_size,
    _aggregate_chunks_to_period,
    _aggregate_raw_outputs,
    _copy_monitor_states,
    _apply_monitors,
    _validate_multi_dt,
    _stim_master_grid,
    _can_merge_subnets,
    _merge_subnet_outputs,
    _merge_and_global_average,
    _merge_and_spatial_average,
    _cfun_type,
    _resolve_named_params,
)
from .templates import MakoUtilMix
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.integrators import (
    HeunDeterministic,
    EulerDeterministic,
    HeunStochastic,
    EulerStochastic,
)


__all__ = [
    "NbHybridBackend",
    "CompiledNetworkFn",
    "NetworkAnalysis",
    "SubnetworkInfo",
    "ProjectionInfo",
    "SweepResult",
]


def _reject_balloon_windkessel(monitors):
    if monitors is not None and any(isinstance(m, BalloonWindkessel) for m in monitors):
        raise NotImplementedError("BalloonWindkessel requires the explicit C++ hybrid backend, not Numba")


# ---------------------------------------------------------------------------
# SweepResult — unified return type for both CPU and GPU sweep
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SweepResult:
    """Container for parameter-sweep results.

    Returned by :meth:`NbHybridBackend.sweep`.  Works identically
    regardless of whether the sweep ran on CPU or GPU.

    Attributes
    ----------
    tavg : dict[str, np.ndarray]
        Per-subnet temporal-average arrays with shape
        ``(n_sweeps, n_samples, n_voi, N_subnet, n_modes)``.
    merged_tavg : np.ndarray
        All subnets concatenated / reordered along the node axis,
        shape ``(n_sweeps, n_samples, n_voi, N_total, n_modes)``.
    ctavg : dict[str, np.ndarray]
        Per-subnet coupling temporal-average with the same sample axis as tavg.
    times : np.ndarray
        Mid-point time vector, shape ``(n_chunks,)``.
    sweep_values : np.ndarray
        The sweep parameter grid, ``(n_sweeps, n_dims)``.
    snapshot : dict or None
        Final execution state for resume (GPU-only for now).
    backend : str
        ``'cpu-seq'``, ``'cpu-prange'``, or ``'cuda'``.
    elapsed : float
        Wall-clock seconds for the sweep execution (excludes compile).
    raw : dict[str, np.ndarray] or None
        Full step-by-step output when ``monitor='raw'``.
    bold : dict[str, np.ndarray] or None
        BOLD signal when ``bold_period`` was given.
    """
    tavg: dict = dataclasses.field(default_factory=dict)
    merged_tavg: np.ndarray = None
    ctavg: dict = dataclasses.field(default_factory=dict)
    times: np.ndarray = None
    sweep_values: np.ndarray = None
    snapshot: Optional[dict] = None
    backend: str = ""
    elapsed: float = 0.0
    raw: Optional[dict] = None
    bold: Optional[dict] = None


# ---------------------------------------------------------------------------
# Lazy-supported-models cache
# ---------------------------------------------------------------------------
# Importing all 27 model modules costs ~2.5 s on first call (each module
# registers Numba dfun_helpers).  We cache the result so the cost is paid
# once per process rather than on every compile().

_SUPPORTED_MODELS_CACHE: tuple = ()


def _get_supported_models_classes() -> tuple:
    """Import all supported model classes and return them as a tuple.

    The imports are expensive (~2.5 s total) so this function caches the
    result in the module-level ``_SUPPORTED_MODELS_CACHE``.  Subsequent
    calls return the cached tuple instantly.
    """
    global _SUPPORTED_MODELS_CACHE
    if _SUPPORTED_MODELS_CACHE:
        return _SUPPORTED_MODELS_CACHE
    from tvb.simulator.models.infinite_theta import (
        MontbrioPazoRoxin,
        CoombesByrne2D,
        CoombesByrne,
        GastSchmidtKnosche_SD,
        GastSchmidtKnosche_SF,
        DumontGutkin,
    )
    from tvb.simulator.models.gast_solla_kennedy import GastSollaKennedy
    from tvb.simulator.models.k_ion_exchange import KIonEx
    from tvb.simulator.models.jansen_rit import JansenRit, ZetterbergJansen
    from tvb.simulator.models.oscillator import (
        Generic2dOscillator,
        SupHopf,
        Kuramoto,
    )
    from tvb.simulator.models.wong_wang import ReducedWongWang
    from tvb.simulator.models.wong_wang_exc_inh import (
        ReducedWongWangExcInh,
        DecoBalancedExcInh,
    )
    from tvb.simulator.models.epileptor import Epileptor, Epileptor2D
    from tvb.simulator.models.epileptorcodim3 import (
        EpileptorCodim3,
        EpileptorCodim3SlowMod,
    )
    from tvb.simulator.models.epileptor_rs import EpileptorRestingState
    from tvb.simulator.models.hopfield import Hopfield
    from tvb.simulator.models.larter_breakspear import LarterBreakspear
    from tvb.simulator.models.wilson_cowan import WilsonCowan
    from tvb.simulator.models.zerlaut import ZerlautAdaptationFirstOrder
    from tvb.simulator.models.stefanescu_jirsa import (
        ReducedSetFitzHughNagumo,
        ReducedSetHindmarshRose,
    )
    from tvb.simulator.models.cerebellar_mf import CerebellarMF
    from tvb.simulator.models.linear import Linear
    _SUPPORTED_MODELS_CACHE = (
        MontbrioPazoRoxin,
        GastSollaKennedy,
        KIonEx,
        JansenRit,
        Generic2dOscillator,
        ReducedWongWang,
        ReducedWongWangExcInh,
        DecoBalancedExcInh,
        Epileptor,
        Epileptor2D,
        EpileptorCodim3,
        EpileptorCodim3SlowMod,
        EpileptorRestingState,
        WilsonCowan,
        ZerlautAdaptationFirstOrder,  # ZerlautSecondOrder is a subclass
        SupHopf,
        Kuramoto,
        Hopfield,
        LarterBreakspear,
        CoombesByrne2D,
        CoombesByrne,
        GastSchmidtKnosche_SD,
        GastSchmidtKnosche_SF,
        DumontGutkin,
        ZetterbergJansen,
        ReducedSetFitzHughNagumo,
        ReducedSetHindmarshRose,
        CerebellarMF,
        Linear,
    )
    return _SUPPORTED_MODELS_CACHE

# Alias for readability
_get_supported_models = _get_supported_models_classes


# Lazy cache for ReducedSetBase (needed by _check_compatibility)
_REDUCED_SET_BASE_CACHE = None


def _get_reduced_set_base():
    """Lazily import ReducedSetBase and cache it."""
    global _REDUCED_SET_BASE_CACHE
    if _REDUCED_SET_BASE_CACHE is not None:
        return _REDUCED_SET_BASE_CACHE
    from tvb.simulator.models.stefanescu_jirsa import ReducedSetBase
    _REDUCED_SET_BASE_CACHE = ReducedSetBase
    return _REDUCED_SET_BASE_CACHE


def _cfun_params(p: "ProjectionInfo") -> "np.ndarray":
    """Return a float32 array of length 16 with cfun parameters for a ProjectionInfo.

    Layout by cfun type:
      none:                  [1.0, 0, 0, 0, 0, 0, 0, 0, ...]
      linear:                [a, b, 0, 0, 0, 0, 0, 0, ...]
      scaling:               [a, 0, 0, 0, 0, 0, 0, 0, ...]
      sigmoidal:             [a, sigma, midpoint, cmin, cmax, 0, 0, 0, ...]
      sigmoidal_jr:          [a, cmin, cmax, r, midpoint, 0, 0, 0, ...]  (classic)
      sigmoidal_jr_legacy:   [a, e0, r, v0, 0, 0, 0, 0, ...]           (legacy)
      kuramoto:              [a, 1/N, 0, 0, 0, 0, 0, 0, ...]
      difference:            [a, 0, 0, 0, 0, 0, 0, 0, ...]
      tanh:                  [a, b, midpoint, sigma, 0, 0, 0, 0, ...]
      pre_sigmoidal:         [H, Q, G, P, theta, 0, 0, 0, ...]          (static)
      pre_sigmoidal_dynamic: [H, Q, G, P, 0, globalT, 0, 0, ...]        (dynamic)
    """
    from tvb.simulator.hybrid.coupling import (
        Linear,
        Scaling,
        Sigmoidal,
        SigmoidalJansenRit,
        Kuramoto as KuramotoCfun,
        Difference,
        HyperbolicTangent,
        PreSigmoidal,
    )

    _cfun_type(p)
    arr = np.zeros(16, dtype=np.float32)
    arr[0] = 1.0  # default: identity scale
    if p.cfun is None:
        return arr
    if isinstance(p.cfun, Linear):
        arr[0] = float(p.cfun.a[0])
        arr[1] = float(p.cfun.b[0])
        return arr
    if isinstance(p.cfun, Scaling):
        arr[0] = float(p.cfun.a[0])
        return arr
    if isinstance(p.cfun, Difference):
        arr[0] = float(p.cfun.a[0])
        return arr
    if isinstance(p.cfun, Sigmoidal):
        arr[0] = float(p.cfun.a[0])
        arr[1] = float(p.cfun.sigma[0])
        arr[2] = float(p.cfun.midpoint[0])
        arr[3] = float(p.cfun.cmin[0])
        arr[4] = float(p.cfun.cmax[0])
        return arr
    if isinstance(p.cfun, SigmoidalJansenRit):
        if _cfun_type(p) == "sigmoidal_jr":
            # Classic mode: [a, cmin, cmax, r, midpoint]
            arr[0] = float(p.cfun.a[0])
            arr[1] = float(p.cfun.cmin[0])
            arr[2] = float(p.cfun.cmax[0])
            arr[3] = float(p.cfun.r[0])
            arr[4] = float(p.cfun.midpoint[0])
        else:
            # Legacy mode: [a, e0, r, v0]
            arr[0] = float(p.cfun.a[0])
            if getattr(p.cfun, 'use_classic', 1):
                # A classic cfun with one source cvar falls back in pre(), but
                # its post() still applies a second amplitude factor.
                arr[0] *= float(p.cfun.a[0])
            arr[1] = float(p.cfun.e0[0])
            arr[2] = float(p.cfun.r[0])
            arr[3] = float(p.cfun.v0[0])
        return arr
    if isinstance(p.cfun, KuramotoCfun):
        arr[0] = float(p.cfun.a[0])
        # N = number of coupling variables for normalization
        n_cvar = p.source_cvar.shape[0]
        arr[1] = 1.0 / n_cvar if n_cvar > 0 else 1.0
        return arr
    if isinstance(p.cfun, HyperbolicTangent):
        arr[0] = float(p.cfun.a[0])
        arr[1] = float(p.cfun.b[0])
        arr[2] = float(p.cfun.midpoint[0])
        arr[3] = float(p.cfun.sigma[0])
        return arr
    if isinstance(p.cfun, PreSigmoidal):
        arr[0] = float(p.cfun.H[0])
        arr[1] = float(p.cfun.Q[0])
        arr[2] = float(p.cfun.G[0])
        arr[3] = float(p.cfun.P[0])
        if _cfun_type(p) == "pre_sigmoidal_dynamic":
            # Dynamic mode: theta not used, threshold from x_j[1]
            arr[4] = 0.0
            arr[5] = float(bool(getattr(p.cfun, 'globalT', False)))
        else:
            # Static mode: theta
            arr[4] = float(p.cfun.theta[0])
        return arr
    return arr


def _needs_xi(p: "ProjectionInfo") -> bool:
    """Return True if the coupling function needs per-edge target state x_i.

    Difference and Kuramoto need x_i to compute x_j - x_i or sin(x_j - x_i)
    per edge before the weighted sum.
    """
    ct = _cfun_type(p)
    return ct in ("difference", "kuramoto")


def _cvar_mapping_mode(p: "ProjectionInfo") -> str:
    """Determine which cvar-mapping branch to use at code-gen time."""
    ns = p.source_cvar.shape[0]
    nt = p.target_cvar.shape[0]
    if ns == 1 and nt == 1:
        return "1_to_1"
    if nt == 1:
        return "many_to_1"
    if ns == 1:
        return "1_to_many"
    if ns == nt:
        return "n_to_n"
    raise ValueError(
        f"Projection '{p.name}': unsupported cvar mapping ({ns} source → {nt} target)"
    )


# ---------------------------------------------------------------------------
# Data classes for code-generation analysis
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SubnetworkInfo:
    name: str
    model: object  # Model instance
    integrator: object  # Integrator instance
    n_nodes: int
    n_modes: int
    is_stochastic: bool = False
    noise_nsig: Optional[np.ndarray] = None  # shape (n_vars,), only when is_stochastic
    has_stimulus: bool = False
    node_indices: Optional[np.ndarray] = None  # connectome positions, shape (n_nodes,)
    clamp_indices: Optional[np.ndarray] = None  # configured state-variable indices
    clamp_values: Optional[np.ndarray] = None  # shape (n_clamps, n_nodes, n_modes)
    dt: float = 0.0  # integration timestep (ms); multi-dt supported
    k: int = 1  # steps per master tick: dt == k * dt0


@dataclasses.dataclass
class ProjectionInfo:
    name: str
    source_subnet: str
    target_subnet: str
    source_cvar: np.ndarray  # (n_src_cvar,)
    target_cvar: np.ndarray  # (n_tgt_cvar,) coupling-slot indices
    target_state_cvar: np.ndarray  # model.cvar[target_cvar], used only for x_i
    weights_data: np.ndarray  # (nnz,) float32
    weights_indices: np.ndarray  # (nnz,) int
    weights_indptr: np.ndarray  # (n_tgt+1,) int
    idelays: np.ndarray  # (nnz,) int
    horizon: int
    scale: float
    target_scales: np.ndarray  # (n_tgt_cvar,) or empty
    cfun: object  # coupling function or None
    is_inter: bool
    n_tgt_nodes: int
    # mode_map only for inter projections
    mode_map: Optional[np.ndarray] = None  # (n_src_modes, n_tgt_modes)

    @property
    def n_src_modes(self) -> int:
        if self.is_inter:
            return self.mode_map.shape[0]
        # intra: stored in buf — derive from horizon dimension later, but
        # mode_map is None for intra; caller must pass n_modes separately.
        return self._n_src_modes

    @n_src_modes.setter
    def n_src_modes(self, v: int):
        self._n_src_modes = v

    @property
    def n_tgt_modes(self) -> int:
        if self.is_inter:
            return self.mode_map.shape[1]
        return self._n_src_modes  # same for intra


@dataclasses.dataclass
class NetworkAnalysis:
    subnetworks: List[SubnetworkInfo]
    inter_projections: List[ProjectionInfo]
    intra_projections: List[ProjectionInfo]
    # stimuli_by_subnet: dict mapping subnet name -> list of Stim objects
    stimuli_by_subnet: dict = dataclasses.field(default_factory=dict)
    # source_horizons: dict mapping source subnet name -> max horizon across outgoing projections
    source_horizons: dict = dataclasses.field(default_factory=dict)
    # dt0: master-clock base timestep (ms). All subnet dts are integer
    # multiples of dt0 (dt_j = k_j * dt0); the master clock ticks at dt0 and
    # each subnet integrates only on its own ticks. 0.0 means single-dt
    # (backward compatible: consumers fall back to the first subnet's dt).
    dt0: float = 0.0

    @property
    def all_projections(self) -> List[ProjectionInfo]:
        return self.inter_projections + self.intra_projections


# ---------------------------------------------------------------------------
# Stimulus lazy-evaluation threshold
# ---------------------------------------------------------------------------

# When the pre-computed stimulus array would exceed this many megabytes,
# a lazy chunk-by-chunk path should be used instead of pre-allocating the
# full (n_cvar, n_nodes, n_modes, nstep) array.
# Override at runtime via the TVB_HYBRID_LAZY_STIM_MB environment variable.
_STIM_LAZY_THRESHOLD_MB: int = 64

# ---------------------------------------------------------------------------
# Module-level compiled-function cache
# ---------------------------------------------------------------------------

# Keyed by SHA-256 of the rendered (pre-autopep8) source string so the same
# topology produces the same key regardless of which NbHybridBackend instance
# Module-level cache for compiled functions, keyed by SHA-256 of rendered source.
# Stores (run_network_fn, module_object) pairs so the sweep kernel can
# reference network_chunk from the same module.
_COMPILED_FN_CACHE: dict = {}
_COMPILED_MOD_CACHE: dict = {}


def _build_as_module(source: str, cache_key: str):
    """Write *source* to a real .py file so Numba's file-based cache works.

    Numba's ``cache=True`` requires a real ``co_filename`` on the compiled
    function.  By writing the generated source to a file and importing it
    as a proper Python module we give Numba that filename, allowing it to
    persist ``.nbi``/``.nbc`` files in ``__pycache__/`` next to the ``.py``.
    """
    import importlib.util
    import os
    import sys

    cache_dir = NbHybridBackend.get_cache_dir()
    mod_name = f"nbhybrid_{cache_key}"
    mod_path = cache_dir / f"{mod_name}.py"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not mod_path.exists():
            # Atomic write: write to .tmp then os.replace to avoid partial reads.
            tmp_path = mod_path.with_suffix(".tmp")
            tmp_path.write_text(source, encoding="utf-8")
            os.replace(tmp_path, mod_path)
    except OSError as exc:
        raise OSError(
            "Unable to create or write the Numba hybrid cache directory "
            f"at '{cache_dir}' configured by TVB_NHYBRID_CACHE_DIR; "
            "ensure the cache directory is writable."
        ) from exc
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so Numba can find it for cache lookup.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    # Also cache the module object so the sweep kernel can reference network_chunk
    _COMPILED_MOD_CACHE[cache_key] = mod
    return mod.run_network


# ---------------------------------------------------------------------------
# CompiledNetworkFn — holds a compiled kernel + helper to run it
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CompiledNetworkFn:
    """A compiled Numba simulation kernel bound to a specific network topology.

    Obtain via :meth:`NbHybridBackend.compile`.
    Call :meth:`run` to execute the simulation without re-compiling.
    """

    _backend: "NbHybridBackend"
    _analysis: "NetworkAnalysis"
    _run_network_fn: object  # the exec'd Python callable
    _network_set: NetworkSet
    _bold_states: Optional[dict]  # per-monitor/subnetwork HRF monitor state
    _compiled: bool = False  # True after Numba JIT warmup has completed

    def warmup(self) -> float:
        """Eagerly trigger Numba JIT compilation with a single warmup step.

        Call this to front-load the one-time compilation cost (~1-3 s)
        so that subsequent :meth:`run` calls execute at full speed (~5 ms
        per 1 000 steps).  Called automatically by :meth:`NbHybridBackend.compile`
        when ``eager=True`` (the default).

        If compilation was already triggered (by a prior :meth:`run` call
        or a warmup), this is a no-op.

        Returns
        -------
        float
            Wall-clock seconds spent on compilation (0.0 if already compiled).
        """
        if self._compiled:
            return 0.0
        import time

        # For stochastic integrators, the warmup step draws from the noise
        # RNG.  Save and restore the RNG state so that the warmup does not
        # perturb subsequent simulation results.
        rng_states = []
        for sn in self._network_set.subnets:
            sn_info = next(
                si for si in self._analysis.subnetworks if si.name == sn.name
            )
            if sn_info.is_stochastic:
                rng = sn.scheme.noise.random_stream
                rng_states.append((sn, rng, rng.get_state()))

        t0 = time.perf_counter()
        self.run(nstep=1)
        elapsed = time.perf_counter() - t0

        # Restore RNG state so warmup noise draw is invisible to callers.
        for sn, rng, saved_state in rng_states:
            rng.set_state(saved_state)

        self._compiled = True
        # The warmup has no monitors; real runs start with clean monitor state.
        self._bold_states = None
        return elapsed

    def run(
        self,
        nstep: int,
        chunk_size: int = None,
        initial_states: Optional[list] = None,
        return_snapshot: bool = False,
        _initial_buffers: Optional[dict] = None,
        monitors: Optional[list] = None,
        _step_offset: int = 0,
        _rng_states: Optional[list] = None,
        _monitor_states: Optional[dict] = None,
    ) -> list:
        """Execute the pre-compiled kernel for *nstep* integration steps.

        Parameters
        ----------
        nstep : int
            Number of integration steps to run.
        chunk_size : int or None
            Number of steps per temporal-average chunk.  When *None* (default),
            the chunk_size is auto-computed from monitor periods to ensure
            sampling alignment: GCD of all monitor ``istep`` values.  If no
            monitors are provided, defaults to 1 (raw output per step).
        initial_states : list of ndarray, optional
            Initial states per subnetwork.  If *None* the subnetwork's
            ``zero_states()`` are used.
        return_snapshot : bool
            If True, also return a snapshot dict suitable for passing to
            :meth:`resume`.  Default False (backward compatible).
        _initial_buffers : dict, optional
            Pre-populated source history buffers keyed by subnet name, as
            returned in ``snapshot['buffers']`` by a prior :meth:`run` call.
            When supplied the existing buffer is reused instead of being
            re-initialised from the initial state.
        monitors : list of Monitor, optional
            If provided, raw outputs are post-processed per monitor type and
            the return format changes to ``list[list[tuple[times, data]]]``
            (outer list by monitor, inner list by subnetwork).  If *None*
            (default), the original ``(times, data, ctavg)`` format is used.

        Returns
        -------
        list or (list, dict)
            When *return_snapshot* is False (default):
              - If *monitors* is None: list of (times, data, ctavg) per subnetwork.
              - If *monitors* is provided: list per monitor of list per subnetwork
                of (times, data).
            When *return_snapshot* is True: ``(outputs, snapshot)`` where *snapshot*
            contains integration states, history buffers, absolute step, RNG
            positions, and monitor runtime state for :meth:`resume`.
        """
        _reject_balloon_windkessel(monitors)
        # Resolve chunk_size: auto-compute from monitor periods when not specified
        # (multi-dt: monitor periods and the chunk grid live on the master
        # dt0 clock, so istep = round(period / dt0)).
        dt = _validate_multi_dt(self._network_set)
        if chunk_size is None:
            if monitors is not None:
                chunk_size = _compute_chunk_size(monitors, dt)
            else:
                chunk_size = 1

        if monitors is not None:
            from tvb.simulator.monitors import Raw, SubSample

            from tvb.simulator.monitors import AfferentCoupling
            for m in monitors:
                if isinstance(m, Raw) and not isinstance(m, AfferentCoupling) and chunk_size != 1:
                    raise ValueError(
                        "Raw monitor requires chunk_size=1; "
                        "pass chunk_size=1 to run_network()"
                    )
                if isinstance(m, SubSample) and chunk_size != 1:
                    raise ValueError(
                        f"SubSample monitor requires chunk_size=1 "
                        f"(got chunk_size={chunk_size}). "
                        "The step-based selection mask assumes one step per chunk. "
                        "Pass chunk_size=1 or use TemporalAverage instead."
                    )
        execution_chunk_size = chunk_size
        kernel_monitors = monitors
        has_bold = False
        has_temporal_average = False
        if monitors is not None:
            from tvb.simulator.monitors import Bold, TemporalAverage

            has_bold = any(isinstance(m, Bold) for m in monitors)
            has_temporal_average = any(isinstance(m, TemporalAverage) for m in monitors)
            if has_bold or has_temporal_average:
                # Stateful Python monitors consume every observed integration state.
                execution_chunk_size = 1
                kernel_monitors = []
                if self._bold_states is None:
                    self._bold_states = {}
        if _monitor_states is not None:
            self._bold_states = _copy_monitor_states(_monitor_states)

        outputs, final_states, final_bufs, monitor_data = self._backend._run_compiled(
            self._run_network_fn,
            self._analysis,
            self._network_set,
            nstep,
            execution_chunk_size,
            initial_states,
            _initial_buffers=_initial_buffers,
            monitors=kernel_monitors,
            step_offset=_step_offset,
            rng_states=_rng_states,
        )
        if monitors is not None:
            bold_raw_outputs = outputs if has_bold else None
            temporal_raw_outputs = outputs if has_temporal_average else None
            if has_bold or has_temporal_average:
                outputs = _aggregate_raw_outputs(outputs, chunk_size)
            outputs = _apply_monitors(outputs, monitors, dt, chunk_size=chunk_size,
                                      monitor_data=monitor_data,
                                      subnet_infos=self._analysis.subnetworks,
                                      bold_states=self._bold_states,
                                      bold_raw_outputs=bold_raw_outputs,
                                      temporal_raw_outputs=temporal_raw_outputs)
        if not return_snapshot:
            return outputs
        snapshot = {
            "states": [
                final_states[sn.name].copy() for sn in self._analysis.subnetworks
            ],
            "buffers": {name: buf.copy() for name, buf in final_bufs.items()},
            "step": _step_offset + nstep,
            "rng_states": self._backend._get_rng_states(
                self._analysis, self._network_set
            ),
            "monitor_states": _copy_monitor_states(self._bold_states),
        }
        return outputs, snapshot

    def resume(
        self,
        snapshot: dict,
        nstep: int,
        chunk_size: int = None,
        return_snapshot: bool = False,
        monitors: Optional[list] = None,
    ) -> list:
        """Resume a simulation from a snapshot returned by :meth:`run`.

        Parameters
        ----------
        snapshot : dict
            A snapshot dict as returned by ``run(..., return_snapshot=True)``.
            Must contain keys ``'states'`` (list of ndarray per subnetwork) and
            ``'buffers'`` (dict of ndarray per source subnet name).
        nstep : int
            Number of additional integration steps to run.
        chunk_size : int or None
            Steps per temporal-average chunk.  When *None*, defaults to 1.
        return_snapshot : bool
            If True, also return a new snapshot of the final state.
        monitors : list of Monitor, optional
            Monitors to continue. Their runtime stock and sample phase are
            restored from the snapshot by monitor position.

        Returns
        -------
        list or (list, dict)
            Same format as :meth:`run`.
        """
        return self.run(
            nstep,
            chunk_size,
            initial_states=snapshot["states"],
            return_snapshot=return_snapshot,
            _initial_buffers=snapshot["buffers"],
            monitors=monitors,
            _step_offset=snapshot.get("step", 0),
            _rng_states=snapshot.get("rng_states"),
            _monitor_states=snapshot.get("monitor_states"),
        )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class NbHybridBackend(MakoUtilMix):
    """Numba backend for hybrid simulator with multi-subnetwork support.

    Call ``run_network(network_set, nstep)`` to compile and run.  The
    network_set must already be configured (``configure()`` called).

    Supported models: ``MontbrioPazoRoxin``, ``KIonEx`` (and any model that
    provides ``state_variable_dfuns``, ``coupling_terms``, and
    ``global_parameter_names``).  Integrators: Heun/Euler deterministic or
    stochastic.
    """

    @staticmethod
    def get_cache_dir():
        """Return the directory where disk cache files are written."""
        import os
        from pathlib import Path

        configured = os.environ.get("TVB_NHYBRID_CACHE_DIR")
        if configured is not None:
            if not configured:
                raise OSError(
                    "TVB_NHYBRID_CACHE_DIR configures an empty Numba hybrid "
                    "cache directory path; provide a writable directory."
                )
            return Path(configured).expanduser()

        cache_home = os.environ.get("XDG_CACHE_HOME")
        if cache_home:
            return Path(cache_home).expanduser() / "tvb" / "nb_hybrid"
        return Path.home() / ".cache" / "tvb" / "nb_hybrid"

    @classmethod
    def clear_cache(cls):
        """Clear the in-process compiled function cache and the disk cache."""
        import shutil

        _COMPILED_FN_CACHE.clear()
        cache_dir = cls.get_cache_dir()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    def compile(
        self,
        network_set: NetworkSet,
        print_source: bool = False,
        debug_nojit: bool = False,
        eager: bool = True,
    ) -> "CompiledNetworkFn":
        """Compile the simulation kernel for *network_set* and return it.

        The compiled kernel is cached in-process by a SHA-256 hash of the
        generated source.  Repeated calls with topologically identical networks
        return the cached kernel immediately (no re-compilation).

        On first call for a new topology, Numba must JIT-compile the generated
        ``network_chunk`` kernel (~1-3 s depending on model complexity).  When
        *eager* is True (default), this compilation happens here, avoiding a
        surprise latency spike on the first :meth:`CompiledNetworkFn.run` call.
        The compiled kernel is also saved to a user-configurable Numba disk
        cache so subsequent Python processes skip JIT entirely (~5 ms instead
        of ~1.7 s). Set ``TVB_NHYBRID_CACHE_DIR`` to override its location.

        Parameters
        ----------
        network_set : NetworkSet
            Fully configured network (``configure()`` must have been called).
        print_source : bool
            If True, print the generated (autopep8-formatted) source.
        debug_nojit : bool
            If True, disable Numba JIT for debugging (very slow).
        eager : bool
            If True (default), eagerly trigger Numba JIT compilation by running
            a single warmup step.  This front-loads the one-time ~1-3 s
            compilation cost so that subsequent :meth:`run` calls are fast.
            Set to False to defer compilation to the first :meth:`run` call.

        Returns
        -------
        CompiledNetworkFn
            Callable object whose :meth:`~CompiledNetworkFn.run` method
            executes the simulation without recompiling.
        """
        import os

        _use_nojit = debug_nojit or (
            os.environ.get("TVB_HYBRID_NO_JIT", "0")
            not in ("", "0", "false", "False", "no")
        )
        self._check_compatibility(network_set)
        analysis = self._analyse(network_set)
        run_network_fn = self._build(
            '<%include file="nb-hybrid-sim.py.mako"/>',
            dict(analysis=analysis, np=np, debug_nojit=_use_nojit),
            print_source=print_source,
        )
        cn = CompiledNetworkFn(
            _backend=self,
            _analysis=analysis,
            _run_network_fn=run_network_fn,
            _network_set=network_set,
            _bold_states=None,
        )
        if eager and not _use_nojit:
            cn.warmup()
        return cn

    def run_network(
        self,
        network_set: NetworkSet,
        nstep: int,
        chunk_size: int = None,
        print_source: bool = False,
        initial_states: Optional[list] = None,
        debug_nojit: bool = False,
        monitors: Optional[list] = None,
    ):
        """Run a hybrid simulation using the Numba code-generation path.

        Equivalent to ``self.compile(network_set).run(nstep, ...)``.  The
        compiled kernel is cached in-process so repeated calls with the same
        topology do not re-compile.

        Parameters
        ----------
        network_set : NetworkSet
            Fully configured network (``configure()`` must have been called).
        nstep : int
            Number of integration steps to run.
        chunk_size : int or None
            Number of steps per temporal-average chunk.  When *None* (default),
            auto-computed from monitor periods.  See :meth:`CompiledNetworkFn.run`.
        print_source : bool
            If True, print the generated source code with line numbers.
        monitors : list of Monitor, optional
            If provided, raw outputs are post-processed per monitor type and
            the return format changes to ``list[list[tuple[times, data]]]``
            (outer list by monitor, inner list by subnetwork).  If *None*
            (default), the original ``(times, data, ctavg)`` format is used.

        Returns
        -------
        list of (times, data, ctavg) or list[list[tuple[times, data]]]
            One tuple per subnetwork in ``network_set.subnets``, where
            ``times`` is a 1-D float64 array of mid-chunk time points,
            ``data`` is a float32 array of shape ``(n_chunks, n_voi, n_nodes, n_modes)``,
            and ``ctavg`` is a float32 array of shape
            ``(n_chunks, n_cvar, n_nodes, n_modes)`` holding the
            temporally-averaged afferent coupling input to each node.

            When *monitors* is provided the format is instead
            ``list[list[tuple[times, data]]]`` — outer list indexed by monitor,
            inner list by subnetwork.
        """
        _reject_balloon_windkessel(monitors)
        return self.compile(network_set, print_source, debug_nojit=debug_nojit).run(
            nstep, chunk_size, initial_states, monitors=monitors
        )

    def compile_batch(
        self,
        network_sets: list,
        max_workers: Optional[int] = None,
        print_source: bool = False,
        debug_nojit: bool = False,
        eager: bool = True,
    ) -> list:
        """Compile multiple network topologies in parallel.

        Each topology is compiled in a separate process, allowing Numba's
        LLVM compilation to proceed concurrently.  After compilation, each
        process writes its Numba disk cache (``.nbi``/``.nbc`` files) so
        that subsequent calls in the main process hit the disk cache at
        ~5 ms instead of ~1.7 s.

        Parameters
        ----------
        network_sets : list of NetworkSet
            Fully configured networks (``configure()`` must have been called).
        max_workers : int or None
            Maximum number of parallel compilation processes.  Defaults to
            ``min(len(network_sets), os.cpu_count())``.
        print_source : bool
            If True, print the generated source for each topology.
        debug_nojit : bool
            If True, disable Numba JIT for debugging.
        eager : bool
            If True (default), eagerly warm up each compiled kernel in the
            worker process, populating the Numba disk cache.

        Returns
        -------
        list of CompiledNetworkFn
            One per input topology.
        """
        import os
        from concurrent.futures import ProcessPoolExecutor

        n = len(network_sets)
        if max_workers is None:
            max_workers = min(n, os.cpu_count() or 1)

        if n == 1:
            return [self.compile(network_sets[0], print_source=print_source,
                                debug_nojit=debug_nojit, eager=eager)]

        # Compile each topology in a separate process.
        # Each process will write its Numba disk cache independently.
        def _compile_one(ns):
            backend = NbHybridBackend()
            cn = backend.compile(ns, print_source=print_source,
                                debug_nojit=debug_nojit, eager=eager)
            # Return the cache key so the main process can reconstruct the
            # CompiledNetworkFn from its disk cache.
            return ns  # return the network_set so caller can match results

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_compile_one, network_sets))

        # Now re-compile in the main process — this time the in-process and
        # disk caches are warm, so it should be ~5 ms per topology.
        results = []
        for ns in network_sets:
            results.append(self.compile(ns, print_source=print_source,
                                       debug_nojit=debug_nojit, eager=False))
        return results

    # ------------------------------------------------------------------
    # _run_compiled — arg assembly + kernel call (no compilation logic)
    # ------------------------------------------------------------------

    def _run_compiled(
        self,
        run_network_fn,
        analysis: NetworkAnalysis,
        network_set: NetworkSet,
        nstep: int,
        chunk_size: int,
        initial_states: Optional[list],
        _initial_buffers: Optional[dict] = None,
        monitors: Optional[list] = None,
        _bold_states: Optional[dict] = None,
        step_offset: int = 0,
        rng_states: Optional[list] = None,
    ) -> tuple:
        """Build the argument list and call the pre-compiled kernel."""
        # Sanity check replacing the old `chunk_size <= min-horizon` cap.
        #
        # The delayed-coupling history buffer is addressed by the global step
        # modulo the buffer length, and is written *after* it is read within a
        # step (read-before-write).  That invariant makes the simulation
        # correct regardless of how many steps a chunk spans, so a chunk is
        # free to cross a delay horizon (and several chunk boundaries may map
        # onto the same modulo slots).  The cap existed only because the old
        # full-run noise array was indexed by the global step; that array is
        # now generated per chunk, so no cap is needed.
        #
        # What must still hold is the per-projection modulo-wrap invariant
        #     horizon >= max(idelay) + 1
        # (`horizon = max_delay + 1` by construction).  Violating it would
        # alias two distinct delays onto the same buffer slot and silently
        # corrupt the coupling, so we keep a loud failure here.
        for p in analysis.all_projections:
            idelays = np.atleast_1d(p.idelays)
            if idelays.size:
                max_idelay = int(np.max(idelays))
                if p.horizon < max_idelay + 1:
                    raise ValueError(
                        f"Projection '{p.name}' has delay-buffer horizon "
                        f"{p.horizon} < max(idelay)+1 = {max_idelay + 1}; the "
                        "circular history buffer would alias distinct delays "
                        "onto the same slot. Increase tract lengths / reduce dt."
                    )
        # Build argument list matching the generated run_network() signature
        args = [nstep, step_offset]

        if rng_states is not None:
            stochastic_index = 0
            for sn_info in analysis.subnetworks:
                if sn_info.is_stochastic:
                    sn_obj = next(
                        s for s in network_set.subnets if s.name == sn_info.name
                    )
                    sn_obj.scheme.noise.random_stream.set_state(
                        rng_states[stochastic_index]
                    )
                    stochastic_index += 1

        # Per-subnetwork initial states (from provided list or zero)
        sn_states = {}
        for i, sn_info in enumerate(analysis.subnetworks):
            sn_obj = next(s for s in network_set.subnets if s.name == sn_info.name)
            if initial_states is not None:
                state = initial_states[i].astype(np.float32)
            else:
                state = sn_obj.zero_states().astype(np.float32)
            sn_states[sn_info.name] = state
            args.append(state)

        # Per-source-subnet shared history buffers (one buffer per source subnet)
        src_bufs = {}
        for sn_info in analysis.subnetworks:
            if _initial_buffers is not None and sn_info.name in _initial_buffers:
                buf = _initial_buffers[sn_info.name].astype(np.float32)
            else:
                horizon = analysis.source_horizons.get(sn_info.name, 1)
                state = sn_states[sn_info.name]
                n_vars, n_nodes, n_modes = state.shape
                buf = np.empty((n_vars, n_nodes, n_modes, horizon), dtype=np.float32)
                buf[:] = state[
                    :, :, :, np.newaxis
                ]  # broadcast ICs across all horizon slots
            src_bufs[sn_info.name] = buf
        for sn_info in analysis.subnetworks:
            args.append(src_bufs[sn_info.name])

        # Per-projection arrays
        for p in analysis.all_projections:
            args.append(p.weights_data.astype(np.float32))
            args.append(p.weights_indices.astype(np.int32))
            args.append(p.weights_indptr.astype(np.int32))
            args.append(p.idelays.astype(np.int32))
            if p.is_inter:
                args.append(p.mode_map.astype(np.float32))
            args.append(p.source_cvar.astype(np.int32))
            args.append(p.target_cvar.astype(np.int32))
            args.append(p.target_state_cvar.astype(np.int32))
            args.append(np.float32(p.scale))
            ts = (
                p.target_scales.astype(np.float32)
                if p.target_scales.size > 0
                else np.zeros(0, dtype=np.float32)
            )
            args.append(ts)
            cfun_params = _cfun_params(p)
            args.append(cfun_params)

        # Per-subnetwork noise sources (stochastic integrators).
        #
        # Previously an (nstep, n_vars, n_nodes, n_modes) array was drawn here,
        # scaled by noise_std, transposed and cast to float32 — an O(nstep)
        # allocation that OOMs for long stochastic runs (and is worse still in
        # the sweep backend).  Instead we hand the kernel the RNG stream and
        # the per-variable noise scale; the run_network template draws a
        # chunk-sized array inside the outer chunk loop, so peak allocation is
        # O(chunk_size).
        #
        # Draw order is preserved across chunk sizes: each chunk draws
        # (this_chunk, n_vars, n_nodes, n_modes) sequentially, in the same
        # element order as the old single draw, so a given seed yields the same
        # trajectory regardless of chunk_size (numpy's legacy RandomState draws
        # elementwise and keeps its gauss cache in the bit-generator state, so
        # splitting the draw across calls does not change the stream).
        for sn_info in analysis.subnetworks:
            if sn_info.is_stochastic:
                sn_obj = next(s for s in network_set.subnets if s.name == sn_info.name)
                dt = sn_obj.scheme.dt
                rng = sn_obj.scheme.noise.random_stream
                noise_std = np.sqrt(2.0 * sn_info.noise_nsig * dt)  # (n_vars,) float64
                args.append(rng)
                args.append(noise_std)

        # Per-subnetwork stimulus arrays (pre-computed batch)
        # TODO §8.4: use lazy chunk-by-chunk path when estimated stim_arr_mb
        #   exceeds _STIM_LAZY_THRESHOLD_MB (or TVB_HYBRID_LAZY_STIM_MB env var).
        #   See _compute_stimulus_lazy() for the planned implementation.
        # Multi-dt decision 7: the pattern is evaluated on the master dt0
        # grid (index = global master step), not the subnet's own dt grid.
        stim_dt0 = (analysis.dt0 if analysis.dt0
                    else float(network_set.subnets[0].scheme.dt))
        for sn_info in analysis.subnetworks:
            if sn_info.has_stimulus:
                n_cvar = len(sn_info.model.cvar)
                stim_arr = np.zeros(
                    (n_cvar, sn_info.n_nodes, sn_info.n_modes, nstep),
                    dtype=np.float32,
                )
                for stim in (_stim_master_grid(s, stim_dt0, step_offset + nstep)
                             for s in
                             analysis.stimuli_by_subnet[sn_info.name]):
                    target_slots = np.asarray(stim.target_cvar)
                    if target_slots.ndim != 1 or target_slots.size == 0:
                        raise ValueError(
                            f"Stimulus for subnetwork '{sn_info.name}' must have a "
                            "non-empty one-dimensional target_cvar array of coupling slots; "
                            f"got shape {target_slots.shape}."
                        )
                    if target_slots.dtype.kind not in "iu":
                        raise ValueError(
                            f"Stimulus for subnetwork '{sn_info.name}' has non-integer "
                            f"target_cvar values {target_slots.tolist()}."
                        )
                    target_slots = target_slots.astype(np.intp, copy=False)
                    if np.any(target_slots < 0) or np.any(target_slots >= n_cvar):
                        raise ValueError(
                            f"Stimulus for subnetwork '{sn_info.name}' has target coupling "
                            f"slots {target_slots.tolist()} outside [0, {n_cvar - 1}]."
                        )
                    target_shape = (
                        target_slots.size,
                        sn_info.n_nodes,
                        sn_info.n_modes,
                    )
                    for step_idx in range(step_offset + 1, step_offset + nstep + 1):
                        sc = np.asarray(stim.get_coupling(step_idx), dtype=np.float32)
                        if sc.ndim == 2:
                            sc = sc[:, :, np.newaxis]
                        if sc.ndim != 3 or any(
                            actual not in (1, expected)
                            for actual, expected in zip(sc.shape, target_shape)
                        ):
                            raise ValueError(
                                f"Stimulus for subnetwork '{sn_info.name}' targeting coupling "
                                f"slots {target_slots.tolist()} returned shape {sc.shape} at "
                                f"step {step_idx}; expected a shape broadcastable to "
                                f"{target_shape} as (target_cvar, nodes, modes)."
                            )
                        sc = np.broadcast_to(sc, target_shape)
                        # target_cvar contains coupling-slot indices, not state indices.
                        stim_arr[target_slots, :, :, step_idx - step_offset - 1] += sc
                args.append(stim_arr)

        # Per-subnetwork spatial parameter arrays (heterogeneous per-node parameters)
        for sn_info in analysis.subnetworks:
            if hasattr(sn_info.model, '_nb_hybrid_runtime_parameter_names'):
                sp_names = list(sn_info.model._nb_hybrid_runtime_parameter_names)
            elif hasattr(sn_info.model, '_nb_hybrid_custom_template'):
                sp_names = []
            else:
                sp_names = list(getattr(sn_info.model, 'spatial_parameter_names', []))
            if sp_names:
                sp_arr = np.array(
                    [np.broadcast_to(np.asarray(getattr(sn_info.model, n)).ravel(),
                                     (sn_info.n_nodes,))
                     for n in sp_names],
                    dtype=np.float32,
                )
            else:
                sp_arr = np.zeros((0, sn_info.n_nodes), dtype=np.float32)
            args.append(sp_arr)

        # Per-subnetwork monitor config arrays (SpatialAverage / Projection)
        from tvb.simulator.monitors import SpatialAverage as SpatialAverageMon, Projection as ProjectionMon, Bold as BoldMon
        _spatial_mean_mon = None
        _projection_mon = None
        _bold_mon = None
        if monitors:
            for m in monitors:
                if isinstance(m, SpatialAverageMon) and hasattr(m, 'spatial_mean'):
                    _spatial_mean_mon = m
                if isinstance(m, ProjectionMon):
                    _projection_mon = m
                if isinstance(m, BoldMon):
                    _bold_mon = m
        for sn_info in analysis.subnetworks:
            # spatial_mean: (n_areas, n_nodes) or empty (0, n_nodes)
            if _spatial_mean_mon is not None:
                sm = np.asarray(_spatial_mean_mon.spatial_mean, dtype=np.float32)
                if sm.shape[1] != sn_info.n_nodes:
                    sm = np.zeros((0, sn_info.n_nodes), dtype=np.float32)
            else:
                sm = np.zeros((0, sn_info.n_nodes), dtype=np.float32)
            args.append(sm)
            # gain: (n_sensors, n_nodes) or empty (0, n_nodes)
            if _projection_mon is not None and hasattr(_projection_mon, 'gain'):
                gn = np.asarray(_projection_mon.gain, dtype=np.float32)
                if gn.shape[1] != sn_info.n_nodes:
                    gn = np.zeros((0, sn_info.n_nodes), dtype=np.float32)
            else:
                gn = np.zeros((0, sn_info.n_nodes), dtype=np.float32)
            args.append(gn)

        # Per-subnetwork Bold Balloon model arrays
        # Bold parameters: [rtau_s, rtau_f, rtau_o, ra, e0, re0, k1, k2, k3]
        # Default values from vbjax/compute_bold_theta()
        _bold_params = None
        _bold_istep = 0
        _bold_v0 = np.float32(0.0)
        _bold_dt = np.float32(0.0)
        if _bold_mon is not None:
            dt = _validate_multi_dt(network_set)
            _bold_dt = np.float32(dt)
            # Extract Bold period in steps (master dt0 grid)
            bold_period = float(_bold_mon.period)  # ms
            _bold_istep = max(1, int(round(bold_period / dt)))
            # Compute Balloon model parameters
            tau_s = np.float32(0.65)
            tau_f = np.float32(0.41)
            tau_o = np.float32(0.98)
            alpha = np.float32(0.32)
            te = np.float32(0.04)
            e0 = np.float32(0.4)
            epsilon = np.float32(0.5)
            nu_0 = np.float32(40.3)
            r_0 = np.float32(25.0)
            v0 = np.float32(4.0)
            _bold_v0 = v0
            k1 = np.float32(4.3) * nu_0 * e0 * te
            k2 = epsilon * r_0 * e0 * te
            k3 = np.float32(1.0) - epsilon
            _bold_params = np.array([
                np.float32(1.0) / tau_s,  # rtau_s
                np.float32(1.0) / tau_f,  # rtau_f
                np.float32(1.0) / tau_o,  # rtau_o
                np.float32(1.0) / alpha,  # ra
                e0,
                np.float32(1.0) / e0,    # re0
                k1, k2, k3,
            ], dtype=np.float32)

        _bold_state_arrays = {}  # name -> bold_state array (for persistence)

        for sn_info in analysis.subnetworks:
            n_voi = len(sn_info.model.variables_of_interest)
            n_nodes = sn_info.n_nodes
            svars = list(sn_info.model.state_variables)
            voi = list(sn_info.model.variables_of_interest)
            voi_idx = [svars.index(v) if v in svars else 0 for v in voi]
            if _bold_mon is not None:
                # Reuse existing Bold state if available (persistence across calls)
                if _bold_states is not None and sn_info.name in _bold_states:
                    bold_state = _bold_states[sn_info.name]
                else:
                    bold_state = np.zeros((n_voi, 4, n_nodes), dtype=np.float32)
                    # Initial conditions: s=0, f=1, v=1, q=1
                    bold_state[:, 1, :] = 1.0
                    bold_state[:, 2, :] = 1.0
                    bold_state[:, 3, :] = 1.0
                bold_params = _bold_params
                bold_voi_idx = np.array(voi_idx, dtype=np.int32)
            else:
                bold_state = np.zeros((0, 4, 0), dtype=np.float32)
                bold_params = np.zeros(9, dtype=np.float32)
                bold_voi_idx = np.zeros(0, dtype=np.int32)
            args.append(bold_state)
            args.append(bold_params)
            args.append(bold_voi_idx)
            # Track Bold state arrays for persistence
            if _bold_mon is not None:
                _bold_state_arrays[sn_info.name] = bold_state

        args.append(_bold_dt)
        args.append(np.int32(_bold_istep))
        args.append(_bold_v0)
        args.append(chunk_size)

        outputs = run_network_fn(*args)
        # outputs[i] = (times, data, ctavg, spatial, proj, bold_times, bold_data)
        raw_outputs = [(t, d, c) for t, d, c, s, p, bt, bd in outputs]
        # Bold state arrays are mutated in-place, so _bold_state_arrays now has updated values
        bold_states_dict = _bold_state_arrays if _bold_state_arrays else None
        monitor_data = {
            'spatial': [s for t, d, c, s, p, bt, bd in outputs],
            'proj': [p for t, d, c, s, p, bt, bd in outputs],
            'bold': [(bt, bd) for t, d, c, s, p, bt, bd in outputs],
            'bold_states': bold_states_dict,
        }
        return raw_outputs, sn_states, src_bufs, monitor_data

    @staticmethod
    def _get_rng_states(analysis, network_set):
        """Return stochastic stream positions in subnetwork order."""
        states = []
        for sn_info in analysis.subnetworks:
            if sn_info.is_stochastic:
                sn_obj = next(
                    s for s in network_set.subnets if s.name == sn_info.name
                )
                states.append(sn_obj.scheme.noise.random_stream.get_state())
        return states

    @staticmethod
    def _compute_stimulus_lazy(
        analysis: "NetworkAnalysis",
        sn_info: "SubnetworkInfo",
        step_start: int,
        step_end: int,
    ) -> np.ndarray:
        """Compute stimulus for steps *step_start*..*step_end* (inclusive).

        Returns a ``(n_cvar, n_nodes, n_modes, window_size)`` float32 array
        whose last axis spans the requested step range.

        This is a stub for the planned lazy chunk-by-chunk stimulus path
        (§8.4).  The intent is that ``_run_compiled`` will call this per
        chunk instead of pre-allocating the full ``(…, nstep)`` array, so
        that peak RSS stays bounded by ``chunk_size`` rather than ``nstep``.

        TODO §8.4: Wire this into ``_run_compiled``:
          - Check ``_stim_estimate_mb(analysis, sn_info, nstep) > threshold``
          - If so, expose ``network_chunk`` from the generated module, replicate
            the ``run_network`` outer loop in Python, and call this method
            per chunk to build a ``(…, this_chunk)`` stim window.
          - Requires template change: stim indexing in ``network_chunk`` must
            use the local offset ``t_local`` rather than the global ``t - 1``
            so that chunk-sized stim views can be passed without out-of-bounds
            access on the second and subsequent chunks.
        """
        n_cvar = len(sn_info.model.cvar)
        window_size = step_end - step_start + 1
        stim_arr = np.zeros(
            (n_cvar, sn_info.n_nodes, sn_info.n_modes, window_size),
            dtype=np.float32,
        )
        for stim in analysis.stimuli_by_subnet[sn_info.name]:
            for local_idx, step_idx in enumerate(range(step_start, step_end + 1)):
                sc = np.asarray(stim.get_coupling(step_idx), dtype=np.float32)
                if sc.ndim == 2:
                    sc = sc[:, :, np.newaxis]
                if sc.shape[2] == 1 and sn_info.n_modes > 1:
                    sc = np.broadcast_to(
                        sc, (sc.shape[0], sn_info.n_nodes, sn_info.n_modes)
                    ).copy()
                stim_arr[:, :, :, local_idx] += sc
        return stim_arr

    @staticmethod
    def _stim_estimate_mb(
        sn_info: "SubnetworkInfo",
        nstep: int,
    ) -> float:
        """Estimate the memory (in MiB) that the pre-computed stim array would use."""
        import os

        n_cvar = len(sn_info.model.cvar)
        n_bytes = n_cvar * sn_info.n_nodes * sn_info.n_modes * nstep * 4  # float32
        return n_bytes / (1024 * 1024)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_compatibility(self, network_set: NetworkSet) -> None:
        # Supported models — lazy import because loading 27 model modules
        # costs ~2.5 s on first call.  Caching the tuple avoids repeated imports.
        _supported_models = _get_supported_models()
        _allowed_integrators = (
            HeunDeterministic,
            EulerDeterministic,
            HeunStochastic,
            EulerStochastic,
        )
        dt0 = _validate_multi_dt(network_set)
        for projection in network_set.projections:
            _cfun_type(projection)
        for sn in network_set.subnets:
            for projection in sn.projections:
                _cfun_type(projection)
        for sn in network_set.subnets:
            model_local_coupling = getattr(sn.scheme, "model_local_coupling", 0.0)
            if np.any(np.asarray(model_local_coupling) != 0.0):
                raise NotImplementedError(
                    "NbHybridBackend does not support nonzero local_coupling; "
                    "the generated model dynamics currently assume local_coupling=0."
                )
            if not isinstance(sn.model, _supported_models):
                raise NotImplementedError(
                    f"NbHybridBackend does not support {type(sn.model).__name__}. "
                    f"Supported: MontbrioPazoRoxin, KIonEx, JansenRit, ZetterbergJansen, "
                    f"Generic2dOscillator, SupHopf, Kuramoto, Hopfield, LarterBreakspear, "
                    f"ReducedWongWang, ReducedWongWangExcInh, DecoBalancedExcInh, "
                    f"Epileptor, Epileptor2D, EpileptorCodim3, EpileptorCodim3SlowMod, EpileptorRestingState, "
                    f"WilsonCowan, ZerlautAdaptation*, "
                    f"CoombesByrne2D, CoombesByrne, GastSchmidtKnosche_SD/SF, DumontGutkin, "
                    f"GastSollaKennedy, "
                    f"ReducedSetFitzHughNagumo, ReducedSetHindmarshRose, Linear."
                )
            if not isinstance(sn.scheme, _allowed_integrators):
                raise NotImplementedError(
                    f"NbHybridBackend only supports Heun/EulerDeterministic or Stochastic; "
                    f"subnetwork '{sn.name}' uses {type(sn.scheme).__name__}"
                )
            # Model-specific validation (look up classes from lazy cache)
            _model_cls_by_name = {m.__name__: m for m in _supported_models}
            Epileptor_ = _model_cls_by_name.get('Epileptor')
            Epileptor2D_ = _model_cls_by_name.get('Epileptor2D')
            EpileptorRestingState_ = _model_cls_by_name.get('EpileptorRestingState')
            WilsonCowan_ = _model_cls_by_name.get('WilsonCowan')
            Hopfield_ = _model_cls_by_name.get('Hopfield')
            if Epileptor_ and isinstance(sn.model, Epileptor_):
                if sn.model.modification[0]:
                    raise NotImplementedError(
                        "NbHybridBackend: Epileptor with modification=True is not supported. "
                        "Set model.modification = numpy.array([False])."
                    )
            if Epileptor2D_ and isinstance(sn.model, Epileptor2D_):
                if sn.model.modification[0]:
                    raise NotImplementedError(
                        "NbHybridBackend: Epileptor2D with modification=True is not supported. "
                        "The dfun_intermediates for 'h' only encodes the non-modification case. "
                        "Set model.modification = numpy.array([False])."
                    )
            if EpileptorRestingState_ and isinstance(sn.model, EpileptorRestingState_):
                if sn.model.modification[0]:
                    raise NotImplementedError(
                        "NbHybridBackend: EpileptorRestingState with modification=True is not supported. "
                        "Set model.modification = numpy.array([False])."
                    )
            if Hopfield_ and isinstance(sn.model, Hopfield_):
                if sn.model.dynamic[0]:
                    raise NotImplementedError(
                        "NbHybridBackend: Hopfield with dynamic=True is not supported. "
                        "The hybrid codegen path only supports static threshold (dynamic=False). "
                        "Set model.dynamic = numpy.array([False])."
                    )
            if WilsonCowan_ and isinstance(sn.model, WilsonCowan_):
                if not sn.model.shift_sigmoid[0]:
                    raise NotImplementedError(
                        "NbHybridBackend: WilsonCowan with shift_sigmoid=False is not supported. "
                        "Use the default shift_sigmoid=True."
                    )
        _ReducedSetBase = _get_reduced_set_base()
        for sn in network_set.subnets:
            if isinstance(sn.model, _ReducedSetBase):
                if (
                    not hasattr(sn.model, "dfun_mode")
                    or sn.model.dfun_mode != "combined"
                ):
                    raise NotImplementedError(
                        f"{type(sn.model).__name__} requires dfun_mode='combined' attribute. "
                        "Run update_derived_parameters() / configure() before using the backend."
                    )

    def _analyse(self, network_set: NetworkSet) -> "NetworkAnalysis":
        from tvb.simulator.noise import Additive

        dt0 = _validate_multi_dt(network_set)

        # Build stimulus lookup: subnet name -> list of Stim objects
        stims_by_subnet: dict = {sn.name: [] for sn in network_set.subnets}
        for sn in network_set.subnets:
            for stim in (sn.stimuli or []):
                stims_by_subnet[sn.name].append(stim)

        subnets = []
        for sn in network_set.subnets:
            is_stoch = isinstance(sn.scheme, (EulerStochastic, HeunStochastic))
            noise_nsig = None
            if is_stoch:
                noise_obj = sn.scheme.noise
                if isinstance(noise_obj, Additive):
                    nsig = noise_obj.nsig
                    if nsig.ndim == 0:
                        noise_nsig = np.full(
                            sn.model.nvar, float(nsig), dtype=np.float64
                        )
                    else:
                        noise_nsig = (
                            np.broadcast_to(nsig, (sn.model.nvar,))
                            .copy()
                            .astype(np.float64)
                        )
                else:
                    raise NotImplementedError(
                        f"Subnetwork '{sn.name}': only tvb.simulator.noise.Additive is supported "
                        f"by the Numba backend; got {type(noise_obj).__name__}. "
                        "Use HeunDeterministic or EulerDeterministic for deterministic integration."
                    )
            clamp_indices = None
            clamp_values = None
            if sn.scheme.clamped_state_variable_values is not None:
                clamp_indices = np.asarray(
                    sn.scheme.clamped_state_variable_indices, dtype=np.int32
                ).ravel()
                values = np.asarray(
                    sn.scheme.clamped_state_variable_values, dtype=np.float32
                )
                clamp_values = np.broadcast_to(
                    values,
                    (len(clamp_indices), sn.nnodes, sn.model.number_of_modes),
                ).copy()
            subnets.append(
                SubnetworkInfo(
                    name=sn.name,
                    model=sn.model,
                    integrator=sn.scheme,
                    n_nodes=sn.nnodes,
                    n_modes=sn.model.number_of_modes,
                    is_stochastic=is_stoch,
                    noise_nsig=noise_nsig,
                    has_stimulus=bool(stims_by_subnet[sn.name]),
                    node_indices=getattr(sn, 'node_indices', None),
                    clamp_indices=clamp_indices,
                    clamp_values=clamp_values,
                    dt=float(sn.scheme.dt),
                    k=max(1, int(round(float(sn.scheme.dt) / dt0))),
                )
            )

        inter_projs = []
        for p in network_set.projections:
            if isinstance(p, IntraProjection):
                continue
            inter_projs.append(self._build_projection_info(p, is_inter=True))

        intra_projs = []
        for sn_obj in network_set.subnets:
            for p in sn_obj.projections:
                pi = self._build_projection_info(p, is_inter=False)
                # For intra, source and target are the same subnetwork
                pi.source_subnet = sn_obj.name
                pi.target_subnet = sn_obj.name
                pi.n_src_modes = sn_obj.model.number_of_modes
                target_model_cvar = np.asarray(sn_obj.model.cvar, dtype=np.int32)
                if np.any(pi.target_cvar < 0) or np.any(pi.target_cvar >= len(target_model_cvar)):
                    raise ValueError(
                        f"Projection '{pi.name}' has target coupling slots "
                        f"{pi.target_cvar.tolist()} outside "
                        f"[0, {len(target_model_cvar) - 1}]"
                    )
                pi.target_state_cvar = target_model_cvar[pi.target_cvar]
                intra_projs.append(pi)

        # Assign unique names to avoid collisions
        all_names = {}
        for p in inter_projs + intra_projs:
            base = p.name
            if base in all_names:
                all_names[base] += 1
                p.name = f"{base}_{all_names[base]}"
            else:
                all_names[base] = 0

        # Compute per-source-subnet max horizon for shared history buffers
        _all_projs = inter_projs + intra_projs
        source_horizons: dict = {}
        for _p in inter_projs:
            source_horizons[_p.source_subnet] = max(
                source_horizons.get(_p.source_subnet, 1), _p.horizon)
        for _p in intra_projs:
            # intra reads reach one step deeper right after a push (the
            # exact read i0 = t//k - 1 - delay at a tick where the source
            # has m pushed steps gives i0 = m - 1 - delay, and the slot
            # would alias step m when H == delay + 1): keep one extra
            # IC-prefilled slot so the negative range never aliases
            # (decisions 3/4)
            source_horizons[_p.source_subnet] = max(
                source_horizons.get(_p.source_subnet, 1), _p.horizon + 1)
        # Ensure every subnetwork has an entry (default 1 for subnets with no outgoing projections)
        for sn in subnets:
            if sn.name not in source_horizons:
                source_horizons[sn.name] = 1

        return NetworkAnalysis(
            subnetworks=subnets,
            inter_projections=inter_projs,
            intra_projections=intra_projs,
            stimuli_by_subnet=stims_by_subnet,
            source_horizons=source_horizons,
            dt0=dt0,
        )

    def _build_projection_info(self, p, is_inter: bool) -> "ProjectionInfo":
        _cfun_type(p)
        ts = (
            p.target_scales
            if p.target_scales is not None
            else np.zeros(0, dtype=np.float64)
        )

        if is_inter:
            src_name = p.source.name
            tgt_name = p.target.name
            n_src_modes = p.source.model.number_of_modes
            n_tgt_modes = p.target.model.number_of_modes
            if p.mode_map is not None:
                mode_map = p.mode_map.astype(np.float32)
            else:
                mode_map = np.ones((n_src_modes, n_tgt_modes), dtype=np.float32)
            proj_name = f"{src_name}_to_{tgt_name}"
        else:
            src_name = ""  # filled by caller
            tgt_name = ""
            n_src_modes = 1  # will be filled
            mode_map = None
            proj_name = getattr(p, "name", None) or "intra"

        # Strip structural zeros from a copy so the original projection is not mutated.
        # p.idelays is positionally aligned with p.weights.data, so we apply the same mask.
        weights_csr = p.weights.copy()
        idelays_raw = np.atleast_1d(p.idelays)
        nz_mask = weights_csr.data != 0
        weights_csr.eliminate_zeros()
        idelays_stripped = idelays_raw[nz_mask].astype(np.int32)
        if is_inter:
            n_tgt_nodes = p.target.nnodes
        else:
            n_tgt_nodes = p.weights.shape[1]  # number of target nodes in intra-projection
        target_cvar_arr = np.atleast_1d(p.target_cvar).astype(np.int32)
        if is_inter:
            target_model_cvar = np.asarray(p.target.model.cvar, dtype=np.int32)
            if np.any(target_cvar_arr < 0) or np.any(target_cvar_arr >= len(target_model_cvar)):
                raise ValueError(
                    f"Projection '{proj_name}' has target coupling slots "
                    f"{target_cvar_arr.tolist()} outside "
                    f"[0, {len(target_model_cvar) - 1}]"
                )
            target_state_cvar = target_model_cvar[target_cvar_arr]
        else:
            target_state_cvar = np.zeros(len(target_cvar_arr), dtype=np.int32)
        pi = ProjectionInfo(
            name=proj_name,
            source_subnet=src_name,
            target_subnet=tgt_name,
            source_cvar=np.atleast_1d(p.source_cvar).astype(np.int32),
            target_cvar=target_cvar_arr,
            target_state_cvar=target_state_cvar,
            weights_data=weights_csr.data.astype(np.float32),
            weights_indices=weights_csr.indices.astype(np.int32),
            weights_indptr=weights_csr.indptr.astype(np.int32),
            idelays=idelays_stripped,
            horizon=int(p._horizon),
            scale=float(p.scale),
            target_scales=np.atleast_1d(ts).astype(np.float32)
            if ts.size > 0
            else np.zeros(0, dtype=np.float32),
            cfun=p.cfun,
            is_inter=is_inter,
            n_tgt_nodes=n_tgt_nodes,
            mode_map=mode_map,
        )
        if not is_inter:
            pi.n_src_modes = n_src_modes  # placeholder; filled per-subnetwork
        return pi

    def _make_projection_buffer(
        self,
        p: ProjectionInfo,
        sn_states: dict,
        network_set: NetworkSet,
    ) -> np.ndarray:
        """Allocate and pre-fill the circular history buffer for a projection."""
        if p.is_inter:
            src_state = sn_states[p.source_subnet]
        else:
            src_state = sn_states[p.target_subnet]  # intra: same subnetwork

        n_vars, n_nodes, n_modes = src_state.shape
        buf = np.zeros((n_vars, n_nodes, n_modes, p.horizon), dtype=np.float32)
        # Pre-fill all slots with the initial state (matching init_projection_buffers)
        for slot in range(p.horizon):
            buf[:, :, :, slot] = src_state
        return buf

    def _build(self, template_source: str, content: dict, print_source: bool = False):
        """Render and exec the template; return the run_network callable.

        The compiled callable is cached in ``_COMPILED_FN_CACHE`` keyed by the
        SHA-256 of the rendered source so that repeated calls with the same
        network topology skip template rendering, ``exec()``, and Numba JIT.
        ``autopep8`` is applied only when *print_source* is True.
        """
        source = self.render_template(template_source, content)
        cache_key = hashlib.sha256(source.encode()).hexdigest()
        if cache_key in _COMPILED_FN_CACHE:
            if print_source:
                formatted = autopep8.fix_code(source)
                print(self.insert_line_numbers(formatted))
            return _COMPILED_FN_CACHE[cache_key]

        if print_source:
            formatted = autopep8.fix_code(source)
            print(self.insert_line_numbers(formatted))

        try:
            fn = _build_as_module(source, cache_key)
        except Exception as exc:
            print(self.insert_line_numbers(autopep8.fix_code(source)))
            raise exc
        _COMPILED_FN_CACHE[cache_key] = fn
        return fn

    # ----- Helper: get/set cfun parameter by index -----

    @staticmethod
    def _cfun_get_param(cfun, pidx):
        """Get the pidx-th parameter of a cfun object.

        Maps param_idx (as used in sweep_descriptor) to the named attribute
        of each coupling function type.  Mirrors the _cfun_params() layout:

          linear:                [a, b]
          scaling:               [a]
          sigmoidal:             [a, sigma, midpoint, cmin, cmax]
          sigmoidal_jr:          [a, cmin, cmax, r, midpoint]  (classic)
          sigmoidal_jr_legacy:   [a, e0, r, v0]                 (legacy)
          kuramoto:              [a, 1/N]
          difference:             [a]
          tanh:                  [a, b, midpoint, sigma]
          pre_sigmoidal:         [H, Q, G, P, theta]
          pre_sigmoidal_dynamic: [H, Q, G, P, 0, globalT]
        """
        from tvb.simulator.hybrid.coupling import (
            Linear, Scaling, Sigmoidal, SigmoidalJansenRit,
            Kuramoto as KuramotoCfun, Difference,
            HyperbolicTangent, PreSigmoidal,
        )
        if cfun is None:
            if pidx == 0:
                return 1.0
            raise IndexError(f"No parameter index {pidx} for None cfun")
        if isinstance(cfun, Linear):
            return [float(cfun.a[0]), float(cfun.b[0])][pidx]
        elif isinstance(cfun, Scaling):
            if pidx == 0:
                return float(cfun.a[0])
            raise IndexError(f"Scaling has only 1 parameter (index 0), got {pidx}")
        elif isinstance(cfun, Sigmoidal):
            return [float(cfun.a[0]), float(cfun.sigma[0]),
                    float(cfun.midpoint[0]), float(cfun.cmin[0]),
                    float(cfun.cmax[0])][pidx]
        elif isinstance(cfun, SigmoidalJansenRit):
            # Legacy indices 0-3 preserved for backward compat with saved sweeps.
            # Classic indices 4-6 are new.
            if pidx <= 3:
                return [float(cfun.a[0]), float(cfun.e0[0]),
                        float(cfun.r[0]), float(cfun.v0[0])][pidx]
            elif pidx <= 6:
                return [float(cfun.cmin[0]), float(cfun.cmax[0]),
                        float(cfun.midpoint[0])][pidx - 4]
            raise IndexError(
                f"SigmoidalJansenRit has 7 sweepable parameters (0-6), got {pidx}"
            )
        elif isinstance(cfun, KuramotoCfun):
            if pidx == 0:
                return float(cfun.a[0])
            raise IndexError(f"Kuramoto: only param_idx=0 is sweepable, got {pidx}")
        elif isinstance(cfun, Difference):
            if pidx == 0:
                return float(cfun.a[0])
            raise IndexError(f"Difference has only 1 sweepable parameter, got {pidx}")
        elif isinstance(cfun, HyperbolicTangent):
            # Indices 0-2 preserved for backward compat with saved sweeps;
            # b is new at index 3.
            if pidx <= 2:
                return [float(cfun.a[0]), float(cfun.midpoint[0]),
                        float(cfun.sigma[0])][pidx]
            elif pidx == 3:
                return float(cfun.b[0])
            raise IndexError(
                f"HyperbolicTangent has 4 sweepable parameters (0-3), got {pidx}"
            )
        elif isinstance(cfun, PreSigmoidal):
            return [float(cfun.H[0]), float(cfun.Q[0]), float(cfun.G[0]),
                    float(cfun.P[0]), float(cfun.theta[0])][pidx]
        else:
            raise TypeError(f"Unknown cfun type: {type(cfun).__name__}")

    @staticmethod
    def _cfun_set_param(cfun, pidx, value):
        """Set the pidx-th parameter of a cfun object.

        Sets the named attribute corresponding to param_idx.
        """
        from tvb.simulator.hybrid.coupling import (
            Linear, Scaling, Sigmoidal, SigmoidalJansenRit,
            Kuramoto as KuramotoCfun, Difference,
            HyperbolicTangent, PreSigmoidal,
        )
        if isinstance(cfun, Linear):
            setattr(cfun, ['a', 'b'][pidx], np.array([float(value)]))
        elif isinstance(cfun, Scaling):
            assert pidx == 0, f"Scaling: only param_idx=0, got {pidx}"
            cfun.a = np.array([float(value)])
        elif isinstance(cfun, Sigmoidal):
            setattr(cfun, ['a', 'sigma', 'midpoint', 'cmin', 'cmax'][pidx],
                    np.array([float(value)]))
        elif isinstance(cfun, SigmoidalJansenRit):
            # Legacy indices 0-3 preserved for backward compat with saved sweeps.
            # Classic indices 4-6 are new.
            if pidx <= 3:
                setattr(cfun, ['a', 'e0', 'r', 'v0'][pidx],
                        np.array([float(value)]))
            elif pidx <= 6:
                setattr(cfun, ['cmin', 'cmax', 'midpoint'][pidx - 4],
                        np.array([float(value)]))
            else:
                raise IndexError(
                    f"SigmoidalJansenRit has 7 sweepable parameters (0-6), got {pidx}"
                )
        elif isinstance(cfun, KuramotoCfun):
            assert pidx == 0, f"Kuramoto: only param_idx=0, got {pidx}"
            cfun.a = np.array([float(value)])
        elif isinstance(cfun, Difference):
            assert pidx == 0, f"Difference: only param_idx=0, got {pidx}"
            cfun.a = np.array([float(value)])
        elif isinstance(cfun, HyperbolicTangent):
            # Indices 0-2 preserved for backward compat; b is new at index 3.
            setattr(cfun, ['a', 'midpoint', 'sigma', 'b'][pidx], np.array([float(value)]))
        elif isinstance(cfun, PreSigmoidal):
            setattr(cfun, ['H', 'Q', 'G', 'P', 'theta'][pidx], np.array([float(value)]))
        else:
            raise TypeError(f"Unknown cfun type: {type(cfun).__name__}")

    # ===================================================================
    # Unified sweep API
    # ===================================================================

    _resolve_named_params = staticmethod(_resolve_named_params)
    def sweep(
        self,
        network_set,
        params,
        nstep: int = 100,
        *,
        backend: str = "auto",
        n_workers: int = 1,
        monitor: str = "tavg",
        monitor_period: int = 1,
        bold_period: Optional[float] = None,
        chunk_size: Optional[int] = None,
        initial_states: Optional[list] = None,
        node_indices: Optional[dict] = None,
    ) -> "SweepResult":
        """Run a parameter sweep — auto dispatches to GPU or multi-core CPU.

        Parameters
        ----------
        params : dict
            Named parameters to sweep.  Keys → 1-D arrays.  All same length.
            ``'coupling_scale'`` or ``'scale'``: first projection's scaling.
            ``'{proj_name}.{attr}'``: named projection cfun attribute.
            ``'{subnet}.{param}'``: model parameter on a subnet.
        backend : str
            ``'auto'`` (try CUDA → fallback CPU), ``'cpu'``, or ``'cuda'``.
        n_workers : int
            CPU worker processes (fork-based, ignored for CUDA).
        monitor : str
            ``'tavg'``, ``'raw'``, or ``'subsample'``.
        """
        import time as _time_mod

        _reject_balloon_windkessel_sweep(monitor, bold_period)
        self._validate_sweep_monitor_options(
            monitor, monitor_period, bold_period, chunk_size
        )
        if backend not in ("auto", "cpu", "cuda"):
            raise ValueError(
                f"Unsupported sweep backend {backend!r}; expected 'auto', 'cpu', or 'cuda'."
            )
        if (isinstance(nstep, (bool, np.bool_))
                or not isinstance(nstep, (int, np.integer))
                or nstep <= 0):
            raise ValueError("nstep must be a positive integer")
        self._check_compatibility(network_set)
        sweep_descriptor, sweep_values = self._resolve_named_params(network_set, params)

        use_cuda = False
        if backend == "cuda":
            use_cuda = True
        elif backend == "auto":
            try:
                from numba import cuda as _cuda
                use_cuda = _cuda.is_available()
            except ImportError:
                pass

        if use_cuda:
            try:
                return self._sweep_cuda(
                    network_set, sweep_descriptor, sweep_values, nstep,
                    monitor, monitor_period, bold_period, chunk_size,
                    initial_states, node_indices)
            except NotImplementedError:
                if backend == "cuda":
                    raise

        return self._sweep_cpu(
            network_set, sweep_descriptor, sweep_values, nstep,
            n_workers, monitor, monitor_period, bold_period, chunk_size,
            initial_states, node_indices)

    def _sweep_cuda(self, network_set, sweep_descriptor, sweep_values,
                     nstep, monitor, monitor_period, bold_period,
                     chunk_size, initial_states, node_indices):
        # the CUDA sweep kernel integrates every subnet on every tick and
        # writes history at t % H — it has no multi-dt support yet.  Refuse
        # unequal dts so sweep(backend='auto') falls back to the CPU sweep
        # (which implements the pinned multi-dt semantics); an explicit
        # backend='cuda' re-raises.
        dts = {float(sn.scheme.dt) for sn in network_set.subnets}
        if len(dts) > 1:
            raise NotImplementedError(
                "the CUDA sweep kernel does not support multi-dt networks "
                "yet; use backend='cpu' (the CPU sweep implements the "
                "multi-dt master-clock semantics)")
        import time as _time_mod
        from tvb.simulator.backend.nb_hybrid_cuda_sweep_backend import NbHybridCUDASweepBackend

        cuda_backend = NbHybridCUDASweepBackend()
        compiled = cuda_backend.compile_sweep(network_set, sweep_descriptor=sweep_descriptor)
        kwargs = dict(sweep_values=sweep_values, nstep=nstep, monitor_type="raw",
                      monitor_period=1, node_indices=node_indices,
                      record_coupling=True)
        if initial_states is not None: kwargs["initial_states"] = initial_states
        if chunk_size is not None: kwargs["chunk_size"] = chunk_size

        t0 = _time_mod.perf_counter()
        raw = compiled.run(**kwargs)
        elapsed = _time_mod.perf_counter() - t0

        subnet_names = [sn.name for sn in network_set.subnets]
        result = SweepResult(
            tavg=dict(zip(subnet_names, raw["raw"])),
            ctavg=dict(zip(subnet_names, raw["ctraw"])),
            sweep_values=sweep_values,
            backend="cuda",
            elapsed=elapsed,
            snapshot=raw.get("snapshot"),
        )
        return self._finalize_sweep(
            result, network_set, monitor, monitor_period, bold_period,
            chunk_size, node_indices
        )

    def _stack_cpu_results(self, raw_results, network_set, node_indices, sweep_values, backend_label, elapsed):
        """Stack CPU list-of-tuples into SweepResult.

        raw_results: list of n_sweeps tuples, each tuple has n_subnets entries
        of (times, data, ctavg).  data shape: (n_chunks, n_voi, N, modes).
        Stacks into (n_sweeps, n_chunks, n_voi, N, modes) preserving the
        time (chunk) dimension so callers can access per-timestep traces.
        """
        subnet_names = [sn.name for sn in network_set.subnets]
        tavg_dict = {}
        ctavg_dict = {}
        for si, sname in enumerate(subnet_names):
            # Preserve time (chunk) dimension for per-timestep traces
            tavg_arr = np.stack([r[si][1] for r in raw_results], axis=0)
            ctavg_arr = np.stack([r[si][2] for r in raw_results], axis=0)
            tavg_dict[sname] = tavg_arr
            ctavg_dict[sname] = ctavg_arr
        # Merge along node axis
        if node_indices and len(node_indices) > 0:
            n_global = max(max(idxs) for idxs in node_indices.values()) + 1
            ref = list(tavg_dict.values())[0]
            merged = np.zeros((ref.shape[0], ref.shape[1], ref.shape[2], n_global, ref.shape[4]), dtype=np.float32)
            for sname in subnet_names:
                if sname in node_indices:
                    merged[:, :, :, node_indices[sname], :] = tavg_dict[sname]
            merged_tavg = merged
        else:
            # Concatenate along node axis only if all subnets share the same n_voi
            vois = set(a.shape[2] for a in tavg_dict.values())
            if len(vois) == 1:
                merged_tavg = np.concatenate(list(tavg_dict.values()), axis=3)
            else:
                merged_tavg = None  # VOI counts differ — can't merge
        times = raw_results[0][0][0] if raw_results else np.array([])
        return SweepResult(tavg=tavg_dict, merged_tavg=merged_tavg, ctavg=ctavg_dict,
                          times=times, sweep_values=sweep_values, backend=backend_label, elapsed=elapsed)

    @staticmethod
    def _validate_sweep_monitor_options(monitor, monitor_period, bold_period,
                                        chunk_size):
        if monitor not in ("tavg", "raw", "subsample"):
            raise ValueError(
                f"Unsupported sweep monitor {monitor!r}; expected "
                "'tavg', 'raw', or 'subsample'."
            )
        if chunk_size is not None and (
            isinstance(chunk_size, (bool, np.bool_))
            or not isinstance(chunk_size, (int, np.integer))
            or chunk_size <= 0
        ):
            raise ValueError("chunk_size must be a positive integer")
        if (isinstance(monitor_period, (bool, np.bool_))
                or not isinstance(monitor_period, (int, np.integer))
                or monitor_period <= 0):
            raise ValueError("monitor_period must be a positive integer number of steps")
        if bold_period is not None:
            try:
                valid_bold_period = (
                    not isinstance(bold_period, (bool, np.bool_))
                    and np.isfinite(float(bold_period))
                    and float(bold_period) > 0
                )
            except (TypeError, ValueError):
                valid_bold_period = False
            if not valid_bold_period:
                raise ValueError("bold_period must be positive")

    def _finalize_sweep(self, result, network_set, monitor, monitor_period,
                        bold_period, chunk_size, node_indices):
        """Apply the common time-preserving sweep monitor contract."""
        if chunk_size is None:
            chunk_size = 1

        per_step_tavg = result.tavg
        per_step_ctavg = result.ctavg
        # multi-dt: sweep time axes (and Bold sampling) live on the master
        # dt0 grid, exactly like the per-run outputs — NOT the first
        # subnet's dt, which with dts ordered [0.02, 0.01] would stretch
        # every master tick to 2 dt0
        dt = _validate_multi_dt(network_set)
        per_step_times = np.arange(
            1, next(iter(per_step_tavg.values())).shape[1] + 1,
            dtype=np.float64,
        ) * dt
        slices = [slice(start, min(start + chunk_size, len(per_step_times)))
                  for start in range(0, len(per_step_times), chunk_size)]
        result.times = np.asarray([
            (per_step_times[part.start] + per_step_times[part.stop - 1]) * 0.5
            for part in slices
        ], dtype=np.float64)
        result.tavg = {
            name: np.stack([values[:, part].mean(axis=1) for part in slices], axis=1)
            for name, values in per_step_tavg.items()
        }
        result.ctavg = {
            name: np.stack([values[:, part].mean(axis=1) for part in slices], axis=1)
            for name, values in per_step_ctavg.items()
        }

        subnet_names = [sn.name for sn in network_set.subnets]
        compatible = len({(values.shape[2], values.shape[4])
                          for values in result.tavg.values()}) == 1
        if not compatible:
            result.merged_tavg = None
        elif node_indices:
            n_global = max(max(indices) for indices in node_indices.values()) + 1
            ref = next(iter(result.tavg.values()))
            merged = np.zeros(
                (ref.shape[0], ref.shape[1], ref.shape[2], n_global, ref.shape[4]),
                dtype=ref.dtype,
            )
            for name in subnet_names:
                if name in node_indices:
                    merged[:, :, :, node_indices[name], :] = result.tavg[name]
            result.merged_tavg = merged
        else:
            result.merged_tavg = np.concatenate(
                [result.tavg[name] for name in subnet_names], axis=3
            )

        if monitor in ("raw", "subsample"):
            if monitor == "raw":
                selected = np.arange(len(per_step_times))
            else:
                selected = np.arange(monitor_period - 1, len(per_step_times), monitor_period)
            result.raw = {
                name: values[:, selected].copy() for name, values in per_step_tavg.items()
            }

        if bold_period is not None:
            from tvb.simulator.monitors import Bold

            bold = {}
            for name, values in per_step_tavg.items():
                sweep_samples = []
                for sweep_data in values:
                    monitor_instance = Bold(period=float(bold_period))
                    monitor_instance._config_dt(dt)
                    monitor_instance.voi = np.arange(sweep_data.shape[1], dtype=int)
                    monitor_instance.compute_hrf()
                    monitor_instance._config_stock(*sweep_data.shape[1:])
                    samples = [
                        sample for step, state in enumerate(sweep_data, 1)
                        if (sample := monitor_instance.sample(step, state)) is not None
                    ]
                    sweep_samples.append(
                        np.stack([sample[1] for sample in samples])
                        if samples else np.empty((0,) + sweep_data.shape[1:])
                    )
                bold[name] = np.stack(sweep_samples)
            result.bold = bold
        return result

    def _sweep_cpu(self, network_set, sweep_descriptor, sweep_values,
                    nstep, n_workers, monitor, monitor_period, bold_period,
                    chunk_size, initial_states, node_indices):
        import time as _time_mod
        if n_workers > 1 and all(
            desc.get("type") == "cfun" for desc in sweep_descriptor
        ):
            # Use prange-based parallel sweep instead of fork
            result = self._sweep_cpu_prange(
                network_set, sweep_descriptor, sweep_values,
                nstep, initial_states, node_indices)
        else:
            t0 = _time_mod.perf_counter()
            raw = self.run_sweep(
                network_set, sweep_values=sweep_values, nstep=nstep,
                sweep_descriptor=sweep_descriptor, initial_states=initial_states,
                chunk_size=1,
            )
            elapsed = _time_mod.perf_counter() - t0
            result = self._stack_cpu_results(
                raw, network_set, node_indices, sweep_values, "cpu-seq", elapsed
            )
        return self._finalize_sweep(
            result, network_set, monitor, monitor_period, bold_period,
            chunk_size, node_indices
        )


    def _sweep_cpu_prange(self, network_set, sweep_descriptor, sweep_values,
                          nstep, initial_states, node_indices):
        """Multi-core CPU sweep using Numba prange (single-process threading).

        Compiles a ``@nb.njit(parallel=True)`` sweep kernel via a Mako
        template (``nb-hybrid-sweep-cpu.py.mako``), appended to the
        single-sim module so ``sweep_kernel`` can call ``network_chunk``
        directly from inside ``nb.prange``.  Each thread operates on its
        own slice of per-sweep arrays, giving true multi-core parallelism
        without the fork-safety issues of multiprocessing.
        """
        from tvb.simulator.backend.nb_hybrid_sweep_cpu import (
            compile_sweep_kernel, run_sweep_prange)

        # Ensure single-sim function is compiled (and module is cached)
        compiled = self.compile(network_set, eager=True)
        analysis = compiled._analysis

        # Compile the sweep kernel (Mako template + single-sim source)
        kernel_fn = compile_sweep_kernel(self, analysis)

        # Run the sweep
        return run_sweep_prange(
            kernel_fn, analysis, network_set, sweep_descriptor,
            sweep_values, nstep, self, initial_states=initial_states)



    def run_sweep(
        self,
        network_set,
        sweep_values: np.ndarray,
        nstep: int = 100,
        initial_states: Optional[list] = None,
        sweep_descriptor: Optional[list] = None,
        chunk_size: Optional[int] = None,
        bold_period: Optional[float] = None,
        print_source: bool = False,
        **monitors,
    ):
        """Run parameter sweep sequentially on CPU.

        Each sweep point calls run_network() internally. Results are
        returned as a list of per-sweep-point tuples matching the
        run_network() return format.

        Parameters
        ----------
        sweep_values : ndarray (n_sweeps,) or (n_sweeps, n_sweep_dims)
        sweep_descriptor : list of dict, optional
            [{type: 'cfun', projection: 'proj_AB', param_idx: 0},
             {type: 'model', subnet: 'A', param: 'tau_E'}]
        """
        _reject_balloon_windkessel_sweep(bold_period, *monitors.values())
        self._check_compatibility(network_set)
        sweep_values = np.asarray(sweep_values, dtype=np.float32)
        if sweep_values.ndim == 1:
            sweep_values = sweep_values.reshape(-1, 1)

        if sweep_descriptor is None:
            if network_set.projections:
                # Use naming convention for projections
                first_proj = network_set.projections[0]
                proj_name = f"{first_proj.source.name}_to_{first_proj.target.name}"
                sweep_descriptor = [{'type': 'cfun', 'projection': proj_name,
                                     'param_idx': 0}]
            else:
                sweep_descriptor = []

        n_sweeps = sweep_values.shape[0]
        results = []

        targets = []
        for desc in sweep_descriptor:
            if desc['type'] == 'cfun':
                pname = desc['projection']
                pidx = desc.get('param_idx', 0)
                matched_proj = None
                for proj in network_set.projections:
                    expected_name = f"{proj.source.name}_to_{proj.target.name}"
                    if expected_name == pname:
                        matched_proj = proj
                        break
                if matched_proj is None:
                    for sn in network_set.subnets:
                        for proj in sn.projections:
                            expected_name = getattr(proj, 'name', None) or 'intra'
                            if expected_name == pname:
                                matched_proj = proj
                                break
                        if matched_proj is not None:
                            break
                if matched_proj is None:
                    raise ValueError(f"Projection '{pname}' not found in sweep")
                attrs = dict(_CFUN_PARAM_ATTRS.get(type(matched_proj.cfun).__name__, ()))
                attr = next((name for name, index in attrs.items() if index == pidx), None)
                if attr is None:
                    raise IndexError(
                        f"No parameter index {pidx} for "
                        f"{type(matched_proj.cfun).__name__}"
                    )
                targets.append(('cfun', matched_proj.cfun, attr, pidx))
            elif desc['type'] == 'model':
                sname = desc['subnet']
                subnet = next((sn for sn in network_set.subnets if sn.name == sname), None)
                if subnet is None:
                    raise ValueError(f"Subnetwork '{sname}' not found in sweep")
                targets.append(('model', subnet.model, desc['param'], None))

        # Sweep setters replace attributes, so retaining these object references
        # keeps mutable arrays isolated and preserves non-contiguous layouts exactly.
        originals = [(owner, attr, getattr(owner, attr))
                     for _kind, owner, attr, _pidx in targets]
        try:
            for tid in range(n_sweeps):
                sv = sweep_values[tid]
                for dim, (kind, owner, attr, pidx) in enumerate(targets):
                    if kind == 'cfun':
                        self._cfun_set_param(owner, pidx, sv[dim])
                    else:
                        setattr(owner, attr, np.array([float(sv[dim])]))

                kwargs = dict(initial_states=initial_states, print_source=print_source)
                if chunk_size is not None:
                    kwargs['chunk_size'] = chunk_size
                results.append(self.run_network(network_set, nstep=nstep, **kwargs))
        finally:
            for owner, attr, original in originals:
                # The value was already valid on this instance. Bypass NArray's
                # copying setter so restoration retains the exact caller object.
                vars(owner)[attr] = original

        return results
