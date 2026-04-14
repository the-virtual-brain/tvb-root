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

import dataclasses
import hashlib
import numpy as np
import autopep8
from typing import List, Optional

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
]


# ---------------------------------------------------------------------------
# Helpers used by both Python (NbHybridBackend) and Mako templates
# ---------------------------------------------------------------------------


def _apply_monitors(
    raw_outputs: list,
    monitors: list,
    dt: float,
) -> list:
    """Transform per-subnet (times, data, ctavg) tuples into monitor-dispatched output.

    Parameters
    ----------
    raw_outputs : list of (times, data, ctavg)
        One tuple per subnetwork, as returned by the compiled kernel.
    monitors : list of Monitor instances
        Each monitor determines which view / transform of the raw data to return.
    dt : float
        Integration time step (ms).

    Returns
    -------
    list[list[tuple[ndarray, ndarray]]]
        Outer list indexed by monitor, inner list by subnetwork.
        Each element is (times, data).
    """
    from tvb.simulator.monitors import (
        TemporalAverage,
        Raw,
        SubSample,
        GlobalAverage,
        AfferentCoupling,
        SpatialAverage,
        Projection,
        Bold,
    )

    for m in monitors:
        if isinstance(m, Raw):
            pass
        elif isinstance(
            m,
            (TemporalAverage, SubSample, GlobalAverage, AfferentCoupling,
             SpatialAverage),
        ):
            pass
        elif isinstance(m, Projection):
            if not hasattr(m, '_gain') or m._gain is None:
                raise ValueError(
                    f"Projection monitor {type(m).__name__} has no gain matrix "
                    "configured. Set the gain matrix before running (e.g. via "
                    "config_for_sim or by setting m._gain directly)."
                )
        elif isinstance(m, Bold):
            # Bold needs compute_hrf() called; ensure dt and istep are set
            if not hasattr(m, 'istep') or m.istep is None:
                m.dt = dt
                m._config_dt(dt)
            if not hasattr(m, 'hemodynamic_response_function'):
                m.compute_hrf()
        else:
            raise NotImplementedError(
                f"Monitor {type(m).__name__} is not yet supported by the Numba backend. "
                "Supported: TemporalAverage, Raw, SubSample, GlobalAverage, "
                "AfferentCoupling, SpatialAverage, Projection (EEG/MEG/iEEG), Bold."
            )

    results: list = []
    for m in monitors:
        per_subnet: list = []
        for times, data, ctavg in raw_outputs:
            if isinstance(m, AfferentCoupling):
                per_subnet.append((times, ctavg))
            elif isinstance(m, Projection):
                # gain is (n_sensors, n_nodes)
                gain = m.gain.astype(data.dtype)
                # data is (n_chunks, n_voi, n_nodes, n_modes) — sum over modes
                data_2d = data.sum(axis=-1)  # (n_chunks, n_voi, n_nodes)
                # Project: einsum over nodes dim
                projected = np.einsum('ij,tkj->tki', gain, data_2d)  # (n_chunks, n_voi, n_sensors)
                # Add back singleton modes dim
                projected = projected[..., np.newaxis]  # (n_chunks, n_voi, n_sensors, 1)
                per_subnet.append((times, projected))
            elif isinstance(m, GlobalAverage):
                per_subnet.append((times, data.mean(axis=-2, keepdims=True)))
            elif isinstance(m, Bold):
                from tvb.datatypes import equations
                hrf = m.hemodynamic_response_function  # (1, stock_steps)
                stock_steps = hrf.shape[1]
                interim_istep = m._interim_istep
                n_chunks, n_voi, n_nodes, n_modes = data.shape
                sample_shape = (n_voi, n_nodes, n_modes)

                # Initialise per-monitor state on first encounter (per subnet)
                if not hasattr(m, '_nb_state'):
                    m._nb_state = True
                    m._nb_interim_stock = np.zeros((interim_istep,) + sample_shape, dtype=np.float32)
                    m._nb_stock = np.zeros((stock_steps,) + sample_shape, dtype=np.float32)
                    m._nb_step_offset = 0
                    m._nb_subnets = []  # list of (interim, stock) per subnet index
                # Grow per-subnet storage if needed
                while len(m._nb_subnets) <= len(per_subnet):
                    m._nb_subnets.append((
                        np.zeros((interim_istep,) + sample_shape, dtype=np.float32),
                        np.zeros((stock_steps,) + sample_shape, dtype=np.float32),
                    ))
                interim_stock, stock = m._nb_subnets[len(per_subnet)]

                bold_results = []
                bold_times = []
                offset = m._nb_step_offset
                for ci in range(n_chunks):
                    step = offset + ci + 1
                    # Update interim stock at every chunk (= integration step when chunk_size=1)
                    interim_stock[(step - 1) % interim_istep] = data[ci]
                    # At interim period, update main stock with temporal average
                    if step % interim_istep == 0:
                        avg = np.mean(interim_stock, axis=0)
                        stock[(step // interim_istep - 1) % stock_steps] = avg
                    # At Bold period, compute HRF convolution
                    if step % m.istep == 0:
                        t_bold = times[ci] if ci < len(times) else step * dt
                        rolled_hrf = np.roll(
                            hrf, (step // interim_istep - 1) % stock_steps, axis=1
                        )
                        # stock is (stock_steps, n_voi, n_nodes, n_modes)
                        stock_t = stock.transpose((1, 2, 0, 3))  # (n_voi, n_nodes, stock_steps, n_modes)
                        bold = np.dot(rolled_hrf, stock_t)  # (1, n_voi, n_nodes, n_modes)
                        bold = bold.reshape(sample_shape)  # squeeze HRF dim
                        # Apply FirstOrderVolterra scaling if applicable
                        if isinstance(m.hrf_kernel, equations.FirstOrderVolterra):
                            k1 = m.hrf_kernel.parameters.get('k_1', 1.0)
                            V0 = m.hrf_kernel.parameters.get('V_0', 1.0)
                            bold = (bold - 1.0) * k1 * V0
                        bold_results.append(bold)
                        bold_times.append(t_bold)
                # Store updated buffers
                m._nb_subnets[len(per_subnet)] = (interim_stock, stock)

                if bold_results:
                    bold_arr = np.stack(bold_results, axis=0)  # (n_bold, n_voi, n_nodes, n_modes)
                    per_subnet.append((np.array(bold_times, dtype=np.float64), bold_arr))
                else:
                    per_subnet.append((
                        np.array([], dtype=np.float64),
                        np.empty((0,) + sample_shape, dtype=np.float32),
                    ))
            elif isinstance(m, SpatialAverage):
                # m.spatial_mean is (n_areas, n_nodes), configured during config_for_sim
                if hasattr(m, 'spatial_mean'):
                    # data is (n_chunks, n_voi, n_nodes, n_modes)
                    spatial = np.einsum('ij,tklm->tkim', m.spatial_mean, data)
                    # spatial is (n_chunks, n_voi, n_areas, n_modes)
                    per_subnet.append((times, spatial))
                else:
                    # spatial_mean not configured — pass through unchanged
                    per_subnet.append((times, data))
            elif isinstance(m, SubSample):
                period = float(m.period)
                mask = np.abs(times - np.round(times / period) * period) < dt
                if np.any(mask):
                    per_subnet.append((times[mask], data[mask]))
                else:
                    per_subnet.append((
                        np.array([], dtype=times.dtype),
                        np.empty((0,) + data.shape[1:], dtype=data.dtype),
                    ))
            elif isinstance(m, TemporalAverage):
                per_subnet.append((times, data))
            elif isinstance(m, Raw):
                per_subnet.append((times, data))
            else:
                raise NotImplementedError(
                    f"Monitor {type(m).__name__} is not yet supported by the Numba backend. "
                    "Supported: TemporalAverage, Raw, SubSample, GlobalAverage, "
                    "AfferentCoupling, SpatialAverage, Projection (EEG/MEG/iEEG)."
                )
        results.append(per_subnet)
    return results


def _cfun_type(p: "ProjectionInfo") -> str:
    """Return the string coupling-function type for a ProjectionInfo."""
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

    if p.cfun is None:
        return "none"
    if isinstance(p.cfun, Linear):
        return "linear"
    if isinstance(p.cfun, Scaling):
        return "scaling"
    if isinstance(p.cfun, Sigmoidal):
        return "sigmoidal"
    if isinstance(p.cfun, SigmoidalJansenRit):
        return "sigmoidal_jr"
    if isinstance(p.cfun, KuramotoCfun):
        return "kuramoto"
    # Difference.post() is a * x — same as Scaling
    if isinstance(p.cfun, Difference):
        return "scaling"
    if isinstance(p.cfun, HyperbolicTangent):
        return "tanh"
    if isinstance(p.cfun, PreSigmoidal):
        return "pre_sigmoidal"
    return "none"


def _cfun_params(p: "ProjectionInfo") -> "np.ndarray":
    """Return a float32 array of length 8 with cfun parameters for a ProjectionInfo.

    Layout by cfun type:
      none:          [1.0, 0, 0, 0, 0, 0, 0, 0]
      linear:        [a, b, 0, 0, 0, 0, 0, 0]
      scaling:       [a, 0, 0, 0, 0, 0, 0, 0]
      sigmoidal:     [a, sigma, midpoint, cmin, cmax, 0, 0, 0]
      sigmoidal_jr:  [a, e0, r, v0, 0, 0, 0, 0]
      kuramoto:      [a, 0, 0, 0, 0, 0, 0, 0]
      tanh:          [a, midpoint, sigma, 0, 0, 0, 0, 0]
      pre_sigmoidal: [H, Q, G, P, theta, 0, 0, 0]
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

    arr = np.zeros(8, dtype=np.float32)
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
        arr[0] = float(p.cfun.a[0])
        arr[1] = float(p.cfun.e0[0])
        arr[2] = float(p.cfun.r[0])
        arr[3] = float(p.cfun.v0[0])
        return arr
    if isinstance(p.cfun, KuramotoCfun):
        arr[0] = float(p.cfun.a[0])
        return arr
    if isinstance(p.cfun, HyperbolicTangent):
        arr[0] = float(p.cfun.a[0])
        arr[1] = float(p.cfun.midpoint[0])
        arr[2] = float(p.cfun.sigma[0])
        return arr
    if isinstance(p.cfun, PreSigmoidal):
        arr[0] = float(p.cfun.H[0])
        arr[1] = float(p.cfun.Q[0])
        arr[2] = float(p.cfun.G[0])
        arr[3] = float(p.cfun.P[0])
        arr[4] = float(p.cfun.theta[0])
        return arr
    return arr


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


@dataclasses.dataclass
class ProjectionInfo:
    name: str
    source_subnet: str
    target_subnet: str
    source_cvar: np.ndarray  # (n_src_cvar,)
    target_cvar: np.ndarray  # (n_tgt_cvar,)
    weights_data: np.ndarray  # (nnz,) float32
    weights_indices: np.ndarray  # (nnz,) int
    weights_indptr: np.ndarray  # (n_tgt+1,) int
    idelays: np.ndarray  # (nnz,) int
    horizon: int
    scale: float
    target_scales: np.ndarray  # (n_tgt_cvar,) or empty
    cfun: object  # coupling function or None
    is_inter: bool
    # mode_map only for inter projections
    mode_map: Optional[np.ndarray] = None  # (n_src_modes, n_tgt_modes)

    @property
    def n_tgt_nodes(self) -> int:
        return self.weights_indptr.shape[0] - 1

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
# triggers the compile.  This survives new NbHybridBackend() instantiations
# within a single Python process.
_COMPILED_FN_CACHE: dict = {}


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
    import tempfile
    from pathlib import Path

    cache_dir = Path(tempfile.gettempdir()) / "tvb_nb_hybrid_cache"
    cache_dir.mkdir(exist_ok=True)
    mod_name = f"nbhybrid_{cache_key[:16]}"
    mod_path = cache_dir / f"{mod_name}.py"
    if not mod_path.exists():
        # Atomic write: write to .tmp then os.replace to avoid partial reads.
        tmp_path = mod_path.with_suffix(".tmp")
        tmp_path.write_text(source, encoding="utf-8")
        os.replace(tmp_path, mod_path)
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so Numba can find it for cache lookup.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
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

    def run(
        self,
        nstep: int,
        chunk_size: int = 1,
        initial_states: Optional[list] = None,
        return_snapshot: bool = False,
        _initial_buffers: Optional[dict] = None,
        monitors: Optional[list] = None,
    ) -> list:
        """Execute the pre-compiled kernel for *nstep* integration steps.

        Parameters
        ----------
        nstep : int
            Number of integration steps to run.
        chunk_size : int
            Number of steps per temporal-average chunk (default 1 = raw output).
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
            is a dict with keys ``'states'`` and ``'buffers'`` suitable for passing
            to :meth:`resume`.
        """
        if monitors is not None:
            from tvb.simulator.monitors import Raw

            for m in monitors:
                if isinstance(m, Raw) and chunk_size != 1:
                    raise ValueError(
                        "Raw monitor requires chunk_size=1; "
                        "pass chunk_size=1 to run_network()"
                    )
        outputs, final_states, final_bufs = self._backend._run_compiled(
            self._run_network_fn,
            self._analysis,
            self._network_set,
            nstep,
            chunk_size,
            initial_states,
            _initial_buffers=_initial_buffers,
        )
        if monitors is not None:
            dt = self._network_set.subnets[0].scheme.dt
            outputs = _apply_monitors(outputs, monitors, dt)
        if not return_snapshot:
            return outputs
        snapshot = {
            "states": [
                final_states[sn.name].copy() for sn in self._analysis.subnetworks
            ],
            "buffers": {name: buf.copy() for name, buf in final_bufs.items()},
        }
        return outputs, snapshot

    def resume(
        self,
        snapshot: dict,
        nstep: int,
        chunk_size: int = 1,
        return_snapshot: bool = False,
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
        chunk_size : int
            Steps per temporal-average chunk.
        return_snapshot : bool
            If True, also return a new snapshot of the final state.

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
        import tempfile
        from pathlib import Path

        return Path(tempfile.gettempdir()) / "tvb_nb_hybrid_cache"

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
    ) -> "CompiledNetworkFn":
        """Compile the simulation kernel for *network_set* and return it.

        The compiled kernel is cached in-process by a SHA-256 hash of the
        generated source.  Repeated calls with topologically identical networks
        return the cached kernel immediately (no re-compilation).

        Parameters
        ----------
        network_set : NetworkSet
            Fully configured network (``configure()`` must have been called).
        print_source : bool
            If True, print the generated (autopep8-formatted) source.

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
        return CompiledNetworkFn(
            _backend=self,
            _analysis=analysis,
            _run_network_fn=run_network_fn,
            _network_set=network_set,
        )

    def run_network(
        self,
        network_set: NetworkSet,
        nstep: int,
        chunk_size: int = 1,
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
        chunk_size : int
            Number of steps per temporal-average chunk (default 1 = raw output).
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
        return self.compile(network_set, print_source, debug_nojit=debug_nojit).run(
            nstep, chunk_size, initial_states, monitors=monitors
        )

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
    ) -> list:
        """Build the argument list and call the pre-compiled kernel."""
        # Guard: chunk_size must not exceed the minimum horizon
        if analysis.all_projections:
            min_horizon = min(p.horizon for p in analysis.all_projections)
            if chunk_size > min_horizon:
                raise ValueError(
                    f"chunk_size={chunk_size} exceeds the minimum projection horizon "
                    f"({min_horizon} steps = min_delay / dt). "
                    f"Reduce chunk_size to at most {min_horizon}, or increase tract "
                    "lengths / reduce dt."
                )
        # Build argument list matching the generated run_network() signature
        args = [nstep]

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
            args.append(np.float32(p.scale))
            ts = (
                p.target_scales.astype(np.float32)
                if p.target_scales.size > 0
                else np.zeros(0, dtype=np.float32)
            )
            args.append(ts)
            cfun_params = _cfun_params(p)
            args.append(cfun_params)

        # Per-subnetwork noise arrays (stochastic integrators)
        for sn_info in analysis.subnetworks:
            if sn_info.is_stochastic:
                sn_obj = next(s for s in network_set.subnets if s.name == sn_info.name)
                dt = sn_obj.scheme.dt
                rng = sn_obj.scheme.noise.random_stream
                # Draw in (nstep, n_vars, n_nodes, n_modes) order so that
                # transposed [:, :, :, t] == t-th sequential randn(n_vars, n_nodes, n_modes) call
                dw = rng.randn(
                    nstep, sn_info.model.nvar, sn_info.n_nodes, sn_info.n_modes
                )
                noise_std = np.sqrt(2.0 * sn_info.noise_nsig * dt)  # (n_vars,)
                dw *= noise_std[np.newaxis, :, np.newaxis, np.newaxis]
                # Transpose to (n_vars, n_nodes, n_modes, nstep)
                dw = np.ascontiguousarray(np.transpose(dw, (1, 2, 3, 0))).astype(
                    np.float32
                )
                args.append(dw)

        # Per-subnetwork stimulus arrays (pre-computed batch)
        # TODO §8.4: use lazy chunk-by-chunk path when estimated stim_arr_mb
        #   exceeds _STIM_LAZY_THRESHOLD_MB (or TVB_HYBRID_LAZY_STIM_MB env var).
        #   See _compute_stimulus_lazy() for the planned implementation.
        for sn_info in analysis.subnetworks:
            if sn_info.has_stimulus:
                n_cvar = len(sn_info.model.coupling_terms)
                stim_arr = np.zeros(
                    (n_cvar, sn_info.n_nodes, sn_info.n_modes, nstep),
                    dtype=np.float32,
                )
                for stim in analysis.stimuli_by_subnet[sn_info.name]:
                    for step_idx in range(1, nstep + 1):
                        sc = np.asarray(stim.get_coupling(step_idx), dtype=np.float32)
                        if sc.ndim == 2:
                            sc = sc[:, :, np.newaxis]
                        if sc.shape[2] == 1 and sn_info.n_modes > 1:
                            sc = np.broadcast_to(
                                sc, (sc.shape[0], sn_info.n_nodes, sn_info.n_modes)
                            ).copy()
                        # broadcast += matches Python path (tgt += stim.get_coupling(step))
                        stim_arr[:, :, :, step_idx - 1] += sc
                args.append(stim_arr)

        args.append(chunk_size)

        outputs = run_network_fn(*args)
        return outputs, sn_states, src_bufs

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
        n_cvar = len(sn_info.model.coupling_terms)
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

        n_cvar = len(sn_info.model.coupling_terms)
        n_bytes = n_cvar * sn_info.n_nodes * sn_info.n_modes * nstep * 4  # float32
        return n_bytes / (1024 * 1024)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_compatibility(self, network_set: NetworkSet) -> None:
        from tvb.simulator.models.infinite_theta import (
            MontbrioPazoRoxin,
            CoombesByrne2D,
            CoombesByrne,
            GastSchmidtKnosche_SD,
            GastSchmidtKnosche_SF,
            DumontGutkin,
        )
        from tvb.simulator.models.k_ion_exchange import KIonEx
        from tvb.simulator.models.jansen_rit import JansenRit, ZetterbergJansen
        from tvb.simulator.models.oscillator import (
            Generic2dOscillator,
            SupHopf,
            Kuramoto,
        )
        from tvb.simulator.models.wong_wang import ReducedWongWang
        from tvb.simulator.models.wong_wang_exc_inh import ReducedWongWangExcInh
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

        _supported_models = (
            MontbrioPazoRoxin,
            KIonEx,
            JansenRit,
            Generic2dOscillator,
            ReducedWongWang,
            ReducedWongWangExcInh,
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
        )
        _allowed_integrators = (
            HeunDeterministic,
            EulerDeterministic,
            HeunStochastic,
            EulerStochastic,
        )
        dt0 = network_set.subnets[0].scheme.dt
        for sn in network_set.subnets:
            if not isinstance(sn.model, _supported_models):
                raise NotImplementedError(
                    f"NbHybridBackend does not support {type(sn.model).__name__}. "
                    f"Supported: MontbrioPazoRoxin, KIonEx, JansenRit, ZetterbergJansen, "
                    f"Generic2dOscillator, SupHopf, Kuramoto, Hopfield, LarterBreakspear, "
                    f"ReducedWongWang, ReducedWongWangExcInh, "
                    f"Epileptor, Epileptor2D, EpileptorCodim3, EpileptorCodim3SlowMod, EpileptorRestingState, "
                    f"WilsonCowan, ZerlautAdaptation*, "
                    f"CoombesByrne2D, CoombesByrne, GastSchmidtKnosche_SD/SF, DumontGutkin, "
                    f"ReducedSetFitzHughNagumo, ReducedSetHindmarshRose."
                )
            if not isinstance(sn.scheme, _allowed_integrators):
                raise NotImplementedError(
                    f"NbHybridBackend only supports Heun/EulerDeterministic or Stochastic; "
                    f"subnetwork '{sn.name}' uses {type(sn.scheme).__name__}"
                )
            if sn.scheme.dt != dt0:
                raise ValueError(
                    "All subnetworks must share the same dt. "
                    f"Expected {dt0}, got {sn.scheme.dt} in '{sn.name}'"
                )
            if isinstance(sn.model, Epileptor):
                if sn.model.modification[0]:
                    raise NotImplementedError(
                        "NbHybridBackend: Epileptor with modification=True is not supported. "
                        "Set model.modification = numpy.array([False])."
                    )
            if isinstance(sn.model, WilsonCowan):
                if not sn.model.shift_sigmoid[0]:
                    raise NotImplementedError(
                        "NbHybridBackend: WilsonCowan with shift_sigmoid=False is not supported. "
                        "Use the default shift_sigmoid=True."
                    )
        from tvb.simulator.models.stefanescu_jirsa import ReducedSetBase

        for sn in network_set.subnets:
            if isinstance(sn.model, ReducedSetBase):
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

        # Build stimulus lookup: subnet name -> list of Stim objects
        stims_by_subnet: dict = {sn.name: [] for sn in network_set.subnets}
        for stim in network_set.stimuli or []:
            stims_by_subnet[stim.target.name].append(stim)

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
        for _p in _all_projs:
            src = _p.source_subnet
            source_horizons[src] = max(source_horizons.get(src, 1), _p.horizon)
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
        )

    def _build_projection_info(self, p, is_inter: bool) -> "ProjectionInfo":
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
        pi = ProjectionInfo(
            name=proj_name,
            source_subnet=src_name,
            target_subnet=tgt_name,
            source_cvar=np.atleast_1d(p.source_cvar).astype(np.int32),
            target_cvar=np.atleast_1d(p.target_cvar).astype(np.int32),
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
