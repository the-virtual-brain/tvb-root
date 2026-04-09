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
    HeunDeterministic, EulerDeterministic,
    HeunStochastic, EulerStochastic,
)


# ---------------------------------------------------------------------------
# Helpers used by both Python (NbHybridBackend) and Mako templates
# ---------------------------------------------------------------------------

def _cfun_type(p: "ProjectionInfo") -> str:
    """Return the string coupling-function type for a ProjectionInfo."""
    from tvb.simulator.hybrid.coupling import Linear, Scaling
    if p.cfun is None:
        return "none"
    if isinstance(p.cfun, Linear):
        return "linear"
    if isinstance(p.cfun, Scaling):
        return "scaling"
    return "none"


def _cfun_params(p: "ProjectionInfo"):
    """Return (cfun_a, cfun_b) floats for a ProjectionInfo."""
    from tvb.simulator.hybrid.coupling import Linear, Scaling
    if p.cfun is None:
        return 1.0, 0.0
    if isinstance(p.cfun, Linear):
        return float(p.cfun.a[0]), float(p.cfun.b[0])
    if isinstance(p.cfun, Scaling):
        return float(p.cfun.a[0]), 0.0
    return 1.0, 0.0


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
        f"Projection '{p.name}': unsupported cvar mapping "
        f"({ns} source → {nt} target)"
    )


# ---------------------------------------------------------------------------
# Data classes for code-generation analysis
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SubnetworkInfo:
    name: str
    model: object          # Model instance
    integrator: object     # Integrator instance
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
    source_cvar: np.ndarray   # (n_src_cvar,)
    target_cvar: np.ndarray   # (n_tgt_cvar,)
    weights_data: np.ndarray  # (nnz,) float32
    weights_indices: np.ndarray  # (nnz,) int
    weights_indptr: np.ndarray   # (n_tgt+1,) int
    idelays: np.ndarray       # (nnz,) int
    horizon: int
    scale: float
    target_scales: np.ndarray  # (n_tgt_cvar,) or empty
    cfun: object              # coupling function or None
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

    @property
    def all_projections(self) -> List[ProjectionInfo]:
        return self.inter_projections + self.intra_projections


# ---------------------------------------------------------------------------
# Module-level compiled-function cache
# ---------------------------------------------------------------------------

# Keyed by SHA-256 of the rendered (pre-autopep8) source string so the same
# topology produces the same key regardless of which NbHybridBackend instance
# triggers the compile.  This survives new NbHybridBackend() instantiations
# within a single Python process.
_COMPILED_FN_CACHE: dict = {}


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

        Returns
        -------
        list of (times, data)
            Same format as :meth:`NbHybridBackend.run_network`.
        """
        return self._backend._run_compiled(
            self._run_network_fn,
            self._analysis,
            self._network_set,
            nstep,
            chunk_size,
            initial_states,
        )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class NbHybridBackend(MakoUtilMix):
    """Numba backend for hybrid simulator with multi-subnetwork support.

    Call ``run_network(network_set, nstep)`` to compile and run.  The
    network_set must already be configured (``configure()`` called).

    Only MontbrioPazoRoxin models with HeunDeterministic or EulerDeterministic
    integrators are supported in this initial version.
    """

    def compile(
        self,
        network_set: NetworkSet,
        print_source: bool = False,
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
        self._check_compatibility(network_set)
        analysis = self._analyse(network_set)
        run_network_fn = self._build(
            '<%include file="nb-hybrid-sim.py.mako"/>',
            dict(analysis=analysis, np=np, debug_nojit=False),
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

        Returns
        -------
        list of (times, data)
            One tuple per subnetwork in ``network_set.subnets``, where
            ``times`` is a 1-D float64 array of mid-chunk time points and
            ``data`` is a float32 array of shape ``(n_chunks, n_voi, n_nodes, n_modes)``.
        """
        return self.compile(network_set, print_source).run(
            nstep, chunk_size, initial_states
        )

    # ------------------------------------------------------------------
    # _run_compiled — arg assembly + kernel call (no compilation logic)
    # ------------------------------------------------------------------

    def _run_compiled(
        self,
        run_network_fn,
        analysis: "NetworkAnalysis",
        network_set: NetworkSet,
        nstep: int,
        chunk_size: int,
        initial_states: Optional[list],
    ) -> list:
        """Build the argument list and call the pre-compiled kernel."""
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

        # Per-projection arrays
        for p in analysis.all_projections:
            buf = self._make_projection_buffer(p, sn_states, network_set)
            args.append(buf)
            args.append(p.weights_data.astype(np.float32))
            args.append(p.weights_indices.astype(np.int32))
            args.append(p.weights_indptr.astype(np.int32))
            args.append(p.idelays.astype(np.int32))
            if p.is_inter:
                args.append(p.mode_map.astype(np.float32))
            args.append(p.source_cvar.astype(np.int32))
            args.append(p.target_cvar.astype(np.int32))
            args.append(np.float32(p.scale))
            ts = p.target_scales.astype(np.float32) if p.target_scales.size > 0 else np.zeros(0, dtype=np.float32)
            args.append(ts)
            cfun_a, cfun_b = _cfun_params(p)
            args.append(np.float32(cfun_a))
            args.append(np.float32(cfun_b))
            args.append(np.int32(p.horizon))

        # Per-subnetwork noise arrays (stochastic integrators)
        for sn_info in analysis.subnetworks:
            if sn_info.is_stochastic:
                sn_obj = next(s for s in network_set.subnets if s.name == sn_info.name)
                dt = sn_obj.scheme.dt
                rng = sn_obj.scheme.noise.random_stream
                # Draw in (nstep, n_vars, n_nodes, n_modes) order so that
                # transposed [:, :, :, t] == t-th sequential randn(n_vars, n_nodes, n_modes) call
                dw = rng.randn(nstep, sn_info.model.nvar, sn_info.n_nodes, sn_info.n_modes)
                noise_std = np.sqrt(2.0 * sn_info.noise_nsig * dt)  # (n_vars,)
                dw *= noise_std[np.newaxis, :, np.newaxis, np.newaxis]
                # Transpose to (n_vars, n_nodes, n_modes, nstep)
                dw = np.ascontiguousarray(np.transpose(dw, (1, 2, 3, 0))).astype(np.float32)
                args.append(dw)

        # Per-subnetwork stimulus arrays (pre-computed batch)
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

        return run_network_fn(*args)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_compatibility(self, network_set: NetworkSet):
        from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
        _allowed_integrators = (HeunDeterministic, EulerDeterministic, HeunStochastic, EulerStochastic)
        dt0 = network_set.subnets[0].scheme.dt
        for sn in network_set.subnets:
            if not isinstance(sn.model, MontbrioPazoRoxin):
                raise NotImplementedError(
                    f"NbHybridBackend only supports MontbrioPazoRoxin; "
                    f"subnetwork '{sn.name}' uses {type(sn.model).__name__}"
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

    def _analyse(self, network_set: NetworkSet) -> NetworkAnalysis:
        from tvb.simulator.noise import Additive

        # Build stimulus lookup: subnet name -> list of Stim objects
        stims_by_subnet: dict = {sn.name: [] for sn in network_set.subnets}
        for stim in (network_set.stimuli or []):
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
                        noise_nsig = np.full(sn.model.nvar, float(nsig), dtype=np.float64)
                    else:
                        noise_nsig = np.broadcast_to(nsig, (sn.model.nvar,)).copy().astype(np.float64)
                else:
                    noise_nsig = np.ones(sn.model.nvar, dtype=np.float64)
            subnets.append(SubnetworkInfo(
                name=sn.name,
                model=sn.model,
                integrator=sn.scheme,
                n_nodes=sn.nnodes,
                n_modes=sn.model.number_of_modes,
                is_stochastic=is_stoch,
                noise_nsig=noise_nsig,
                has_stimulus=bool(stims_by_subnet[sn.name]),
            ))

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

        return NetworkAnalysis(
            subnetworks=subnets,
            inter_projections=inter_projs,
            intra_projections=intra_projs,
            stimuli_by_subnet=stims_by_subnet,
        )

    def _build_projection_info(self, p, is_inter: bool) -> ProjectionInfo:
        ts = p.target_scales if p.target_scales is not None else np.zeros(0, dtype=np.float64)

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
            src_name = ""   # filled by caller
            tgt_name = ""
            n_src_modes = 1  # will be filled
            mode_map = None
            proj_name = getattr(p, "name", None) or "intra"

        pi = ProjectionInfo(
            name=proj_name,
            source_subnet=src_name,
            target_subnet=tgt_name,
            source_cvar=np.atleast_1d(p.source_cvar).astype(np.int32),
            target_cvar=np.atleast_1d(p.target_cvar).astype(np.int32),
            weights_data=p.weights.data.astype(np.float32),
            weights_indices=p.weights.indices.astype(np.int32),
            weights_indptr=p.weights.indptr.astype(np.int32),
            idelays=np.atleast_1d(p.idelays).astype(np.int32),
            horizon=int(p._horizon),
            scale=float(p.scale),
            target_scales=np.atleast_1d(ts).astype(np.float32) if ts.size > 0 else np.zeros(0, dtype=np.float32),
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

        globals_ = {}
        try:
            exec(source, globals_)
        except Exception as exc:
            print(self.insert_line_numbers(autopep8.fix_code(source)))
            raise exc
        fn = globals_["run_network"]
        _COMPILED_FN_CACHE[cache_key] = fn
        return fn
