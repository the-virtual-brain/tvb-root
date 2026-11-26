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
Top-level simulator for hybrid network models

``Simulator`` owns a ``NetworkSet`` and a list of TVB ``Monitor`` objects.  It
validates integration time steps across subnetworks (uniform for the Python
backend; integer-multiple multirate for compiled Numba/C++ backends), wires up
the monitors, and executes the simulation loop.

Typical usage::

    sim = Simulator(nets=my_nets, monitors=[Raw()], simulation_length=1000.0)
    sim.configure()
    [(times, data)] = sim.run()

See Also
--------
tvb.simulator.simulator : Classic TVB simulator for comparison
"""

import math
import warnings
import numpy as np
from typing import List

import tvb.basic.neotraits.api as t
from tvb.simulator.monitors import EEG, MEG, iEEG, BalloonWindkessel, Monitor
from .network import NetworkSet


class Simulator(t.HasTraits):
    """Simulator for hybrid network models.

    The Simulator class manages the simulation of a hybrid network model,
    including configuration of monitors, backend selection, and time stepping.

    Attributes
    ----------
    nets : NetworkSet
        The network model to simulate
    monitors : list
        List of monitors for recording data
    simulation_length : float
        Total simulation time in milliseconds
    backend : str, default="python"
        Execution backend: ``"python"`` (pure Python ``NetworkSet.step``),
        ``"numba"`` (JIT-compiled ``NbHybridBackend``), or ``"cpp"`` (SIMD
        C++ ``CppHybridBackend``).
    """

    _BACKEND_OPTIONS = ("python", "numba", "cpp")

    nets: NetworkSet = t.Attr(NetworkSet)
    monitors: List[Monitor] = t.List(of=Monitor)
    simulation_length: float = t.Float()

    def __init__(self, **kwargs):
        backend = kwargs.pop("backend", "python")
        if backend not in self._BACKEND_OPTIONS:
            raise ValueError(
                f"backend must be one of {self._BACKEND_OPTIONS}, got {backend!r}"
            )
        if backend == "cpp":
            try:
                from tvb.simulator.backend.cpp_hybrid import CppHybridBackend  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "backend='cpp' requires the compiled '_cpp_hybrid' extension "
                    "(Python >= 3.12 or WASM build with '_cpp_hybrid')."
                ) from exc
        self.backend = backend
        super().__init__(**kwargs)
        self.validate_dts()
        self.validate_vois()

    def validate_vois(self):
        """Configure monitor stock buffers based on total variables of interest.

        Counts the combined number of variables of interest across all
        subnetworks and the total node count, then calls ``_config_stock`` on
        any monitor that exposes it so the monitor allocates a correctly-sized
        internal ring buffer before the simulation starts.

        This method is called automatically during ``__init__``.
        """
        if len(self.monitors) == 0:
            return
        if self.nets._is_merged_mode():
            total_vois = len(self.nets.subnets[0].model.variables_of_interest)
            num_nodes = (
                max(int(ix) for sn in self.nets.subnets for ix in sn.node_indices) + 1
            )
        else:
            total_vois = sum(
                [len(sn.model.variables_of_interest) for sn in self.nets.subnets]
            )
            num_nodes = sum([sn.nnodes for sn in self.nets.subnets])
        for monitor in self.monitors:
            if hasattr(monitor, "_config_stock"):
                if hasattr(monitor, "compute_hrf") and monitor._interim_istep is None:
                    monitor.compute_hrf()
                monitor._config_stock(total_vois, num_nodes, 1)

    def validate_dts(self):
        """Validate integration time steps across subnetworks and configure monitors.

        For ``backend == "python"``, every subnetwork must share the same
        ``dt`` as the first subnetwork (``NetworkSet.step`` is single-rate).
        For compiled backends (``"numba"``, ``"cpp"``), each subnetwork's
        ``dt`` must be an integer multiple of the smallest ``dt`` across
        subnetworks (``self._dt0``).  All global monitors are then configured
        with ``self._dt0`` and ``voi = slice(None)``.

        Raises
        ------
        ValueError
            If ``backend == "python"`` and subnetworks differ in ``dt``, or if
            a compiled backend receives non-integer-multiple ``dt`` values.

        Notes
        -----
        Called automatically during ``__init__``; the validated base step size
        is stored as ``self._dt0`` for use by ``configure`` and ``run``.
        """
        self._validate_monitor_placement()
        if self.backend == "python":
            self._dt0 = self.nets.subnets[0].scheme.dt
            for sn in self.nets.subnets[1:]:
                if self._dt0 != sn.scheme.dt:
                    raise ValueError(
                        f"dt mismatch across subnetworks: expected {self._dt0}, "
                        f"got {sn.scheme.dt} for subnetwork '{sn.name}'"
                    )
        else:
            from tvb.simulator.backend.hybrid_common import _validate_multi_dt

            self._dt0 = _validate_multi_dt(self.nets)
        for monitor in self.monitors:
            monitor: Monitor
            monitor._config_dt(self._dt0)
            if not isinstance(monitor, BalloonWindkessel):
                monitor.voi = slice(None)  # all vars

    def _validate_monitor_placement(self):
        if self.backend != "cpp" and any(
                isinstance(rec.monitor, BalloonWindkessel)
                for sn in self.nets.subnets for rec in sn.monitors):
            raise NotImplementedError(
                "BalloonWindkessel local Recorders require the explicit C++ hybrid "
                "backend (backend='cpp')"
            )
        if self.backend != "cpp" and any(isinstance(m, BalloonWindkessel) for m in self.monitors):
            raise NotImplementedError("BalloonWindkessel requires the explicit C++ hybrid backend (backend='cpp')")

        if self.backend == "cpp" and any(isinstance(rec.monitor, (EEG, MEG, iEEG))
                                         for sn in self.nets.subnets for rec in sn.monitors):
            raise NotImplementedError("C++ EEG/MEG/iEEG cannot be local or Recorder monitors")

    def configure(self):
        """Configure the simulator and its monitors."""
        self._validate_monitor_placement()
        if self.monitors:
            missing = [sn.name for sn in self.nets.subnets if sn.node_indices is None]
            if missing:
                warnings.warn(
                    f"Global monitors are attached but subnetwork(s) {missing} do not "
                    f"have node_indices set. Monitor output will be in concatenated "
                    f"subnetwork order, not original connectome order. Set node_indices "
                    f"on each Subnetwork to obtain connectome-ordered output.",
                    UserWarning,
                    stacklevel=2,
                )

        for subnet in self.nets.subnets:
            # Subnetwork.configure() also configures its IntraProjections and stimuli
            subnet.configure(simulation_length=self.simulation_length)
            # Configure recorders in each subnetwork
            for recorder in subnet.monitors:
                recorder.configure(self.simulation_length)

        # Configure the NetworkSet, which configures InterProjections
        self.nets.configure()

    def run(self, **kwargs):
        """Run the simulation and return recorded monitor data.

        If the simulator has not yet been configured (i.e. ``configure`` has
        not been called), it is configured automatically before the time loop
        begins.

        Parameters
        ----------
        initial_conditions : list of ndarray, optional
            One state array per subnetwork, each with shape
            ``(nvar, nnodes, modes)``.  When provided, these arrays are used
            directly as the initial state and to pre-fill projection history
            buffers.  Takes precedence over ``random_state``.
        random_state : None, int, or numpy.random.RandomState, optional
            Seed or generator used to draw random initial conditions from each
            model's ``state_variable_range`` (matching classic TVB default
            behaviour).  Ignored when ``initial_conditions`` is supplied.

            * ``None`` (default) — use the global ``numpy.random`` module.
            * ``int`` — create a ``numpy.random.RandomState`` seeded with
              this value, so results are reproducible.
            * ``numpy.random.RandomState`` — use the provided instance
              directly.

        Returns
        -------
        list of tuple
            One ``(times, data)`` tuple per monitor, where ``times`` is a
            1-D ``ndarray`` of recording time points and ``data`` is an
            ``ndarray`` of the corresponding monitor output.  Returns an
            empty list when no monitors are attached.

        Examples
        --------
        >>> [(t, d)] = sim.run()
        >>> [(t, d)] = sim.run(random_state=42)          # reproducible ICs
        >>> [(t, d)] = sim.run(initial_conditions=[ic])  # explicit ICs
        """
        self._validate_monitor_placement()
        # Configure if not already done
        if not hasattr(self, "_dt0"):
            self.configure()

        # Accept initial_conditions and random_state as parameters to run()
        initial_conditions = kwargs.pop("initial_conditions", None)
        random_state = kwargs.pop("random_state", None)

        if self.backend == "numba":
            return self._run_numba(initial_conditions, random_state)
        if self.backend == "cpp":
            return self._run_cpp(initial_conditions, random_state)
        return self._run_python(initial_conditions, random_state)

    def _resolve_ics(self, initial_conditions, random_state):
        """Resolve initial conditions from explicit arrays or random state."""
        if initial_conditions is not None:
            return self.nets.zero_states(
                initial_states=[ic.copy() for ic in initial_conditions]
            )
        if random_state is None:
            rng = np.random
        elif isinstance(random_state, int):
            rng = np.random.RandomState(random_state)
        else:
            rng = random_state
        return self.nets.random_states(rng)

    def _run_python(self, initial_conditions=None, random_state=None):
        """Run simulation using the Python NetworkSet.step loop."""
        mts = [[] for _ in self.monitors]
        mxs = [[] for _ in self.monitors]

        x = self._resolve_ics(initial_conditions, random_state)

        # Initialize projection history buffers with initial state
        # to match classic TVB simulator behavior (history filled with ICs)
        self.nets.init_projection_buffers(x)

        stop = int(math.ceil(self.simulation_length / self._dt0))
        for step in range(1, stop + 1):
            x = self.nets.step(step, x)
            if self.monitors:
                ox = self.nets.observe(x, flat=True)
                for mt, mx, mon in zip(mts, mxs, self.monitors):
                    maybe_tx = mon.record(step, ox)
                    if maybe_tx is not None:
                        mt.append(maybe_tx[0])
                        mx.append(maybe_tx[1])
        return [(np.array(t), np.array(x)) for t, x in zip(mts, mxs)]

    def _run_numba(self, initial_conditions=None, random_state=None):
        """Run simulation using the NbHybridBackend."""
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        return self._run_compiled_backend(
            NbHybridBackend(), initial_conditions, random_state
        )

    def _run_cpp(self, initial_conditions=None, random_state=None):
        """Run simulation using the CppHybridBackend."""
        from tvb.simulator.backend.cpp_hybrid import CppHybridBackend

        return self._run_compiled_backend(
            CppHybridBackend(), initial_conditions, random_state
        )

    def _run_compiled_backend(self, be, initial_conditions=None, random_state=None):
        """Run simulation using a compiled hybrid backend."""
        x = self._resolve_ics(initial_conditions, random_state)
        ics = list(x)
        nstep = int(math.ceil(self.simulation_length / self._dt0))

        has_recorders = any(sn.monitors for sn in self.nets.subnets)
        if self.backend == "cpp" and has_recorders:
            subnet_monitors = {
                sn.name: [rec.monitor for rec in sn.monitors]
                for sn in self.nets.subnets
                if sn.monitors
            }
            raw, subnet_results = be.run_network(
                self.nets,
                nstep=nstep,
                monitors=self.monitors if self.monitors else None,
                initial_states=ics,
                subnet_monitors=subnet_monitors,
            )
            result = self._reassemble_monitor_outputs(raw) if self.monitors else []
            self._populate_subnet_recorders(subnet_results)
            return result

        saved_rng_states = None
        if self.monitors and has_recorders:
            from tvb.simulator.integrators import IntegratorStochastic

            saved_rng_states = [
                (
                    sn.scheme.noise.random_stream,
                    sn.scheme.noise.random_stream.get_state(),
                )
                for sn in self.nets.subnets
                if isinstance(sn.scheme, IntegratorStochastic)
            ]

        result = []
        if self.monitors:
            raw = be.run_network(
                self.nets,
                nstep=nstep,
                monitors=self.monitors,
                initial_states=ics,
            )
            result = self._reassemble_monitor_outputs(raw)

        if has_recorders:
            if saved_rng_states is not None:
                for stream, state in saved_rng_states:
                    stream.set_state(state)
            raw_steps = be.run_network(
                self.nets,
                nstep=nstep,
                chunk_size=1,
                initial_states=ics,
            )
            self._record_subnet_monitors(raw_steps, nstep)
        elif not self.monitors:
            # No monitors or recorders: run and return empty list (matching Python path)
            be.run_network(self.nets, nstep=nstep, initial_states=ics)

        return result

    def _populate_subnet_recorders(self, subnet_results):
        """Populate per-subnetwork Recorder buffers from single-pass C++ outputs."""
        for sn in self.nets.subnets:
            if not sn.monitors:
                continue
            mon_outputs = subnet_results.get(sn.name, ())
            for rec, (t_arr, d_arr) in zip(sn.monitors, mon_outputs):
                t_arr = np.asarray(t_arr, dtype=np.float32)
                d_arr = np.asarray(d_arr, dtype=np.float32)
                n_emitted = int(t_arr.shape[0])
                if n_emitted == 0:
                    rec.times = None
                    rec.samples = None
                    rec._current_idx = 0
                    continue
                if rec.num_samples is None or n_emitted > rec.num_samples:
                    rec.num_samples = n_emitted
                if (
                    rec.samples is None
                    or rec.samples.shape[0] < rec.num_samples
                    or rec.samples.shape[1:] != d_arr.shape[1:]
                ):
                    rec._allocate_arrays(d_arr.shape[1:])
                rec.times[:n_emitted] = t_arr
                rec.samples[:n_emitted] = d_arr
                rec._current_idx = n_emitted

    def _reassemble_monitor_outputs(self, raw):
        """Reassemble compiled backend monitor outputs into ``(times, data)`` pairs."""
        # raw is list[list[tuple(times, data)]] indexed by [monitor][subnet].
        # Merged monitors (GlobalAverage, SpatialAverage, Projection)
        # return a single-element list from _apply_monitors.
        result = []
        for monitor, mon_raw in zip(self.monitors, raw):
            if isinstance(monitor, BalloonWindkessel):
                # The stream is already stamped on its exact physical grid.
                [(times, data)] = mon_raw
                result.append((times, data))
                continue
            times, subnet_data = zip(*mon_raw)
            times = tuple(
                (np.rint(np.asarray(t, dtype=np.float64) / self._dt0 * 2.0) / 2.0)
                * self._dt0
                for t in times
            )
            # Single-subnet or merged monitor: single-element list, use directly
            if len(mon_raw) == 1:
                result.append((times[0], subnet_data[0]))
                continue

            t = times[0]
            if not all(np.array_equal(t, other) for other in times[1:]):
                raise ValueError("Monitor time grids differ across subnetworks")

            voi_counts = {d.shape[1] for d in subnet_data}
            mode_counts = {d.shape[3] for d in subnet_data}
            if (
                self.nets._is_merged_mode()
                and len(voi_counts) == 1
                and all(
                    d.shape[2] == len(sn.node_indices)
                    for sn, d in zip(self.nets.subnets, subnet_data)
                )
            ):
                # Merged mode: scatter per-subnet nodes into connectome order.
                # Use d.shape[3] (observed output has modes summed to 1)
                # rather than model.number_of_modes.
                total_nodes = (
                    max(int(ix) for sn in self.nets.subnets for ix in sn.node_indices)
                    + 1
                )
                total_vois = subnet_data[0].shape[1]
                n_modes = max(mode_counts)
                data = np.zeros(
                    (len(t), total_vois, total_nodes, n_modes), dtype=np.float64
                )
                for sn, d in zip(self.nets.subnets, subnet_data):
                    data[:, :, sn.node_indices, : d.shape[3]] = d
            elif len(voi_counts) == 1 and len(mode_counts) == 1:
                # Unmerged with identical VOI counts: concatenate nodes only
                data = np.concatenate(subnet_data, axis=2)
            else:
                # Unmerged with heterogeneous VOI counts: block-diagonal
                # VOI + node placement matching NetworkSet.observe(flat=True)
                total_vois = sum(d.shape[1] for d in subnet_data)
                total_nodes = sum(d.shape[2] for d in subnet_data)
                n_modes = max(mode_counts)
                data = np.zeros(
                    (len(t), total_vois, total_nodes, n_modes), dtype=np.float64
                )
                voi_off = 0
                node_off = 0
                for d in subnet_data:
                    nv, nn, nm = d.shape[1], d.shape[2], d.shape[3]
                    data[:, voi_off : voi_off + nv, node_off : node_off + nn, :nm] = d
                    voi_off += nv
                    node_off += nn
            result.append((t, data))
        return result

    def _record_subnet_monitors(self, raw_steps, nstep):
        """Feed per-subnetwork Recorder instances from chunk_size=1 raw outputs."""
        from tvb.simulator.monitors import (
            AfferentCoupling,
            AfferentCouplingTemporalAverage,
        )

        for si, sn in enumerate(self.nets.subnets):
            if not sn.monitors:
                continue
            _, tavg, ctavg = raw_steps[si]
            k_j = max(1, int(round(float(sn.scheme.dt) / self._dt0)))
            for step in range(1, nstep + 1):
                if step % k_j != 0:
                    continue
                sn_step = step // k_j
                for rec in sn.monitors:
                    if isinstance(
                        rec.monitor,
                        (AfferentCoupling, AfferentCouplingTemporalAverage),
                    ):
                        sample = ctavg[step - 1]
                    else:
                        sample = tavg[step - 1]
                    rec.record(sn_step, sample)
