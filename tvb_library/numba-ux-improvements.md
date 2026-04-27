# Numba Backend UX Improvement Proposals

Three concrete proposals addressing the pain points identified in
`numba-backend-ux.md`. Each includes the API change, before/after code,
effort estimate, and compatibility notes.

---

## Proposal 1: Stimulus Quick-Build Functions

### Problem

Creating a simple constant stimulus requires ~12 lines of boilerplate
(`Connectivity`, `StimuliRegion`, equation parameters, `Stim` wrapper,
`configure()`). Most of this is mechanical and should be a 1-liner.

### Solution

Add convenience factory functions to `stimulus_utils.py` that handle the
entire pipeline — `Connectivity` construction, equation setup,
`StimuliRegion` creation, `Stim` wrapping, and `configure()` — in a single
call.

#### New functions

```python
# In tvb/simulator/hybrid/stimulus_utils.py

def constant_stim(subnet, amplitude, target_node=0, target_cvar=0,
                  projection_scale=1.0, simulation_length=None):
    """One-line constant-amplitude stimulus."""
    ...

def pulse_stim(subnet, amplitude, onset, period, pulse_width,
               target_node=0, target_cvar=0, projection_scale=1.0,
               simulation_length=None):
    """One-line pulse-train stimulus."""
    ...

def sinusoid_stim(subnet, amplitude, frequency, target_node=0,
                  target_cvar=0, projection_scale=1.0,
                  simulation_length=None):
    """One-line sinusoidal stimulus."""
    ...
```

Each function:
1. Builds a zero-weight `Connectivity` of the right size (or accepts one).
2. Creates and configures the equation (`Linear`, `PulseTrain`, `Sinusoid`).
3. Constructs a `StimuliRegion` with the given `weight` vector.
4. Wraps it in `Stim` with the given `target_cvar` and `projection_scale`.
5. Calls `stim.configure(simulation_length)` if `simulation_length` is provided.

#### Before (current, ~12 lines)

```python
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.patterns import StimuliRegion
from tvb.datatypes import equations as eqs
from tvb.simulator.hybrid.stimulus import Stim

conn = Connectivity(
    centres=np.zeros((N, 3)), weights=np.zeros((N, N)),
    tract_lengths=np.zeros((N, N)),
    region_labels=np.array([str(i) for i in range(N)]),
    speed=np.array([1.0]),
)
conn.configure()
temporal = eqs.Linear()
temporal.parameters["a"] = 0.0
temporal.parameters["b"] = 0.05
weight = np.zeros(N)
weight[0] = 1.0
stim_pattern = StimuliRegion(temporal=temporal, connectivity=conn, weight=weight)
stim = Stim(target=sn, stimulus=stim_pattern,
            target_cvar=np.array([0], dtype=np.int_), projection_scale=1.0)
stim.configure(simulation_length=NSTEP * DT)
```

#### After (proposed, 1–2 lines)

```python
from tvb.simulator.hybrid.stimulus_utils import constant_stim

stim = constant_stim(sn, amplitude=0.05, target_node=0,
                     simulation_length=NSTEP * DT)
```

#### Implementation effort: **Low**

The test helpers in `test_nb_hybrid_validate.py` already implement
`_make_constant_stim`, `_make_sinusoidal_stim`, and `_make_pulse_stim`
with exactly this logic. They just need to be productionized into
`stimulus_utils.py` with proper docstrings and edge-case handling
(e.g. `target_node=None` → all nodes, accepting existing `Connectivity`).

Estimated: ~100–150 lines of new code. No changes to existing modules.

#### Breaking changes / compatibility: **None**

These are additive — new public functions in an existing module. The
low-level `Stim`, `StimuliRegion`, `Connectivity` APIs remain unchanged.
Users who prefer explicit construction can continue to use it.

---

## Proposal 2: `backend` parameter on `Simulator`

### Problem

`Simulator.run()` always uses the Python `NetworkSet.step` loop. Users who
want Numba acceleration must abandon `Simulator` entirely and call
`NbHybridBackend.run_network()` directly, losing the convenience of
`Simulator.configure()`, monitor wiring, and `random_state` handling.

### Solution

Add a `backend` attribute to `Simulator` that selects the execution engine.
When `backend="numba"`, `run()` delegates to `NbHybridBackend` internally.

#### API change

```python
class Simulator(t.HasTraits):
    nets: NetworkSet = t.Attr(NetworkSet)
    monitors: List[Monitor] = t.List(of=Monitor)
    simulation_length: float = t.Float()
    backend: str = t.String(default="python")  # NEW: "python" or "numba"
```

The `run()` method gains a branch:

```python
def run(self, **kwargs):
    if not hasattr(self, "_dt0"):
        self.configure()
    if self.backend == "numba":
        return self._run_numba(**kwargs)
    return self._run_python(**kwargs)

def _run_python(self, **kwargs):
    # existing loop code (unchanged)
    ...

def _run_numba(self, **kwargs):
    from tvb.simulator.backend.nb_hybrid import NbHybridBackend
    initial_conditions = kwargs.pop("initial_conditions", None)
    random_state = kwargs.pop("random_state", None)
    ics = self._resolve_ics(initial_conditions, random_state)
    be = NbHybridBackend()
    raw = be.run_network(
        self.nets, nstep=self._nstep, monitors=self.monitors,
        initial_states=ics,
    )
    # Convert Numba monitor output to Simulator's (times, data) format
    return self._format_monitor_output(raw)
```

#### Before (current — two different APIs)

```python
# Python path (via Simulator)
sim = Simulator(nets=ns, simulation_length=10.0, monitors=[tavg])
sim.configure()
(t, y), = sim.run(random_state=42)

# Numba path (bypass Simulator entirely)
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
backend = NbHybridBackend()
results = backend.run_network(ns, nstep=100, monitors=[tavg],
                              initial_states=[ic])
```

#### After (proposed — single API)

```python
# Python path (unchanged)
sim = Simulator(nets=ns, simulation_length=10.0, monitors=[tavg],
                backend="python")
sim.configure()
(t, y), = sim.run(random_state=42)

# Numba path (just change one parameter)
sim = Simulator(nets=ns, simulation_length=10.0, monitors=[tavg],
                backend="numba")
sim.configure()
(t, y), = sim.run(random_state=42)
```

#### Implementation effort: **Medium**

The main work is in `_run_numba`:
- Extract and refactor the existing `run()` body into `_run_python`.
- Implement `_run_numba` that calls `NbHybridBackend.run_network` and
  converts the `list[list[(times, data)]]` output into the
  `list[(times, data)]` format that `Simulator.run()` returns.
- Handle `random_state` → `initial_states` conversion (already implemented
  in `_run_python`, can be shared).
- Handle `simulation_length` → `nstep` conversion.
- Ensure `configure()` is still called correctly for both paths.

Estimated: ~80–120 lines of changes in `simulator.py`. The
`NbHybridBackend` side needs no changes.

#### Breaking changes / compatibility: **None**

- `backend` defaults to `"python"`, so all existing `Simulator` usage
  behaves identically.
- `Simulator` users who were already calling `NbHybridBackend` directly
  can continue to do so — the low-level API is not deprecated.
- The only risk is if a user has subclassed `Simulator` and overridden
  `run()` — but the existing code is simply refactored into `_run_python`,
  so a `super().run()` call would still work.

---

## Proposal 3: Lightweight `StimuliRegion` without `Connectivity`

### Problem

`StimuliRegion` requires a full `Connectivity` object even for trivial
single-node stimuli. The `Connectivity` constructor is 6 lines of boilerplate
with `centres`, `weights`, `tract_lengths`, `region_labels`, `speed`,
and a `configure()` call. For a stimulus this is pure ceremony — the
connectivity is never used for coupling.

### Solution

Add a `StimuliRegion.from_weights()` class method that constructs the
necessary `Connectivity` internally, and a companion
`_make_minimal_connectivity()` helper in `stimulus_utils.py`.

#### API change

```python
# In tvb/datatypes/patterns.py (on StimuliRegion)

class StimuliRegion(SpatioTemporalPattern):
    ...

    @classmethod
    def from_weights(cls, weight, temporal):
        """Create a StimuliRegion from a weight vector and temporal equation.

        Builds a minimal Connectivity internally so the user doesn't have to.

        Parameters
        ----------
        weight : array-like, shape (n_nodes,)
            Per-node stimulus weights.
        temporal : TemporalApplicableEquation
            Temporal equation (e.g. ``Linear()``, ``PulseTrain()``).

        Returns
        -------
        StimuliRegion
            Ready-to-configure stimulus pattern.
        """
        from tvb.datatypes.connectivity import Connectivity
        n = len(weight)
        conn = Connectivity(
            centres=np.zeros((n, 3)),
            weights=np.zeros((n, n)),
            tract_lengths=np.zeros((n, n)),
            region_labels=np.array([str(i) for i in range(n)]),
            speed=np.array([1.0]),
        )
        conn.configure()
        return cls(temporal=temporal, connectivity=conn, weight=weight)
```

#### Before (current, ~8 lines for pattern alone)

```python
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.patterns import StimuliRegion
from tvb.datatypes import equations as eqs

conn = Connectivity(
    centres=np.zeros((4, 3)), weights=np.zeros((4, 4)),
    tract_lengths=np.zeros((4, 4)),
    region_labels=np.array(["0", "1", "2", "3"]),
    speed=np.array([1.0]),
)
conn.configure()
temporal = eqs.Linear()
temporal.parameters["a"] = 0.0
temporal.parameters["b"] = 0.05
weight = np.array([1.0, 0.0, 0.0, 0.0])
stim_pattern = StimuliRegion(temporal=temporal, connectivity=conn, weight=weight)
```

#### After (proposed, ~4 lines for pattern)

```python
from tvb.datatypes.patterns import StimuliRegion
from tvb.datatypes import equations as eqs

temporal = eqs.Linear()
temporal.parameters["a"] = 0.0
temporal.parameters["b"] = 0.05
stim_pattern = StimuliRegion.from_weights(
    weight=np.array([1.0, 0.0, 0.0, 0.0]),
    temporal=temporal,
)
```

And combined with Proposal 1, the full stimulus becomes a single call:

```python
stim = constant_stim(sn, amplitude=0.05, simulation_length=10.0)
```

#### Implementation effort: **Low**

The `from_weights` class method is ~15 lines. A private
`_make_minimal_connectivity(n)` helper can be shared with the
`constant_stim` / `pulse_stim` / `sinusoid_stim` functions from Proposal 1.

Estimated: ~30 lines in `patterns.py` + minor refactor of Proposal 1's
internal helper.

#### Breaking changes / compatibility: **None**

- `from_weights()` is a new class method. The existing `__init__` that
  requires `connectivity=` is unchanged.
- Existing code that constructs `Connectivity` + `StimuliRegion` manually
  continues to work identically.
- The internal `Connectivity` created by `from_weights` is a zero-weight
  placeholder — it is only used by `StimuliRegion` for spatial pattern
  lookup, never for coupling computation. This matches what every test
  in the test suite already does.

---

## Summary table

| # | Proposal | New public API | Effort | Breaking? |
|---|----------|---------------|--------|-----------|
| 1 | Stimulus quick-build functions | `constant_stim()`, `pulse_stim()`, `sinusoid_stim()` in `stimulus_utils` | Low | No |
| 2 | `backend` param on `Simulator` | `Simulator(backend="numba")` | Medium | No |
| 3 | `StimuliRegion.from_weights()` | Class method on `StimuliRegion` | Low | No |

All three proposals are **purely additive** — no existing API is removed or
changed. They can be implemented independently and in any order.
Proposals 1 and 3 complement each other (1 uses 3 internally), but neither
depends on the other.
