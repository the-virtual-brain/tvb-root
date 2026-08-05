# Numba Backend UX Analysis

Comparing the user-facing code required to set up and run a hybrid simulation
with the **Numba backend** (`NbHybridBackend`) versus the **regular Python
backend** (`NetworkSet.step` loop or `Simulator`).

Source: minimal examples extracted from `test_nb_hybrid.py` and
`test_nb_hybrid_validate.py`.

---

## 1. Simplest case: single subnet, no projections, no stimulus

### Python (loop) — 4 execution lines

```python
# Setup (identical for both backends)
model = MontbrioPazoRoxin(); model.configure()
scheme = HeunDeterministic(dt=0.1)
sn = Subnetwork(name="ctx", model=model, scheme=scheme, nnodes=4)
sn.configure()
ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
ns.configure()

# Python execution
x = ns.zero_states(initial_states=[ic])
ns.init_projection_buffers(x)          # ← 1
for step in range(1, nstep + 1):      # ← 2
    x = ns.step(step, x)              # ← 3 (loop body)
# x is final state                      ← 4 (closing)
```

### Numba (one-shot) — 3 execution lines

```python
# Setup is identical (same 6 lines)

# Numba execution
backend = NbHybridBackend()                              # ← 1
results = backend.run_network(ns, nstep=100,              # ← 2
                              initial_states=[ic])
times, data, ctavg = results[0]                          # ← 3
```

### Numba (compile-then-run) — 4 execution lines

```python
backend = NbHybridBackend()                              # ← 1
compiled = backend.compile(ns)                           # ← 2
outputs, snapshot = compiled.run(100, chunk_size=1,      # ← 3
    initial_states=[ic], return_snapshot=True)
final = snapshot["states"]                               # ← 4
```

**Execution lines: Python 4, Numba 3–4.**
The setup phase is identical; only the run phase differs.

---

## 2. With stimulus

### Setup boilerplate (shared) — ~12 lines

Both backends require the same stimulus construction:

```python
conn = Connectivity(centres=..., ...); conn.configure()   # 2 lines
temporal = eqs.Linear()                                    # 1 line
temporal.parameters["a"] = 0.0                             # 1 line
temporal.parameters["b"] = 0.05                            # 1 line
weight = np.zeros(N); weight[0] = 1.0                      # 1 line
stim_pattern = StimuliRegion(temporal=temporal,             # 1 line
                             connectivity=conn, weight=weight)
stim = Stim(target=sn, stimulus=stim_pattern,              # 2 lines
            target_cvar=np.array([0], dtype=np.int_))
stim.configure(simulation_length=nstep * DT)               # 1 line
ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])  # 1 line
ns.configure()                                             # 1 line
```

### Run phase — identical to case 1

The stimulus is embedded in the `NetworkSet`. The execution code is exactly
the same as the no-stimulus case — the user does **not** need to change the
run loop or the Numba call.

---

## 3. High-level API: `Simulator` class

```python
ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
tavg = TemporalAverage(period=1.0)
sim = Simulator(nets=ns, simulation_length=10.0, monitors=[tavg])
sim.configure()
(t, y), = sim.run()
```

> **Note:** `Simulator.run()` uses the **Python loop** internally.
> To get Numba acceleration, use `NbHybridBackend.run_network()` or
> `NbHybridBackend().compile()` directly.

---

## Summary

| Aspect | Python loop | Numba backend |
|--------|-------------|---------------|
| **Run code (no stim)** | 4 lines | 3 lines (one-shot) |
| **Run code (with stim)** | 4 lines (same) | 3 lines (same) |
| **Setup code** | Identical | Identical |
| **Extra import** | — | `from tvb.backend.nb_hybrid import NbHybridBackend` |
| **Per-step output** | Manual collection in loop | Automatic `(times, data, ctavg)` |
| **Monitor support** | Use `Simulator` wrapper | Pass `monitors=` to `run_network` |
| **Resume/checkpoint** | Manual state save | Built-in `return_snapshot` + `.resume()` |
| **Learning curve** | Trivial (for loop) | Trivial (3-line API) |

### Verdict

The Numba backend adds **one extra import** and replaces the Python
`for`-loop with a 2–3 line call. Setup code is **100% shared** — the same
`Subnetwork`, `NetworkSet`, `Stim`, and projection objects work with both
backends. The Numba path is actually slightly simpler for users who want
time-series output because it returns `(times, data, ctavg)` automatically
rather than requiring the user to collect per-step observations manually.

### Pain points

1. **Stimulus setup is verbose** (~12 lines of boilerplate for a simple
   constant stimulus). This is not Numba-specific — it affects both backends
   equally. A convenience factory like `create_constant_stim(subnet, amp=0.05)`
   would cut this to 1–2 lines.

2. **`Simulator` doesn't auto-select Numba.** Users who start with the
   `Simulator` class and then want Numba acceleration must switch to
   `NbHybridBackend` directly. A `backend="numba"` option on `Simulator`
   would eliminate this friction.

3. **`Connectivity` required for `StimuliRegion`.** Even for a trivial
   single-node stimulus, the user must construct a zero-weight `Connectivity`
   object. A `StimuliRegion.from_weights(weights, dt=DT)` factory would help.
