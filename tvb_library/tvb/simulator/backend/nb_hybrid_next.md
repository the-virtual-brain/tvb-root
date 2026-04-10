# Hybrid-Numba Next Phase — Plan

> **Date**: 2026-04-10  
> **Prerequisite**: 78 tests passing, 8 models supported, Phases 1–7/A/C1/C2/D/E complete.

---

## 1. Model Codegen Attributes — Bulk Addition (Ralph Loop)

### 1.1 Status

Models already have codegen attrs (`coupling_terms`, `parameter_names`,
`dfun_intermediates`, `state_variable_dfuns`):

| Model | File | nvar |
|-------|------|------|
| MontbrioPazoRoxin | infinite_theta.py | 2 |
| KIonEx | k_ion_exchange.py | 5 |
| JansenRit | jansen_rit.py | 6 |
| Generic2dOscillator | oscillator.py | 2 |
| ReducedWongWang | wong_wang.py | 1 |
| Epileptor | epileptor.py | 6 |
| WilsonCowan | wilson_cowan.py | 2 |
| Linear | linear.py | 1 |

### 1.2 Models needing codegen attrs (scalar per-mode — straightforward)

| Model | File | nvar | Difficulty | Notes |
|-------|------|------|------------|-------|
| SupHopf | oscillator.py | 2 | Easy | Already `ModelNumbaDfun`; has `dfun_helpers` |
| Epileptor2D | epileptor.py | 2 | Easy | Already `ModelNumbaDfun` |
| Kuramoto | oscillator.py | 1 | Easy | sin-based coupling |
| Hopfield | hopfield.py | 2 | Easy | Simple attractor |
| LarterBreakspear | larter_breakspear.py | 3 | Medium | tanh nonlinearities |
| EpileptorRestingState | epileptor_rs.py | 8 | Medium | 8 svars but straightforward scalar ops |
| EpileptorCodim3 | epileptorcodim3.py | 3 | Medium | Already `ModelNumbaDfun` |
| EpileptorCodim3SlowMod | epileptorcodim3.py | 5 | Medium | Already `ModelNumbaDfun` |
| ZetterbergJansen | jansen_rit.py | 12 | Hard | 12 svars; otherwise scalar |
| ReducedWongWangExcInh | wong_wang_exc_inh.py | 2 | Medium | Uses `guvectorize` dfun; needs translation |
| CoombesByrne | infinite_theta.py | 4 | Medium | Theta-neuron OA reduction |
| CoombesByrne2D | infinite_theta.py | 2 | Easy | Simplified version |
| GastSchmidtKnosche_SD | infinite_theta.py | 4 | Medium | Theta-neuron variant |
| GastSchmidtKnosche_SF | infinite_theta.py | 4 | Medium | Theta-neuron variant |
| DumontGutkin | infinite_theta.py | 8 | Hard | 8 svars; complex theta-neuron |
| ZerlautAdaptationFirstOrder | zerlaut.py | 5 | Hard | Adaptive EIF; complex parameter set |
| ZerlautAdaptationSecondOrder | zerlaut.py | 8 | Hard | Extended Zerlaut |

**Total: 17 models**, all scalar per-mode.

### 1.3 Multi-mode models needing combined-mode dfun generation

| Model | File | nvar | n_modes | Difficulty |
|-------|------|------|---------|------------|
| ReducedSetFitzHughNagumo | stefanescu_jirsa.py | 4 | 3 | Hard |
| ReducedSetHindmarshRose | stefanescu_jirsa.py | 6 | 3 | Hard |

These use `numpy.dot(xi, Aik)` — inter-mode matrix ops. Requires template
enhancement to generate a "combined mode" dfun that receives all modes at once
and returns derivatives for all modes. See §3 below.

### 1.4 Strategy

Use the `ralph_add_model_codegen.sh` script (see §5) to iterate over each model
file with `opencode run` and the `zai/glm-5.1` model provider. Each invocation:

1. Reads the model's `dfun()` method
2. Reads an exemplar model (e.g., `Generic2dOscillator`) for the pattern
3. Generates `coupling_terms`, `parameter_names`, `dfun_intermediates`,
   `state_variable_dfuns` attributes
4. The script validates by importing the model and checking attrs exist

---

## 2. Monitor Support

### 2.1 Current state

The Numba kernel returns `(times, data, ctavg)` per subnetwork per chunk, where:
- `data` = temporal average of `state[voi]` (equivalent to TemporalAverage)
- `ctavg` = temporal average of coupling arrays (equivalent to AfferentCouplingTemporalAverage)

Monitor dispatch happens in the Python `Simulator.run()` loop — each monitor's
`.record(step, observed_state)` is called at every step. The Numba backend
bypasses this loop entirely, so monitors are not called.

### 2.2 Goal

Support the monitor types that are most useful for hybrid simulations:

| Monitor | Priority | Effort | Notes |
|---------|----------|--------|-------|
| TemporalAverage | ✅ Done | — | Already output as `data` |
| AfferentCouplingTemporalAverage | ✅ Done | — | Already output as `ctavg` |
| Raw | HIGH | Low | Output every step instead of averaging; controlled by chunk_size=1 |
| SubSample | MEDIUM | Low | Skip steps; can be done in Python post-processing |
| SpatialAverage | MEDIUM | Medium | Needs averaging weight vector per subnetwork |
| GlobalAverage | LOW | Low | Mean across all nodes; post-processing |
| Projection (EEG/MEG/iEEG) | LOW | Medium | Gain matrix multiplication in post-processing or kernel |
| BOLD | LOW | High | Hemodynamic response — complex time-domain convolution |
| AfferentCoupling (raw) | HIGH | Low | Already have ctavg; need raw per-step version |
| ProgressLogger | SKIP | — | UI only, not relevant for Numba |

### 2.3 Implementation approach

**Phase M1: Python-side monitor dispatch (LOW effort, HIGH value)**

After `run_network_fn` returns the chunk data, apply monitors in Python:
- Accept a `monitors=` kwarg on `compile()` and `run()`
- For each chunk, feed `(times, data)` into each monitor's `.record()` method
- Return monitor-formatted output instead of raw `(times, data, ctavg)` tuples

This approach requires NO template changes. Monitors that need per-step state
(like Raw) can set `chunk_size=1`. Monitors that average (TemporalAverage) use
the existing `data` directly.

**Phase M2: In-kernel SubSample / Raw (MEDIUM effort)**

Add a template-generated `step_interval` parameter to `network_chunk`:
- When `step_interval > 1`, only write to output every N-th step (SubSample)
- When `step_interval == 1`, write every step (Raw)
- Avoids the massive memory cost of `chunk_size=1` for long simulations

**Phase M3: In-kernel projection monitors (MEDIUM effort)**

For EEG/MEG/iEEG: multiply the state by a gain matrix inside the kernel.
The gain matrix is a dense `(n_sensors, n_nodes)` float32 array passed as an
additional argument. This avoids returning the full `(nstep, n_voi, n_nodes)`
array when only `(nstep, n_voi, n_sensors)` is needed.

---

## 3. ReducedSet Combined-Mode Dfun Generation

### 3.1 Problem

The current template generates per-mode scalar dfun calls:
```python
for m in range(n_modes):
    xi = state[0, i, m]
    ...
    (d_xi, d_eta, d_alpha, d_beta) = dfun_sn(xi, eta, alpha, beta, c0)
```

ReducedSetFHN needs ALL modes simultaneously to compute `dot(xi, Aik)`.

### 3.2 Solution: "combined mode" template branch

Add a new code path in `nb-hybrid-sim.py.mako` for models that declare
`dfun_mode = "combined"` (new model attribute, default `"scalar"`).

When `dfun_mode == "combined"`:
- dfun function receives **all modes at once**:
  ```python
  def dfun_sn(xi_0, xi_1, xi_2, eta_0, eta_1, eta_2, ..., c0_0, c0_1, c0_2):
  ```
- Inter-mode matrix products are unrolled at code-gen time for fixed n_modes=3:
  ```python
  # dot(xi, Aik) for mode 0:
  Axi_0 = Aik_00 * xi_0 + Aik_01 * xi_1 + Aik_02 * xi_2
  ```
- Integrator receives all modes and advances them together
- The Aik/Bik/Cik matrices are baked in as nb.float32 constants (they're
  derived parameters, fixed after `update_derived_parameters()`)

This is feasible because n_modes is always exactly 3 for ReducedSet models.

### 3.3 Model attributes needed

```python
class ReducedSetFitzHughNagumo:
    dfun_mode = "combined"  # triggers combined-mode template branch
    coupling_terms = ["c_0"]
    parameter_names = ['tau', 'a', 'b', 'K11', 'K12', 'K21', 'sigma', 'mu']
    # derived_matrix_params: populated by update_derived_parameters()
    # These are (n_modes, n_modes) or (1, n_modes) arrays baked as constants
    derived_matrix_params = ['Aik', 'Bik', 'Cik', 'e_i', 'f_i', 'IE_i', 'II_i', 'm_i', 'n_i']
    state_variable_dfuns = {
        # expressions use mode-suffixed variables: xi_0, xi_1, xi_2, etc.
        'xi': [
            "tau * (xi_{m} - e_i_{m} * xi_{m}**3 / 3.0 - eta_{m}) + K11 * (Axi_{m} - xi_{m}) - K12 * (Balpha_{m} - xi_{m}) + tau * (IE_i_{m} + c_0_{m})",
            ...  # one per mode, OR a single expression with {m} placeholder
        ],
        ...
    }
```

### 3.4 Effort estimate

- Template changes: ~100 lines (new branch in integrate, new dfun signature)
- Model attrs: ~50 lines per model (ReducedSetFHN + ReducedSetHR)
- Tests: 8 tests (4 per model)
- **Total: MEDIUM-HIGH effort**. Implement AFTER bulk scalar model attrs.

---

## 4. Execution Order

1. **Bulk model codegen attrs** (Ralph loop) — §1
   - 17 models, automated via opencode
   - After each batch: add to `_check_compatibility`, run tests
2. **Monitor Phase M1** — §2.3 Python-side dispatch
   - Accept `monitors=` kwarg
   - No template changes
3. **Combined-mode models** (ReducedSetFHN + HR) — §3
   - Template `dfun_mode="combined"` branch
   - 2 models
4. **Monitor Phase M2** — §2.3 in-kernel SubSample
   - Template `step_interval` parameter

---

## 5. Ralph Loop Script

See `ralph_add_model_codegen.sh` in same directory.
