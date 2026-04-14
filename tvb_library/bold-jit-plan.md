# Bold JIT Plan: Replace HRF Convolution with Balloon ODE Model

## Goal
Replace the HRF-convolution-based Bold monitor with a Balloon model (ODE-based) that runs inside the `@nb.njit` JIT kernel. This eliminates the module-level `_BOLD_STATE` dict, the complex HRF stock/interim buffers, and makes Bold naturally stateful across calls.

## Architecture

### Balloon Model (4 ODEs per node per voi)
From `vbjax/neural_mass.py`:

```
ds/dt = x - (1/tau_s)*s - (1/tau_f)*(f - 1)
df/dt = s
dv/dt = (1/tau_o)*(f - v^(1/alpha))
dq/dt = (1/tau_o)*(f*(1 - (1-e0)^(1/f))/e0 - v^(1/alpha)*(q/v))
BOLD = v0*(k1*(1-q) + k2*(1-q/v) + k3*(1-v))
```

Initial conditions: s=0, f=1, v=1, q=1 (steady state with x=0).

Parameters (precomputed reciprocals):
- tau_s=0.65, tau_f=0.41, tau_o=0.98, alpha=0.32
- te=0.04, v0=4.0, e0=0.4
- epsilon=0.5, nu_0=40.3, r_0=25.0
- k1 = 4.3*nu_0*e0*te = 4.3*40.3*0.4*0.04 = 2.773
- k2 = epsilon*r_0*e0*te = 0.5*25.0*0.4*0.04 = 0.2
- k3 = 1 - epsilon = 0.5

### Input: x
The Balloon model takes neural activity `x` as input. In our case, `x` is the temporal-averaged voi value (same signal that feeds into tavg). We feed the **first voi** (index 0) as the neural driving signal, summed over modes, for each node. This matches TVB convention where voi[0] is typically the "interesting" variable.

Actually, to be more general, we should feed **all vois** through the Balloon model separately, producing a separate BOLD signal per voi. But that's 4x the state per voi. The vbjax reference feeds a single scalar x.

**Decision**: Feed voi[0] summed over modes as the neural input x to the Balloon model. The output is BOLD per node (not per voi). This matches standard neuroscience practice (BOLD signal per brain region, not per variable of interest).

Wait, looking at the existing Bold monitor output shape: `(n_bold, n_voi, n_nodes, n_modes)`. It produces one BOLD signal per voi per node per mode. The HRF convolution treats each voi/mode independently.

**Decision**: For compatibility with existing tests, we need BOLD output shape `(n_bold, n_voi, n_nodes, n_modes)`. But running a separate Balloon ODE per voi is wasteful. Compromise: run Balloon for voi[0] only, replicate the output across all vois. OR run separate Balloon per voi.

Actually, looking at the existing test expectations, the test `test_bold_matches_python` expects `nb_d.shape == py_bold_arr.shape` which is `(n_bold, n_voi, n_nodes, n_modes)`. And the existing HRF approach convolves each voi independently.

**Revised decision**: Run 4 ODEs per node per voi (NOT per mode — modes are summed). This keeps output shape compatible. Shape of Bold state: `(n_voi, 4, n_nodes)` where the 4 are (s, f, v, q).

### Integration Scheme
Use Euler (not Heun) for the Balloon ODE, since it's a slow hemodynamic model and Euler is simpler in the JIT context. The Bold period is ~2000ms so we need ~200000 Euler steps at dt=0.01ms, which is fine since they're trivial.

Actually, the Balloon model has its own timescale. In vbjax, it uses the same dt as the neural model. The derivatives are stable with Euler at dt=0.01ms.

**Decision**: Use Heun (same as neural model) for consistency and accuracy. At each neural integration step, also advance the Balloon ODE by one step.

### Sampling
BOLD signal is sampled at the monitor's period (typically 2000ms). At each step, compute BOLD = v0*(k1*(1-q) + k2*(1-q/v) + k3*(1-v)). Only store the value when step_count % bold_istep == 0.

### Implementation Steps

#### Step 1: Define Balloon dfun in template as `@nb.njit`
Add a `bold_dfun` function inside the generated code. It computes derivatives for the 4 Balloon state variables given (s, f, v, q) and neural input x.

#### Step 2: Add Bold state arrays to `network_chunk`
Per subnetwork:
- `bold_state`: `(n_voi, 4, n_nodes)` float32 — [s, f, v, q] per voi per node
- `bold_buf`: `(n_voi, 4, n_nodes)` float32 — k2 for Heun intermediate
- Bold parameters: `bold_tau_s`, `bold_tau_f`, `bold_tau_o`, `bold_alpha`, `bold_e0`, `bold_v0`, `bold_k1`, `bold_k2`, `bold_k3` (all float32 scalars)
- `bold_istep`: int — steps between BOLD samples
- `bold_count`: int — sample counter
- `bold_output_buf`: list/accumulator for sampled BOLD values

Actually, Numba njit can't handle Python lists. We need to pre-allocate a fixed-size buffer.

**Approach**: Allocate `bold_samples` as `(max_bold_samples, n_voi, n_nodes, 1)` float32, and a counter. After `network_chunk` returns, collect the samples.

But `network_chunk` runs for `nstep` steps within a single call, and we need to know how many Bold samples will be produced. With chunk_size=1, at most 1 Bold sample per chunk. With larger chunks, at most `chunk_size / bold_istep` samples.

**Simpler approach**: Don't accumulate BOLD samples in the JIT. Instead, at each step:
1. Advance Balloon ODE one step
2. Compute current BOLD signal value
3. Store it in a per-step array

Then in `run_network` (Python), downsample at the Bold period.

This avoids complexity with variable-length output in the JIT.

**Even simpler**: Accumulate BOLD in `run_network` which is already Python. The JIT just advances the Balloon ODE state. After each chunk, `run_network` checks if the step count hits the Bold period, and if so, computes BOLD = v0*(k1*(1-q) + k2*(1-q/v) + k3*(1-v)) and appends it.

**Wait, but the whole point is to move Bold to JIT.** The HRF convolution is already in Python. The benefit of Balloon ODE is:
1. It's simpler (4 ODEs vs HRF stock/interim)
2. State is naturally maintained in arrays, no module-level dict
3. It CAN run in JIT if we want

Let me reconsider. The Balloon ODE advance itself should happen in JIT (it's trivial work per step). The BOLD signal sampling can happen in Python `run_network` since it's just reading the state at period boundaries.

**Final approach**:
- JIT (`network_chunk`): advances Balloon ODE state every step. No BOLD signal computation in JIT.
- Python (`run_network`): after each chunk, checks if we crossed a Bold period boundary, computes BOLD signal from the current (s,f,v,q) state, appends to output.
- Python (`_apply_monitors`): for Bold monitor, uses the pre-computed BOLD samples from `run_network` instead of the HRF convolution.

This keeps the JIT changes minimal and the BOLD signal computation trivial in Python.

### Data Flow

```
network_chunk (JIT):
  - neural integration (existing)
  - Balloon ODE step: bold_state += bold_dfun(bold_state, x) * dt  (per voi, per node)
  - returns with updated bold_state (in-place mutation)

run_network (generated Python):
  - allocates bold_state[(n_voi, 4, n_nodes)] per subnet, initialized to [0,1,1,1]
  - passes to network_chunk
  - after each chunk: check if step crossed bold period
    - if yes: compute BOLD = v0*(k1*(1-q) + k2*(1-q/v) + k3*(1-v))
    - append (time, bold_value) to bold_outputs
  - returns bold_samples alongside existing outputs

_run_compiled (Python backend):
  - detects Bold monitor in monitors list
  - extracts Bold parameters (tau_s, tau_f, etc.) and period
  - passes to run_network
  - collects bold_samples from output

_apply_monitors (Python):
  - Bold branch: uses pre-computed bold_samples from monitor_data
  - Removes _BOLD_STATE dict entirely
```

### Template Changes

#### `network_chunk` additions:
- New params per subnet:
  - `bold_state`: `(n_voi, 4, n_nodes)` float32 — [s, f, v, q]
  - `bold_params`: `(9,)` float32 — [recip_tau_s, recip_tau_f, recip_tau_o, recip_alpha, e0, recip_e0, k1, k2, k3]
  - `bold_buf`: `(n_voi, 4, n_nodes)` float32 — Heun intermediate
- New code after tavg accumulation:
  ```
  # Bold Balloon model step
  # x = voi[0] value summed over modes (neural drive)
  for vi in range(n_voi):
    for ni in range(n_nodes):
      x = 0.0
      for mi in range(n_modes):
        x += <voi_value>
      # Balloon dfun
      s = bold_state[vi, 0, ni]
      f = bold_state[vi, 1, ni]
      v = bold_state[vi, 2, ni]
      q = bold_state[vi, 3, ni]
      ds = x - bp[0]*s - bp[1]*(f - 1)
      df = s
      dv = bp[2]*(f - v**bp[3])
      dq = bp[2]*(f*(1 - (1 - bp[4])**(1/f))*bp[5] - v**bp[3]*(q/v))
      bold_buf[vi, 0, ni] = ds
      bold_buf[vi, 1, ni] = df
      bold_buf[vi, 2, ni] = dv
      bold_buf[vi, 3, ni] = dq
  # Euler step: bold_state += dt * bold_buf
  for idx in range(n_voi * 4 * n_nodes):
    flat_state[idx] += dt * flat_buf[idx]
  ```

Wait, this is getting complicated. Let me use Euler (not Heun) for Balloon since it's simpler and the timescale is slow:

  ```
  for vi in range(n_voi):
    for ni in range(n_nodes):
      x = <voi value summed over modes>
      s, f, v, q = bold_state[vi, 0, ni], ...
      ds = x - rtau_s*s - rtau_f*(f-1)
      df = s
      dv = rtau_o*(f - v**ra)
      dq = rtau_o*(f*(1-(1-e0)**(1/f))*re0 - v**ra*(q/v))
      bold_state[vi, 0, ni] += dt*ds
      bold_state[vi, 1, ni] += dt*df
      bold_state[vi, 2, ni] += dt*dv
      bold_state[vi, 3, ni] += dt*dq
  ```

This is simple, compact, and correct for a slow hemodynamic model.

#### `run_network` additions:
- Allocate `bold_state` per subnet
- Pass `bold_state` and `bold_params` to `network_chunk`
- Track `bold_step_counter`
- After each chunk:
  - Compute BOLD signal if period boundary crossed
  - Append to `bold_outputs`
- Return `bold_outputs` alongside existing outputs

### Parameter Passing
Bold parameters are passed as a float32 array of shape `(9,)`:
```
[1/tau_s, 1/tau_f, 1/tau_o, 1/alpha, e0, 1/e0, k1, k2, k3]
```
Plus `v0` as a separate scalar (used for BOLD signal computation, not dfun).

Plus `bold_istep` as int32 (steps between BOLD samples).

### State Persistence Across Calls
`bold_state` arrays are allocated in `run_network` and persist across chunks within a single `run_network` call. But across `run()` calls, the state needs to persist.

**Solution**: Store `bold_state` in the `CompiledNetworkFn` instance (like `initial_states`). Or store it in the backend's analysis object.

Actually, the simplest approach: `run_network` receives `bold_state` as a parameter (pre-allocated). `_run_compiled` manages it, creating it on first call and reusing it on subsequent calls. Store it alongside `initial_states` / `src_bufs`.

**Storage**: In `_run_compiled`, check if `bold_state` exists in `_initial_buffers` (or a new dict). If not, create and initialize to [0,1,1,1]. Pass to `run_network`. On return, save updated state back.

Actually, looking at the existing code, `_initial_buffers` stores source buffers for coupling. We can use a new mechanism. Let me check how states persist...

The `CompiledNetworkFn.run()` method manages state via `self._initial_states`. We can add `self._bold_state` similarly.

### Multiple Bold Monitors
Only one Bold monitor is typically used. If multiple are passed, we use the first one's parameters. This matches the current behavior (HRF params are per-monitor but only one is used).

### No Bold Monitor Case
When no Bold monitor is in the monitor list, pass empty arrays (shape (0, 4, 0)) and the JIT loops over range(0) — zero cost.

## Implementation Order

1. Add `bold_dfun` as inline code in the template's `network_chunk`
2. Add Bold state arrays to `network_chunk` signature and body
3. Add Bold state allocation and sampling in `run_network`
4. Update `_run_compiled` to detect Bold monitors, pass params, manage state
5. Update `_apply_monitors` to use JIT Bold output
6. Remove `_BOLD_STATE` dict
7. Update tests
8. Full test suite
9. Code review

## Risk Assessment
- **Low risk**: The Balloon model is self-contained, doesn't affect neural integration
- **Medium risk**: Numerical differences from Euler vs TVB's HRF convolution — expected and acceptable
- **Backward compat**: BOLD signal values will differ from HRF approach (different model), but timing and shapes should match
