# Numba Backend for Hybrid Simulator

## Implementation Plan

### Overview

This document outlines the implementation of a Numba-accelerated backend for the TVB Hybrid Simulator. The Hybrid Simulator differs from the standard TVB simulator in that it manages multiple subnetworks, each with potentially different models, integrators, and internal connectivity. The key novel components requiring code generation are:

1. **Inter-subnetwork projections** — delayed sparse coupling between subnetworks
2. **NetworkSet orchestration** — coordinating all subnetworks and projections
3. **Intra-subnetwork projections** — within-subnetwork connectivity (simpler, but still benefits from Numba)

The model dfun generation can borrow from existing templates (specifically for MPR model, the only model with code-gen support at this time). Integrator schemes can similarly be reused from the existing Numba backend templates.

**Constraints for initial version:**
- **Model support**: MPR (MontbrioPazoRoxin) only — this is the only model with existing Numba dfun templates. Other models will be added as code-gen templates are developed for them.
- **Same dt**: All subnetworks must share the same integration time step. Sub-stepping is not supported (nor is it in the pure Python hybrid).
- **Deterministic integrators only**: Stochastic integrators are deferred.
- **Per-projection history buffers**: Each projection owns its own circular buffer (matching the pure Python implementation). Shared-per-source buffers are a future optimisation.

---

### 1. Architecture

#### 1.1 New Backend Class

Create `tvb/simulator/backend/nb_hybrid.py`:

```python
class NbHybridBackend(MakoUtilMix):
    """Numba backend for hybrid simulator with multi-subnetwork support.

    Inherits from MakoUtilMix (not NpBackend) following the same pattern
    as NbMPRBackend — we need template rendering + exec but have a
    completely different entry point (run_network taking a NetworkSet
    instead of run_sim taking a Simulator).
    """

    def run_network(self, network_set, nstep, print_source=False):
        """Run hybrid simulation using code generation.

        Returns list of (times, data) tuples matching Simulator.run() format.
        """
        # 1. Analyze network topology
        # 2. Generate code for projections and network orchestration
        # 3. Compile and execute
        # 4. Return (times, data) per monitor, matching Simulator.run() output
```

#### 1.2 Template Files

New templates in `tvb/simulator/backend/templates/`:

| Template | Purpose |
|----------|---------|
| `nb-hybrid-sim.py.mako` | Main simulation loop, NetworkSet orchestration |
| `nb-hybrid-inter-coupling.py.mako` | Inter-subnetwork projection (sparse, delayed) |
| `nb-hybrid-intra-coupling.py.mako` | Intra-subnetwork projection (sparse) |
| `nb-hybrid-state-update.py.mako` | Model dfun + integrator step per subnetwork |

Each template generates `@nb.njit` functions that are independently testable.
Numba handles inlining of small njit functions into callers automatically,
so separating concerns across templates does not sacrifice performance.

Reusable existing templates:
- `nb-dfuns.py.mako` → Generate MPR dfun (via wrapper)
- `nb-integrate.py.mako` → Integration schemes (Heun, Euler, etc.)

---

### 2. Data Flow

#### 2.1 Per-Step Data Flow

Each simulation step (for each timestep `t`):

```
1. INTER-PROJECTION PHASE (all InterProjections)
   For each InterProjection:
   a. Read delayed state from source history buffer:
      For each non-zero weight entry k: x_src[t - 1 - idelays[k]]
      (idelays is per-connection, not per-source-node)
   b. Apply sparse weights: w_k * x_src
   c. Sum per target node (CSR row reduction)
   d. Apply cfun.pre() on summed input (AFTER weighted sum, BEFORE scale)
   e. Apply scale factor
   f. Apply cfun.post() (AFTER scale)
   g. Apply mode_map transformation: result @ mode_map
      (mode_map shape: n_src_modes × n_tgt_modes)
   h. Apply target_scales (per-target-cvar, if provided)
   i. Accumulate into target coupling array (3 cvar-mapping cases)

2. SUBNETWORK PHASE (all Subnetworks)
   For each Subnetwork:
   a. Collect inter-projection coupling from (1)
   b. Compute intra-projection coupling (same pipeline, identity mode_map)
   c. Add stimulus (if any)
   d. Run integrator step:
      i.   Evaluate model dfun (MPR)
      ii.  Apply coupling (inter + intra)
      iii. Apply integration scheme (Heun/Euler/etc.)
      iv.  Apply boundary conditions
   e. Update ALL projection buffers with post-integration state
   f. Record to monitors (temporal average accumulation)

3. MONITOR PHASE (every N steps, at chunk boundary)
   a. Extract temporal average from accumulator
   b. Push to Python-side monitors
```

#### 2.2 Chunked Execution Model

The overall execution uses a two-level loop:

```
outer Python loop (per chunk, e.g. one monitor period):
    - pre-compute stimulus batch for this chunk (if needed)
    - call inner @njit loop for chunk_size steps
    - extract temporal average from accumulator
    - update Python-side monitors
    - yield (times, data) per monitor
```

This enables:
- Bounded-memory temporal averaging inside the fast @njit loop
- Periodic stimulus evaluation (batch or code-generated)
- Standard Python monitor interface at chunk boundaries

---

### 3. Inter-Projection: Sparse Delayed Coupling

This is the primary novel component. The `BaseProjection.apply()` method in pure Python performs:

```python
# Pseudo-code matching the actual base_projection.py pipeline
def apply_inter_projection(history_buffer, weights_csr, idelays, mode_map,
                           source_cvar, target_cvar, scale, target_scales, cfun, horizon, t):
    # history_buffer: (n_vars_src, n_nodes_src, n_modes_src, horizon)
    # weights_csr: CSR (n_nodes_tgt, n_nodes_src)
    # idelays: (nnz,) — one delay per non-zero connection entry
    # mode_map: (n_modes_src, n_modes_tgt)
    # scale: scalar
    # target_scales: (n_target_cvar,) or None

    # 1. Compute time indices for each non-zero weight entry
    #    Uses t-1 to match classic TVB: coupling at step t uses state from t-1-delay
    time_indices = (t - 1 - idelays + horizon) % horizon  # shape (nnz,)

    # 2. Gather delayed states from buffer
    #    delayed_states[ic, k, m] = buffer[source_cvar[ic], src_node_k, m, time_k]
    delayed_states = history_buffer[
        source_cvar[:, None],     # (n_src_cvar, 1)
        weights.indices,           # (nnz,) source node per entry
        :,                         # all source modes
        time_indices,              # (nnz,) per-entry time index
    ]
    # Result: (n_src_cvar, nnz, n_src_modes)

    # 3. Apply sparse weights element-wise
    weighted = weights.data[None, :, None] * delayed_states
    # Result: (n_src_cvar, nnz, n_src_modes)

    # 4. Sum per target node (CSR row reduction)
    summed = np.add.reduceat(weighted, weights.indptr[:-1], axis=1)
    # Result: (n_src_cvar, n_target_nodes, n_src_modes)

    # 5. Apply cfun.pre() — AFTER weighted sum, BEFORE scale
    if cfun is not None:
        summed = cfun.pre(summed)

    # 6. Apply scale
    scaled = scale * summed

    # 7. Apply cfun.post() — AFTER scale
    if cfun is not None:
        scaled = cfun.post(scaled)

    # 8. Apply mode mapping
    #    mode_map shape: (n_src_modes, n_tgt_modes)
    aff = scaled @ mode_map
    # Result: (n_src_cvar, n_target_nodes, n_tgt_modes)

    # 9. Accumulate into target with cvar mapping (3 cases):
    if target_cvar.size == 1:
        # Many source cvars → one target cvar: sum along axis 0
        summed_aff = aff.sum(axis=0)
        if target_scales is not None:
            summed_aff *= target_scales[0]
        tgt[target_cvar[0], :, :] += summed_aff

    elif source_cvar.size == 1:
        # One source cvar → many target cvars: broadcast
        squeezed_aff = aff[0]
        if target_scales is not None:
            squeezed_aff = squeezed_aff * target_scales[:, None, None]
        tgt[target_cvar, :, :] += squeezed_aff

    elif source_cvar.size == target_cvar.size:
        # N-to-N element-wise mapping
        if target_scales is not None:
            aff *= target_scales[:, None, None]
        tgt[target_cvar, :, :] += aff

    else:
        raise ValueError("Unsupported cvar mapping")
```

#### 3.1 Numba Template for Inter-Coupling

The Numba version replaces vectorized NumPy ops with explicit loops suitable for `@nb.njit`:

```python
# nb-hybrid-inter-coupling.py.mako

@nb.njit(inline="always")
def compute_inter_coupling_${proj_name}(
    history_buffer,      # (n_vars_src, n_nodes_src, n_modes_src, horizon)
    weights_data,        # (nnz,) CSR data
    weights_indices,     # (nnz,) CSR column indices
    weights_indptr,      # (n_tgt_nodes + 1,) CSR row pointers
    idelays,             # (nnz,) integer delay per non-zero entry
    mode_map,            # (n_modes_src, n_modes_tgt)
    source_cvar,         # (n_src_cvar,) indices into source state vars
    target_cvar,         # (n_tgt_cvar,) indices into target coupling vars
    scale,               # float
    target_scales,       # (n_tgt_cvar,) or empty array if not used
    cfun_a, cfun_b,      # coupling function params (Linear: a*x+b)
    horizon,             # int
    t,                   # int, current step
    target,              # (n_cvar_tgt, n_nodes_tgt, n_modes_tgt) — INPLACE
):
    n_src_cvar = source_cvar.shape[0]
    n_tgt_cvar = target_cvar.shape[0]
    n_tgt_nodes = weights_indptr.shape[0] - 1
    n_modes_src = mode_map.shape[0]
    n_modes_tgt = mode_map.shape[1]
    has_target_scales = target_scales.shape[0] > 0

    # Allocate work buffers
    # summed_input: per target node, per source cvar, per source mode
    for j in range(n_tgt_nodes):
        row_start = weights_indptr[j]
        row_end = weights_indptr[j + 1]

        for ic in range(n_src_cvar):
            cv = source_cvar[ic]

            # Weighted sum over source nodes (CSR row)
            for m_src in range(n_modes_src):
                wsum = 0.0
                for ptr in range(row_start, row_end):
                    src_node = weights_indices[ptr]
                    w = weights_data[ptr]
                    delay = idelays[ptr]
                    buf_idx = (t - 1 - delay + horizon) % horizon
                    wsum += w * history_buffer[cv, src_node, m_src, buf_idx]

                # Apply cfun.pre (Linear: identity by default, or a*x+b)
                # cfun.post applied after scale
                ## pre (before scale):
                # wsum = wsum  (identity for Linear pre)

                # Apply scale
                wsum = scale * wsum

                # Apply cfun.post (Linear: a*x+b)
                wsum = cfun_a * wsum + cfun_b

                # Apply mode_map and accumulate into target modes
                for m_tgt in range(n_modes_tgt):
                    contrib = wsum * mode_map[m_src, m_tgt]

                    # Cvar mapping + target_scales
                    ## (template generates the appropriate branch based on
                    ##  n_src_cvar vs n_tgt_cvar relationship at code-gen time)
    % if cvar_mapping == '1_to_1' or cvar_mapping == 'n_to_n':
                    ts = target_scales[ic] if has_target_scales else 1.0
                    target[target_cvar[ic], j, m_tgt] += ts * contrib
    % elif cvar_mapping == 'many_to_1':
                    ts = target_scales[0] if has_target_scales else 1.0
                    target[target_cvar[0], j, m_tgt] += ts * contrib
    % elif cvar_mapping == '1_to_many':
                    for itc in range(n_tgt_cvar):
                        ts = target_scales[itc] if has_target_scales else 1.0
                        target[target_cvar[itc], j, m_tgt] += ts * contrib
    % endif
```

**Note on epsilon trick**: The CSR matrices will contain structural zeros from the epsilon trick
used in `BaseProjection.__init__()`. In the Numba loops these contribute zero to the weighted
sum and do not need to be stripped. This preserves identical CSR structure between the Python
and Numba paths, enabling direct numerical comparison for testing.

---

### 4. Intra-Projection: Within-Subnetwork Coupling

Intra-projection uses the same `BaseProjection.apply()` pipeline but with:
- Identity mode_map (source and target share the same subnetwork)
- Same delay mechanism (idelays per connection, circular buffer)
- Its own history buffer per projection

The template is structurally identical to the inter-projection template.
At code generation time, the cvar mapping mode and coupling function are
resolved statically, so the same template can serve both with appropriate
parameters.

```python
# nb-hybrid-intra-coupling.py.mako — same structure as inter, with identity mode_map

@nb.njit(inline="always")
def compute_intra_coupling_${proj_name}(
    history_buffer,      # (n_vars, n_nodes, n_modes, horizon)
    weights_data,        # (nnz,) CSR data
    weights_indices,     # (nnz,) CSR column indices
    weights_indptr,      # (n_nodes + 1,) CSR row pointers
    idelays,             # (nnz,) integer delay per non-zero entry
    source_cvar,         # (n_src_cvar,)
    target_cvar,         # (n_tgt_cvar,)
    scale,               # float
    target_scales,       # (n_tgt_cvar,) or empty
    cfun_a, cfun_b,      # coupling function params
    horizon,             # int
    t,                   # int, current step
    target,              # (n_cvar, n_nodes, n_modes) — INPLACE
):
    """Intra-subnetwork coupling. Same pipeline as inter but mode_map=identity."""
    n_nodes = weights_indptr.shape[0] - 1
    n_modes = history_buffer.shape[2]
    n_src_cvar = source_cvar.shape[0]

    for j in range(n_nodes):
        row_start = weights_indptr[j]
        row_end = weights_indptr[j + 1]

        for ic in range(n_src_cvar):
            cv = source_cvar[ic]

            for m in range(n_modes):
                wsum = 0.0
                for ptr in range(row_start, row_end):
                    src_node = weights_indices[ptr]
                    w = weights_data[ptr]
                    delay = idelays[ptr]
                    buf_idx = (t - 1 - delay + horizon) % horizon
                    wsum += w * history_buffer[cv, src_node, m, buf_idx]

                # cfun.pre (identity for Linear)
                wsum = scale * wsum
                wsum = cfun_a * wsum + cfun_b
                # cfun.post

                # Identity mode_map: m_src == m_tgt, so accumulate directly
                ## cvar mapping branch (generated at code-gen time)
    % if cvar_mapping == '1_to_1' or cvar_mapping == 'n_to_n':
                ts = target_scales[ic] if has_target_scales else 1.0
                target[target_cvar[ic], j, m] += ts * wsum
    % elif cvar_mapping == 'many_to_1':
                ts = target_scales[0] if has_target_scales else 1.0
                target[target_cvar[0], j, m] += ts * wsum
    % elif cvar_mapping == '1_to_many':
                for itc in range(n_tgt_cvar):
                    ts = target_scales[itc] if has_target_scales else 1.0
                    target[target_cvar[itc], j, m] += ts * wsum
    % endif
```

**Note**: Since both inter and intra projections follow the same pipeline, a single
parameterised template could serve both, with `mode_map` passed as identity for intra.
Whether to merge or keep separate is an implementation-time decision — separate templates
are easier to read; a merged template avoids duplication.

---

### 5. NetworkSet Orchestration Template

The main simulation uses a two-level loop: an outer Python loop per chunk and
an inner `@nb.njit` loop for the fast time-stepping.

```python
# nb-hybrid-sim.py.mako — inner @njit kernel

<%include file="nb-hybrid-inter-coupling.py.mako" />
<%include file="nb-hybrid-intra-coupling.py.mako" />
<%include file="nb-dfuns.py.mako" />
<%include file="nb-integrate.py.mako" />

@nb.njit(inline="always")
def update_buffer(buffer, state, t, horizon):
    """Write current state into circular buffer slot."""
    buf_idx = t % horizon
    buffer[..., buf_idx] = state

@nb.njit
def network_chunk(
    nstep,               # steps in this chunk
    t_start,             # global step offset
    dt,                  # shared dt

    # Per-subnetwork state arrays
    % for sn in subnets:
    ${sn.name}_state,    # (n_vars, n_nodes, n_modes) float32
    ${sn.name}_parmat,   # model parameter matrix
    % endfor

    # Per-projection data (inter + intra)
    % for p in all_projections:
    ${p.name}_buffer,    # (n_vars_src, n_nodes_src, n_modes_src, horizon)
    ${p.name}_w_data, ${p.name}_w_indices, ${p.name}_w_indptr,
    ${p.name}_idelays,
    ${p.name}_mode_map,
    ${p.name}_source_cvar, ${p.name}_target_cvar,
    ${p.name}_scale,
    ${p.name}_target_scales,
    ${p.name}_cfun_a, ${p.name}_cfun_b,
    ${p.name}_horizon,
    % endfor

    # Stimulus arrays (pre-computed for this chunk)
    % for stim in stimuli:
    ${stim.name}_data,   # (n_cvar, n_nodes, n_modes, chunk_size) or empty
    % endfor

    # Temporal average accumulators
    % for sn in subnets:
    ${sn.name}_tavg_acc, # (n_voi, n_nodes, n_modes) running sum
    % endfor
    tavg_count,          # number of steps accumulated
):
    """Inner fast loop for one chunk of simulation steps."""

    for t_local in range(nstep):
        t = t_start + t_local

        # ==================================================
        # STEP 1: Zero coupling arrays
        # ==================================================
        % for sn in subnets:
        ${sn.name}_coupling = np.zeros((...))  # (n_cvar, n_nodes, n_modes)
        % endfor

        # ==================================================
        # STEP 2: Compute all inter-projection couplings
        # ==================================================
        % for p in inter_projections:
        compute_inter_coupling_${p.name}(
            ${p.source}_buffer,
            ${p.name}_w_data, ${p.name}_w_indices, ${p.name}_w_indptr,
            ${p.name}_idelays,
            ${p.name}_mode_map,
            ${p.name}_source_cvar, ${p.name}_target_cvar,
            ${p.name}_scale, ${p.name}_target_scales,
            ${p.name}_cfun_a, ${p.name}_cfun_b,
            ${p.name}_horizon, t,
            ${p.target}_coupling,
        )
        % endfor

        # ==================================================
        # STEP 3: Compute intra-projection couplings
        # ==================================================
        % for p in intra_projections:
        compute_intra_coupling_${p.name}(
            ${p.name}_buffer,
            ${p.name}_w_data, ${p.name}_w_indices, ${p.name}_w_indptr,
            ${p.name}_idelays,
            ${p.name}_source_cvar, ${p.name}_target_cvar,
            ${p.name}_scale, ${p.name}_target_scales,
            ${p.name}_cfun_a, ${p.name}_cfun_b,
            ${p.name}_horizon, t,
            ${p.target}_coupling,
        )
        % endfor

        # ==================================================
        # STEP 4: Add stimulus (if any)
        # ==================================================
        % for stim in stimuli:
        ${stim.target}_coupling += ${stim.name}_data[..., t_local]
        % endfor

        # ==================================================
        # STEP 5: Integrate each subnetwork
        # ==================================================
        % for sn in subnets:
        ${sn.name}_state = integrate_${sn.integrator}_${sn.model}(
            ${sn.name}_state,
            ${sn.name}_coupling,
            ${sn.name}_parmat,
            dt,
        )
        ## Apply boundary conditions
        bound_state(${sn.name}_state, ...)
        % endfor

        # ==================================================
        # STEP 6: Update ALL projection buffers
        # ==================================================
        % for p in inter_projections:
        update_buffer(${p.name}_buffer, ${p.source}_state, t, ${p.name}_horizon)
        % endfor
        % for p in intra_projections:
        update_buffer(${p.name}_buffer, ${p.target}_state, t, ${p.name}_horizon)
        % endfor

        # ==================================================
        # STEP 7: Accumulate temporal average
        # ==================================================
        % for sn in subnets:
        ## observe() extracts VOI from state
        ${sn.name}_tavg_acc += observe_${sn.name}(${sn.name}_state)
        % endfor
        tavg_count[0] += 1
```

#### 5.1 Outer Python Loop

```python
# In NbHybridBackend.run_network():
def run_network(self, network_set, nstep, print_source=False):
    # ... code generation and compilation ...

    chunk_size = monitor_period  # or configurable
    results = {sn.name: [] for sn in network_set.subnets}

    for chunk_start in range(0, nstep, chunk_size):
        chunk_steps = min(chunk_size, nstep - chunk_start)

        # Pre-compute stimulus for this chunk (if needed)
        stim_data = precompute_stimuli(network_set, chunk_start, chunk_steps)

        # Zero temporal average accumulators
        reset_tavg_accumulators(...)

        # Call inner @njit kernel
        network_chunk(chunk_steps, chunk_start, dt, ...)

        # Extract temporal average and push to monitors
        for sn in network_set.subnets:
            tavg = tavg_acc[sn.name] / tavg_count
            for mon in sn.monitors:
                mon.record(chunk_start + chunk_steps, tavg)
            results[sn.name].append((times, tavg.copy()))

    # Return in Simulator.run() format: list of (times, data) tuples
    return format_output(results, network_set)
```

---

### 6. History Buffer Management

Each projection owns a circular history buffer, matching the pure Python implementation.
The buffer stores the **full source state** (not just coupling variables) so that
different projections from the same source can read different cvars.

**Buffer shape**: `(n_vars_src, n_nodes_src, n_modes_src, horizon)`
where `horizon = max_delay + 1` (minimum 1).

**Circular indexing**:
```python
# Write current state at step t:
buffer[..., t % horizon] = current_state

# Read delayed state for connection k at step t:
# Uses t-1 convention: coupling at step t uses state from step t-1-delay
buf_idx = (t - 1 - idelays[k] + horizon) % horizon
delayed_val = buffer[cvar, src_node, mode, buf_idx]
```

**Pre-fill**: Before simulation starts, all buffer slots are filled with the
initial state (matching `NetworkSet.init_projection_buffers()`):
```python
for slot in range(horizon):
    buffer[..., slot] = initial_state
```

**Design note**: The existing `nb-sim.py.mako` uses a *linear* (non-circular)
buffer of shape `(nsvar, nnode, horizon + nstep)` — no modulo needed but memory
grows with simulation length. We choose circular buffers for the hybrid backend
because hybrid simulations may be long-running and memory-bounded is preferred.
The trade-off is slightly more complex indexing (one modulo per delay lookup)
which Numba handles efficiently.

**Future optimisation**: Multiple projections from the same source subnetwork
currently each maintain a separate buffer. A shared-per-source buffer (sized to
the max horizon across all outgoing projections) would save memory and redundant
writes. Deferred because it adds complexity in managing different cvar subsets
across projections.

---

### 7. Implementation Phases

#### Phase 1: Infrastructure

- [ ] Create `NbHybridBackend(MakoUtilMix)` class with `run_network()` entry point
- [ ] Add template file structure
- [ ] Implement `NetworkAnalysis` — inspect NetworkSet and extract all metadata needed for code generation
- [ ] Set up code generation pipeline: template rendering → autopep8 → exec/module import
- [ ] Enforce same-dt constraint: validate all subnetworks share `scheme.dt`

#### Phase 2: Inter-Projection Templates (Priority: HIGHEST)

- [ ] `nb-hybrid-inter-coupling.py.mako` — sparse delayed coupling with circular buffer
- [ ] CSR iteration with per-connection delays (`idelays[ptr]`)
- [ ] Mode_map transformation (shape: `n_src_modes × n_tgt_modes`)
- [ ] Coupling function pipeline: `pre()` → `scale` → `post()` (correct ordering)
- [ ] Three-way cvar mapping: 1-to-1/N-to-N, many-to-1, 1-to-many (static branch at codegen time)
- [ ] `target_scales` support (per-target-cvar scaling)
- [ ] Coupling functions: Linear (`a*x+b`), Scaling (`a*x`), identity (no cfun)

#### Phase 3: Intra-Projection Templates (Priority: HIGH)

- [ ] `nb-hybrid-intra-coupling.py.mako` — same pipeline as inter, identity mode_map
- [ ] Decide: shared template with inter or separate (implementation-time decision)

#### Phase 4: NetworkSet Loop (Priority: HIGHEST)

- [ ] `nb-hybrid-sim.py.mako` — inner `@njit` kernel for chunk of steps
- [ ] Wire all inter + intra projections
- [ ] Circular buffer update after integration
- [ ] Temporal average accumulation inside fast loop
- [ ] Outer Python loop: chunk iteration, stimulus batch, monitor update

#### Phase 5: Model & Integrator Integration (Priority: HIGH)

- [ ] Reuse existing `nb-dfuns.py.mako` for MPR dfun
- [ ] Reuse existing `nb-integrate.py.mako` for Heun/Euler
- [ ] Per-subnetwork state arrays and parameter matrices
- [ ] Boundary condition application after integration

#### Phase 6: Stimulus Support (Priority: MEDIUM)

- [ ] Option A: Code-generate simple stimulus patterns (e.g., pulse train) as `@njit` functions
- [ ] Option B: Pre-compute stimulus array in Python, pass batch to `@njit` kernel
- [ ] Both options should coexist; choice per-stimulus at code-gen time

#### Phase 7: Testing & Validation (Priority: HIGH)

- [ ] Create MPR-only test fixtures (two subnetworks, inter + intra projections)
- [ ] Verify Numba output matches pure Python `NetworkSet` simulation within tolerance
- [ ] Test all three cvar mapping modes
- [ ] Test with/without delays, with/without coupling functions
- [ ] Benchmark: Numba vs pure Python speedup
- [ ] Output format: verify matches `Simulator.run()` return format

---

### 8. Key Technical Decisions

#### Decision 8.1: Sparse Matrix Format

**Choice: CSR (Compressed Sparse Row)**

- CSR is already used by TVB (via `scipy.sparse.csr_matrix`) for connectivity matrices
- Efficient for row-wise iteration (what Numba needs for per-target-node coupling)
- Numba receives the three CSR arrays (`data`, `indices`, `indptr`) as separate arguments

```python
# CSR format iteration pattern in Numba
for i in range(n_target_nodes):
    for ptr in range(indptr[i], indptr[i + 1]):
        j = indices[ptr]  # source node
        w = data[ptr]     # weight
        d = idelays[ptr]  # delay for this specific connection
        # ... computation
```

**Note on epsilon zeros**: The CSR matrices contain structural zeros from the
epsilon trick in `BaseProjection.__init__()`. These are preserved (not stripped)
to keep identical sparsity structure for testing against the Python implementation.
They contribute zero to the weighted sum.

#### Decision 8.2: Memory Layout — Separate Arrays Per Subnetwork

Each subnetwork gets its own state array:

```python
cortex_state = np.zeros((n_vars_cortex, n_nodes_cortex, n_modes_cortex), dtype=np.float32)
thalamus_state = np.zeros((n_vars_thalamus, n_nodes_thalamus, n_modes_thalamus), dtype=np.float32)
```

This is simpler than a single contiguous array and matches the hybrid framework's design.

#### Decision 8.3: Coupling Function Support

Start with **Linear**, **Scaling**, and **identity** (no cfun):

- Identity (cfun=None): `y = x` — no transformation
- Linear: `post(x) = a * x + b`, `pre(x) = x` (identity pre)
- Scaling: `post(x) = a * x`, `pre(x) = x`

These are the most common coupling functions. The pipeline order is:
1. Weighted sum (CSR reduction)
2. `cfun.pre()` — applied AFTER sum, BEFORE scale
3. Multiply by `scale`
4. `cfun.post()` — applied AFTER scale

Others (Sigmoidal, SigmoidalJansenRit from `hybrid/coupling.py`) can be added later
by extending the template with additional coupling function branches.

#### Decision 8.4: History Buffer — Circular

**Choice: Circular buffer with modular indexing**

- Shape: `(n_vars, n_nodes, n_modes, horizon)` where `horizon = max_delay + 1`
- Write: `buffer[..., t % horizon] = state`
- Read: `buffer[cv, node, mode, (t - 1 - delay + horizon) % horizon]`
- Bounded memory regardless of simulation length
- Trade-off: one modulo per delay lookup (negligible in Numba)

Alternative considered: linear buffer (as in `nb-sim.py.mako`) with shape
`(nsvar, nnode, horizon + nstep)` — simpler indexing but memory grows with
simulation length. Rejected for hybrid because simulations may be long-running.

#### Decision 8.5: Same dt Across All Subnetworks

**Enforced constraint**: All subnetworks must share identical `scheme.dt`.
Validated at code-generation time. Sub-stepping (different dt per subnetwork)
is not supported and is also not supported by the pure Python hybrid implementation.

#### Decision 8.6: Chunked Execution with Temporal Averaging

**Choice: Two-level loop**

- Inner `@njit` kernel runs `chunk_size` steps (defaults to monitor period)
- Accumulates temporal average inside the fast loop
- Outer Python loop handles: stimulus batch eval, monitor update, output formatting
- Enables bounded-memory operation and periodic Python-side callbacks

#### Decision 8.7: Output Format

**Choice: Match `Simulator.run()` format**

`run_network()` returns a list of `(times, data)` tuples, one per monitor,
matching the standard TVB `Simulator.run()` return format. This enables
drop-in replacement for testing and user code.

---

### 9. Code Generation Input Specification

To generate code, the backend needs to inspect the NetworkSet and extract:

```python
class NetworkAnalysis:
    """Analysis result used for code generation."""

    # List of subnetworks
    subnetworks: List[SubnetworkInfo]

    # List of inter-projections
    inter_projections: List[ProjectionInfo]

    # List of intra-projections (per subnetwork)
    intra_projections: Dict[subnetwork_name, List[ProjectionInfo]]

    # Global parameters
    dt: float                  # shared integration time step


class SubnetworkInfo:
    name: str                  # e.g., "cortex"
    n_vars: int                # number of state variables
    n_nodes: int               # number of nodes
    n_modes: int               # number of modes
    model_class: str           # "MPR" (only supported model for now)
    model_params: Dict         # specific model parameters
    integrator: str            # "HeunDeterministic", "EulerDeterministic", etc.
    cvar: List[int]            # coupling variable indices
    voi: List[int]             # variables of interest (for monitors)
    stimulus: Optional[...]    # stimulus info for code-gen or batch eval


class ProjectionInfo:
    """Shared info for both inter and intra projections."""
    name: str                  # e.g., "cortex_to_thalamus" or "cortex_local_0"
    source_subnet: str         # source subnetwork name
    target_subnet: str         # target subnetwork name
    source_cvar: ndarray       # (n_src_cvar,) source coupling variable indices
    target_cvar: ndarray       # (n_tgt_cvar,) target coupling variable indices
    cvar_mapping: str          # "1_to_1", "n_to_n", "many_to_1", or "1_to_many"
    weights_data: ndarray      # (nnz,) CSR data array
    weights_indices: ndarray   # (nnz,) CSR column indices
    weights_indptr: ndarray    # (n_tgt_nodes + 1,) CSR row pointers
    idelays: ndarray           # (nnz,) integer delay per non-zero connection
    mode_map: ndarray          # (n_modes_src, n_modes_tgt) — identity for intra
    horizon: int               # max_delay + 1
    scale: float               # global projection scale
    target_scales: ndarray     # (n_tgt_cvar,) per-cvar scale, or empty array
    cfun_type: str             # "none", "linear", "scaling"
    cfun_a: float              # coupling param a (1.0 if no cfun)
    cfun_b: float              # coupling param b (0.0 if no cfun)
```

---

### 10. Testing Strategy

#### 10.1 Reference Implementation

The pure Python `NetworkSet` simulation serves as the ground truth. Since the
Numba backend only supports MPR for code generation, test fixtures must use
MPR models in all subnetworks (unlike the existing hybrid tests which use
JansenRit, FHN, etc.). Dedicated MPR-only test cases will be created.

#### 10.2 Test Levels

1. **Unit tests per projection function**: Generate and call a single
   `compute_inter_coupling_*` or `compute_intra_coupling_*` function,
   compare output against `BaseProjection.apply()` on the same inputs.

2. **Unit tests per dfun/integrator**: Verify generated MPR dfun and
   Heun/Euler integrators match `model.dfun()` and `scheme.scheme()`.

3. **Integration test — full NetworkSet**: Run identical configuration
   through both `NetworkSet.step()` loop and `NbHybridBackend.run_network()`,
   compare state arrays within tolerance (`atol=1e-5`, `rtol=1e-5`).

4. **Output format test**: Verify `run_network()` returns the same
   `(times, data)` structure as `Simulator.run()`.

#### 10.3 Test Configurations

- Two MPR subnetworks (e.g., "cortex" + "thalamus")
- Inter-projections covering:
  - 1-to-1 cvar mapping
  - Many-to-1 cvar mapping
  - 1-to-many cvar mapping
- With and without delays (zero-length vs non-zero)
- With and without coupling functions (identity, Linear, Scaling)
- With and without `target_scales`
- Intra-projections (within one subnetwork)
- HeunDeterministic and EulerDeterministic integrators

#### 10.4 Benchmark Tests

- Compare wall-clock time: pure Python vs Numba for identical configuration
- Scaling: vary node count, connection density, number of subnetworks

---

### 11. Files to Create/Modify

#### New Files

| File | Purpose |
|------|---------|
| `tvb/simulator/backend/nb_hybrid.py` | `NbHybridBackend(MakoUtilMix)` — main backend class |
| `tvb/simulator/backend/templates/nb-hybrid-sim.py.mako` | Network loop template (inner @njit + outer Python) |
| `tvb/simulator/backend/templates/nb-hybrid-inter-coupling.py.mako` | Inter-projection template |
| `tvb/simulator/backend/templates/nb-hybrid-intra-coupling.py.mako` | Intra-projection template (may share with inter) |
| `tvb/simulator/backend/templates/nb-hybrid-state-update.py.mako` | Model dfun + integrator step per subnetwork |
| `tvb/tests/library/simulator/backend/test_nb_hybrid.py` | Tests for Numba hybrid backend |

#### Files to Modify

| File | Changes |
|------|---------|
| `tvb/simulator/backend/__init__.py` | Export `NbHybridBackend` |

---

### 12. Open Questions / Future Work

1. **More models**: MPR only for now; extend as code-gen templates are developed for other models
2. **Stochastic integrators**: Require noise handling (pre-drawn samples); add after deterministic works
3. **Shared-per-source history buffers**: Single buffer per source subnetwork instead of per-projection — saves memory and writes, but adds cvar-management complexity
4. **Sub-stepping (different dt per subnetwork)**: Not supported in pure Python either; requires significant loop restructuring
5. **Adaptive integrators**: Not supported in initial version
6. **More coupling functions**: Sigmoidal, SigmoidalJansenRit from `hybrid/coupling.py`
7. **`AfferentCoupling` monitors**: Record coupling instead of state — requires passing coupling data to monitor accumulator
8. **Performance tuning**: Investigate `nb.prange` for outer node loop parallelisation, loop fusion opportunities
9. **Code-generated stimuli**: For simple patterns (pulse train, sinusoidal), generate `@njit` functions; for complex stimuli, batch-evaluate in Python between chunks
10. **Merged observation mode**: When all subnetworks carry `node_indices`, place outputs in global connectome ordering

---

### 13. Summary

The key insight is that the **inter-subnetwork projection** is the novel component requiring new code generation. The inter-projection involves:

1. **Sparse CSR iteration** — iterate over non-zero weights per target node
2. **Per-connection delayed state access** — circular buffer indexing with `idelays[ptr]`
3. **Coupling function pipeline** — `pre()` → `scale` → `post()` (correct order)
4. **Mode map transformation** — `(n_src_modes, n_tgt_modes)` matrix multiply after coupling
5. **Three-way cvar mapping** — 1-to-1, many-to-1, 1-to-many (resolved statically at code-gen)
6. **`target_scales`** — optional per-target-cvar scaling on top of global `scale`

The execution model is a **two-level loop**:
- Outer Python loop per chunk: stimulus batching, monitor update, output I/O
- Inner `@njit` loop per chunk: coupling → integration → buffer update → tavg accumulation

Individual `@njit` functions (per-projection coupling, per-subnetwork dfun/integrator)
are kept separate for unit testability. Numba's inlining handles fusion automatically
when they are called from the main `@njit` kernel.

This follows the same pattern as the existing Numba backend (`NbMPRBackend` with
`MakoUtilMix` templates) but adapted for the multi-subnetwork hybrid architecture
with circular buffers and the `Simulator.run()`-compatible output format.