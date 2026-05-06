# CUDA Sweep — Remaining Items Todo

## Items
- [x] 1. Bold monitor (Balloon-Windkessel ODE)
- [x] 2. Combined dfun at 76 nodes (ReducedSet models work — was already fixed in prior session)
- [x] 3. GPU memory chunking (split large sweeps across VRAM-limited launches)

## Design Notes

### 1. Bold Monitor
- Bold state: (n_sweeps, n_voi, 4, N, n_modes) per subnet, initialized s=0,f=1,v=1,q=1
- Bold params: 9 floats [1/τ_s, 1/τ_f, 1/τ_o, 1/α, e0, 1/e0, k1, k2, k3]
- Bold voi_idx: which state vars feed the Balloon ODE
- dt_bold = dt (Euler integration per timestep)
- Sampling: when t % bold_istep == 0, compute BOLD = V0*(k1*(1-q) + k2*(1-q/v) + k3*(1-v))
- BOLD output: (n_sweeps, n_bold_samples, n_voi, N) per subnet
- Bold is only for specific timepoints so output goes to host on sample boundaries
- Bold state lives in GPU local memory per thread (cuda.local.array(4,) per voi per node)
  Actually: too large for local. Use device array (n_sweeps, n_voi, 4, N, n_modes)

### 2. Combined dfun at 76 nodes
- Error 700 = illegal address. Likely: local.array too large for registers,
  or derived matrices accessed with wrong indices.
- ReducedSetFHN: 4 vars, 3 modes, 9 derived matrices, 3 ops
- Need to test and debug. May need to move some local arrays to device global memory.

### 3. GPU Memory Chunking
- Split sweep points into batches to fit VRAM
- batch_size = min(n_sweeps, max_sweeps_that_fit)
- Outer loop over batches, inner loop is temporal chunking
- Concatenate tavg and raw outputs across batches
- Bold state must persist across batches for continuity

## Progress
(checked items are done)