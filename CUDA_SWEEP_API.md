# CUDA Sweep API — Quick Reference

## 1. Installation

- NVIDIA GPU with Compute Capability ≥ 7.0
- CUDA Toolkit 12.x
- `numba >= 0.65` with CUDA support
- Verify: `python -c "from numba import cuda; cuda.detect()"`

## 2. Basic Usage

```python
from tvb.simulator.backend.nb_hybrid_cuda_sweep_backend import NbHybridCUDASweepBackend

backend = NbHybridCUDASweepBackend()
compiled = backend.compile_sweep(network_set)
result = compiled.run(nstep=1000, sweep_values=values)
# result['tavg'], result['merged_tavg'], result['snapshot']
```

Convenience one-liner:
```python
result = backend.run_sweep(network_set, sweep_values=values, nstep=1000)
```

## 3. Monitors

| `monitor_type` | Mode | Output shape per subnet |
|---------------|------|------------------------|
| `0` / `"tavg"` | Temporal average (default) | `(n_sweeps, n_voi, N, n_modes)` |
| `1` / `"raw"` | Full step-by-step output | `(n_sweeps, nstep, n_voi, N, n_modes)` |
| `2` / `"subsample"` | Periodic subsample | `(n_sweeps, nstep//period, n_voi, N, n_modes)` |
| `bold_period=ms` | Balloon-Windkessel BOLD | `(n_sweeps, n_bold_samples, n_voi, N)` |

```python
result = compiled.run(nstep=1000, sweep_values=values, monitor_type='raw')
result = compiled.run(..., bold_period=2000.0)   # 2 s BOLD sampling
```

## 4. Sweep Descriptors

Default descriptor sweeps the first cfun parameter of the first projection:

```python
# Coupling-function sweep
descriptor = [{'type': 'cfun', 'projection': 'p1', 'param_idx': 0}]
compiled = backend.compile_sweep(network_set, sweep_descriptor=descriptor)

# Model-parameter sweep
descriptor = [{'type': 'model', 'subnet': 'cortex', 'param': 'tau'}]

# 2-D sweep (cfun + model)
descriptor = [
    {'type': 'cfun', 'projection': 'p1', 'param_idx': 0},
    {'type': 'model', 'subnet': 'cortex', 'param': 'a'},
]
result = compiled.run(nstep=1000, sweep_values=values_2d)
```

## 5. Chunking, Batching & Snapshots

```python
result = compiled.run(
    nstep=10000,
    sweep_values=values,
    chunk_size=1000,          # launch kernel in 1k-step chunks
    max_batch_sweeps=None,     # auto-detect from free VRAM
    snapshot=None,            # or pass previous result['snapshot'] to resume
)
```

- `chunk_size` — splits long runs into smaller kernel launches for pause/resume.
- `max_batch_sweeps` — `None` uses `0.8 × free_VRAM / bytes_per_sweep`.
- `snapshot` — dict with `'states'`, `'srcbufs'`, `'step_offset'`, and optionally `'bold_states'`.

## 6. Zerlaut Models

Zerlaut models automatically select a custom CUDA dfun template (`nb-zerlaut-sweep-cuda.py.mako`). No extra configuration required.

```python
from tvb.simulator.models import ZerlautAdaptationFirstOrder
# compile_sweep() detects _nb_hybrid_custom_template and routes to the Zerlaut kernel
```

## 7. Memory Estimation

`max_batch_sweeps=None` triggers automatic VRAM budgeting:

```python
budget = 0.8 * cuda.current_context().get_memory_info()[0]
bytes_per_sweep = state + srcbuf + tavg + noise + raw + BOLD  # auto-estimated
max_batch_sweeps = max(1, budget // bytes_per_sweep)
```

Override manually for reproducibility:
```python
compiled.run(..., max_batch_sweeps=256)
```

## 8. CPU Sweep Fallback

For small parameter grids where GPU overhead dominates, use the CPU backend in a loop:

```python
from tvb.simulator.backend.nb_hybrid import NbHybridBackend

cpu = NbHybridBackend()
fn = cpu.compile(network_set)
results = []
for v in sweep_values:
    # update network_set parameter manually
    results.append(fn.run(nstep=1000))
```

## 9. New Monitor Types

Returned in the `result` dict regardless of `monitor_type`:

| Key | Monitor | Description |
|-----|---------|-------------|
| `ctavg` | AfferentCoupling | Temporally averaged coupling input `(n_sweeps, n_cvar, N, n_modes)` |
| `spatial_tavg` | SpatialAverage | Parcellation average `(n_sweeps, n_voi, n_areas, 1)` |
| `proj_tavg` | Projection (EEG/MEG/iEEG) | Sensor projection `(n_sweeps, n_voi, n_sensors, 1)` |
| `ga_avg` | GlobalAverage | Mean across nodes if `global_average=True` |

Configure via the `monitors` dict:

```python
monitors = {
    'spatial_mean': {'cortex': spatial_mean_matrix},   # (n_areas, N)
    'gain': {'cortex': gain_matrix},                     # (n_sensors, N)
}
result = compiled.run(..., monitors=monitors,
                      node_indices={'cortex': indices},
                      global_average=True)
```

- `node_indices` — merges subnet outputs into connectome order instead of concatenation.

## 10. Throughput

Enable `verbose=True` to print wall-clock and throughput:

```python
result = compiled.run(..., verbose=True)
# [cuda_sweep] total kernel time: 245.8 ms
# [cuda_sweep] throughput: 436.0 kstep/s
```

Throughput is reported as **kstep/s** (`n_sweeps × nstep / elapsed / 1000`).
