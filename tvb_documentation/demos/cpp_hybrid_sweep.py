# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Multithreaded SIMD parameter sweeps with the C++ hybrid backend
#
# This notebook shows how to use the SIMD-enabled C++ hybrid backend for
# parameter sweeps on an arbitrarily complex hybrid network (multi-subnet,
# multi-mode, arbitrary model params, projection scaling), and how to compute
# data features on the monitor output for each sweep member.
#
# The workhorse is `notebook.sweep` — it batches sweep members into 8-wide SIMD
# lanes and runs lane-groups in parallel threads (the C++ kernel holds no GIL).

# %%
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

from tvb.simulator.backend.cpp_hybrid import notebook as nb

from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import HeunDeterministic

# %% [markdown]
# ## 1. Build a network (here: two coupled MPR subnetworks)
# Any hybrid `NetworkSet` works — subnets, modes, model params, projections.

# %%
def subnet(name, n, dt=0.1):
    m = MontbrioPazoRoxin(); m.configure()
    return Subnetwork(name=name, model=m, scheme=HeunDeterministic(dt=dt), nnodes=n)

A = subnet('A', 8); B = subnet('B', 8); n = 8
w = sparse.csr_matrix(np.full((n, n), 0.01, np.float64) - np.eye(n) * 0.01)
proj = InterProjection(
    source=A, target=B, source_cvar=np.array([0], np.int32),
    target_cvar=np.array([0], np.int32), weights=w,
    lengths=sparse.csr_matrix(np.zeros((n, n))), cv=1.0, dt=0.1,
    scale=2.0, cfun=Linear())
ns = NetworkSet(subnets=[A, B], projections=[proj]); ns.configure()

# valid swept-parameter names, with autocomplete on error
nb.available_sweep_params(ns)

# %% [markdown]
# ## 2. Parameter sweep (SIMD lanes + threads)
# `nstep` steps per member; `progress=True` for a tqdm bar. A parameter is
# named either by its projection alias (`coupling_scale`) or dotted
# `{subnet}.{param}`. All named parameter arrays must have the same length
# and are **paired elementwise** (lockstep zip): passing two length-40 1-D
# arrays evaluates **40 paired points** (`res.sweep_values.shape == (40, 2)`),
# not a 40x40 Cartesian cross product. To sweep a full 2-D Cartesian grid,
# pass flattened `np.meshgrid` arrays (e.g. `g, e = np.meshgrid(gs, etas);`
# `nb.sweep(..., coupling_scale=g.ravel(), **{"A.eta": e.ravel()})`).

# %%
res = nb.sweep(ns, 2000,
               coupling_scale=np.linspace(0.2, 2.0, 40),   # 40 paired members
               **{'A.eta': np.linspace(-4.0, -2.0, 40)},
               progress=True, workers=4)
print(res.backend)
print('sweep_values shape:', res.sweep_values.shape)
print('sweep values (col0=coupling_scale, col1=A.eta):')
print(res.sweep_values[:5])

# %% [markdown]
# ## 3. Inspect results
# - `res.squeezed(subnet)` → `(sweep, time, svar, node)` mean activity
# - `res.to_dataframe(svar=0)` → param + per-node columns (needs pandas)

# %%
tavg = res.squeezed(subnet='A')          # (40, 2000, 2, 8)
plt.figure(figsize=(10, 4))
plt.plot(res.sweep_values[:, 0], tavg[:, :, 0, :].mean(axis=(1, 2)))
plt.xlabel('coupling_scale'); plt.ylabel('mean firing rate r');
plt.title('Sweeping coupling_scale vs A.eta (SIMD-C++)')
plt.show()

# %% [markdown]
# ## 4. Features on monitor output
# `res.features(...)` computes data features over the per-sweep time series,
# returning `{feature: (sweep, node)}` — perfect for a feature vs. parameter
# regression. Feature names come from `tvb...features.FEATURES` (a vendored
# subset of the VBI taxonomy: mean/std/rms/entropy/zero-crossing/PSD stats...).

# %%
f = res.features(['mean', 'std', 'rms', 'zero_crossing', 'entropy',
                  'spectrum_auc'], fs=10.0, svar=0, subnet='A')
print({k: np.asarray(v).shape for k, v in f.items()})

plt.figure(figsize=(10, 4))
plt.plot(res.sweep_values[:, 0], f['rms'].mean(axis=1), label='rms')
plt.plot(res.sweep_values[:, 0], f['spectrum_auc'].mean(axis=1), label='spectrum_auc')
plt.xlabel('coupling_scale'); plt.legend(); plt.show()

# %% [markdown]
# ## 5. Direct feature computation on any monitor time series
# `features.compute_feature(ts, name)` works on any `(n_samples, leading...)`
# array (time first), e.g. a single-channel monitor output.

# %%
from tvb.simulator.backend.cpp_hybrid import features as feat
ts = np.random.default_rng(0).standard_normal((500, 8))   # 500 samples, 8 nodes
print('feature table shape (nodes x features):',
      feat.feature_table(ts, fs=10.0).shape)
print(sorted(feat.FEATURES))
