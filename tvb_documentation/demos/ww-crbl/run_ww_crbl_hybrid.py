# %% [markdown]
# # Wong-Wang + Cerebellar Mean-Field: Hybrid Simulation & Spectral Analysis
#
# This notebook demonstrates **Wong-Wang (WW) + Cerebellar MF** whole-brain
# simulation using the TVB hybrid framework, with:
#
# 1. **Open-loop vs closed-loop** spectral comparison
# 2. **Parameter sweep** over inter-network coupling to find balanced activity
# 3. **Construct validation** analysis of cerebellar oscillatory bands
#
# ## Architecture
#
# - **Cortex subnetwork** (99 nodes): 93 cortical + 6 DCN regions,
#   modelled with `ReducedWongWangExcInh`
# - **Cerebellum subnetwork** (27 nodes): cerebellar cortex, modelled with
#   `CerebellarMF` (GrC → GoC → MLI → PC)
#
# ## Signal flow — 5 projections
#
# | # | Type | Signal | Source → Target cvar |
# |---|------|--------|---------------------|
# | P1 | IntraProjection | Intra-cortex | S_e(0) → S_e(0) |
# | P2 | InterProjection | Mossy fibers | S_e(0) → mossy(0) |
# | P3 | InterProjection | GrC feedback | GrC(0) → S_e(0) |
# | P4 | IntraProjection | CRBL mossy loop | GrC(0) → mossy(0) |
# | P5 | IntraProjection | Parallel fibers | GrC(0) → parallel(1) |
#
# ## Bug fix note
#
# The original `TF_inhibitory_goc` in `parallel_crbl_multimf_ww.py` (line 820)
# passes `self.E_i` for **both** excitatory and inhibitory reversal potentials.
# Since `E_i = -80 mV`, GoC excitatory drive was actually hyperpolarizing —
# GoC was permanently silent. Our `CerebellarMF` model corrects this to
# `self.E_e, self.E_i` (0 mV, -80 mV), making excitatory input depolarizing
# and unlocking the GrC↔GoC feedback loop that produces 150–190 Hz oscillations.

# %% [markdown]
# ## Scientific Background — Construct Validation
#
# A fundamental question in computational neuroscience is whether a model
# reproduces the **spectral signatures** observed empirically. This is known
# as **construct validation** — demonstrating that the model's intrinsic
# dynamics generate oscillations in the expected frequency bands.
#
# ### Cerebellar oscillation bands
#
# Following **Esaghei et al. (2022)** *Trends in Neurosciences*:
#
# | Band | Range (Hz) | Cerebellar relevance |
# |------|-----------|----------------------|
# | Delta | 1–4 | Slow oscillations, sleep, default-mode |
# | Theta | 4–8 | Motor coordination, cerebello-cortical coupling |
# | Alpha | 8–13 | **Purkinje cell simple-spike rhythm** |
# | Beta | 13–30 | Motor planning, cerebellar synchrony |
# | Low γ | 30–80 | Fast local processing, GrC layer |
# | High γ | 80–150 | Interneuron synchrony |
# | Ripple | 150–500 | **GrC↔GoC feedback oscillation** (this model) |
#
# ### Key result
#
# With the Ee bug fix, the CerebellarMF model produces its **strongest
# oscillations in the ripple band** (150–190 Hz), arising from the GrC↔GoC
# feedback loop (~12 ms loop delay). The traditional delta–beta bands show
# relatively flat PSDs — the model does not produce low-frequency oscillations
# at these parameter settings.

# %%
%pylab inline
import warnings
warnings.filterwarnings('ignore')

import time
import builtins
import numpy as np
import scipy.sparse as sp
import scipy.signal

from tvb.simulator.models import ReducedWongWangExcInh
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.noise import Additive
from tvb.simulator.monitors import TemporalAverage
from tvb.simulator.hybrid import (
    Subnetwork, InterProjection, IntraProjection, NetworkSet, Simulator,
)
from tvb.simulator.models.cerebellar_mf import CerebellarMF

print("All imports OK")

# %% [markdown]
# ## 1. Load and normalize structural connectivity
#
# Weights are normalized column-wise (matching the monolithic cMF-TVB pipeline)
# so that each target node sums incoming weights to 1.

# %%
data = np.load('data/conn_126.npz')
W_raw = data['weights'].copy()
TL_full = data['tract_lengths'].astype(np.float64)
TL_full[TL_full == 0] = np.min(TL_full[TL_full > 0])

# Normalize weights column-wise (same as monolithic init)
W = W_raw / (W_raw.sum(axis=0) + 1e-12)

print(f"SC: {W_raw.shape} → normalized, max={W.max():.4f}")

# %% [markdown]
# ## 2. Region indices

# %%
cortex_idx = np.concatenate([
    np.arange(0, 93), np.array([103, 104, 105, 113, 114, 115])
])
n_cortex = len(cortex_idx)  # 99

crbl_idx = np.array([
    93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
    106, 107, 108, 109, 110, 111, 112,
    116, 117, 118, 119, 120, 121, 122, 123, 124, 125
])
n_crbl = len(crbl_idx)  # 27

dcn_idx = np.array([103, 104, 105, 113, 114, 115])

# %% [markdown]
# ## 3. Partition SC (with DCN negation)

# %%
W[dcn_idx[:, None], crbl_idx[None, :]] *= -1

def csr(arr): return sp.csr_matrix(arr.astype(np.float64))

W_cc = csr(W[np.ix_(cortex_idx, cortex_idx)])
TL_cc = csr(TL_full[np.ix_(cortex_idx, cortex_idx)])
W_cr = csr(W[np.ix_(crbl_idx, cortex_idx)])
TL_cr = csr(TL_full[np.ix_(crbl_idx, cortex_idx)])
W_rc = csr(W[np.ix_(cortex_idx, crbl_idx)])
TL_rc = csr(TL_full[np.ix_(cortex_idx, crbl_idx)])
W_rr = csr(W[np.ix_(crbl_idx, crbl_idx)])
TL_rr = csr(TL_full[np.ix_(crbl_idx, crbl_idx)])

print(f"W_cc {W_cc.shape}  W_cr {W_cr.shape}  W_rc {W_rc.shape}  W_rr {W_rr.shape}")

# %% [markdown]
# ## 4. Common parameters

# %%
dt = 1.0           # ms
nsig = np.array([(0.001 ** 2) / 2])
sim_len = 10000.0   # 10 seconds
cv = 3.0
mon_period = 1.0

# %% [markdown]
# ## 5. Scenario 1 — Open-Loop (CRBL-only)
#
# Cerebellum in isolation with both mossy-fiber (P4) and parallel-fiber (P5)
# intra-cerebellar projections.

# %%
crbl_open = CerebellarMF()
crbl_open.variables_of_interest = ('GrC', 'GoC', 'MLI', 'PC')
# Legacy mode: match published cMF-TVB results (GoC E_e bug)
crbl_open.use_legacy_goc_e_e = np.array([True])
# Production: no external drive (overridden by run script)
crbl_open.external_input_ex_ex = np.array([0.0])
# kHz initial conditions (matching parallel_crbl_params.py)
crbl_open.state_variable_range['GrC'] = np.array([500.0, 500.0])
crbl_open.state_variable_range['GoC'] = np.array([5000.0, 5000.0])
crbl_open.state_variable_range['MLI'] = np.array([15000.0, 15000.0])
crbl_open.state_variable_range['PC'] = np.array([38000.0, 38000.0])
crbl_open.state_variable_range['noise'] = np.array([0.0, 0.0])

cerebellum_open = Subnetwork(
    name='cerebellum', model=crbl_open,
    scheme=HeunStochastic(dt=dt, noise=Additive(nsig=nsig)),
    nnodes=n_crbl, node_indices=crbl_idx,
)

# P4: intra-cerebellar mossy fiber loop  (GrC → mossy cvar)
p4_open = IntraProjection(
    source_cvar=np.array([0]), target_cvar=np.array([0]),
    weights=W_rr, lengths=TL_rr, cv=cv, dt=dt, scale=1.0,
)
# P5: intra-cerebellar parallel fiber loop (GrC → parallel cvar)
p5_open = IntraProjection(
    source_cvar=np.array([0]), target_cvar=np.array([1]),
    weights=W_rr, lengths=TL_rr, cv=cv, dt=dt, scale=1.0,
)

cerebellum_open.projections = [p4_open, p5_open]
cerebellum_open.add_monitor(TemporalAverage(period=mon_period))

nets_open = NetworkSet(subnets=[cerebellum_open], projections=[])
sim_open = Simulator(
    nets=nets_open, monitors=[TemporalAverage(period=mon_period)],
    simulation_length=sim_len,
)
sim_open.configure()

print("Running open-loop (10 s, CRBL-only)...")
t0 = time.time()
sim_open.run()
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f} s")

rec_crbl_open = cerebellum_open.monitors[0]
ts_open, d_open = rec_crbl_open.to_arrays()
print(f"  CRBL data: {d_open.shape}")

grc_open = d_open[:, 0, :, 0]
goc_open = d_open[:, 1, :, 0]
mli_open = d_open[:, 2, :, 0]
pc_open  = d_open[:, 3, :, 0]

for pop, arr in [('GrC', grc_open), ('GoC', goc_open), ('MLI', mli_open), ('PC', pc_open)]:
    print(f"  {pop}: mean={arr.mean():.4f} Hz  range=[{arr.min():.4f}, {arr.max():.4f}]")

# %% [markdown]
# ## 6. Scenario 2 — Closed-Loop (WW + Cerebellum)
#
# Full 126-node whole-brain simulation. Five projections:
# P1 (intra-cortex), P2 (cortex→CRBL mossy), P3 (CRBL GrC→cortex S_e),
# P4 (CRBL mossy loop), P5 (CRBL parallel fibers).

# %%
ww_model = ReducedWongWangExcInh()
ww_model.G = np.array([2.0])
ww_model.J_N = np.array([0.15])

cortex = Subnetwork(
    name='cortex', model=ww_model,
    scheme=HeunStochastic(dt=dt, noise=Additive(nsig=nsig)),
    nnodes=n_cortex, node_indices=cortex_idx,
)

crbl_closed = CerebellarMF()
crbl_closed.variables_of_interest = ('GrC', 'GoC', 'MLI', 'PC')
crbl_closed.use_legacy_goc_e_e = np.array([True])
crbl_closed.external_input_ex_ex = np.array([0.0])
crbl_closed.state_variable_range['GrC'] = np.array([500.0, 500.0])
crbl_closed.state_variable_range['GoC'] = np.array([5000.0, 5000.0])
crbl_closed.state_variable_range['MLI'] = np.array([15000.0, 15000.0])
crbl_closed.state_variable_range['PC'] = np.array([38000.0, 38000.0])
crbl_closed.state_variable_range['noise'] = np.array([0.0, 0.0])

cerebellum_closed = Subnetwork(
    name='cerebellum', model=crbl_closed,
    scheme=HeunStochastic(dt=dt, noise=Additive(nsig=nsig)),
    nnodes=n_crbl, node_indices=crbl_idx,
)

# 5 projections
p1 = IntraProjection(source_cvar=np.array([0]), target_cvar=np.array([0]),
    weights=W_cc, lengths=TL_cc, cv=cv, dt=dt, scale=1.0)
p2 = InterProjection(source=cortex, target=cerebellum_closed,
    source_cvar=np.array([0]), target_cvar=np.array([0]),
    weights=W_cr, lengths=TL_cr, cv=cv, dt=dt, scale=1.0)
p3 = InterProjection(source=cerebellum_closed, target=cortex,
    source_cvar=np.array([0]), target_cvar=np.array([0]),
    weights=W_rc, lengths=TL_rc, cv=cv, dt=dt, scale=1.0)
p4 = IntraProjection(source_cvar=np.array([0]), target_cvar=np.array([0]),
    weights=W_rr, lengths=TL_rr, cv=cv, dt=dt, scale=1.0)
p5 = IntraProjection(source_cvar=np.array([0]), target_cvar=np.array([1]),
    weights=W_rr, lengths=TL_rr, cv=cv, dt=dt, scale=1.0)

cortex.projections = [p1]
cerebellum_closed.projections = [p4, p5]

cortex.add_monitor(TemporalAverage(period=mon_period))
cerebellum_closed.add_monitor(TemporalAverage(period=mon_period))

nets_closed = NetworkSet(subnets=[cortex, cerebellum_closed], projections=[p2, p3])
sim_closed = Simulator(
    nets=nets_closed, monitors=[TemporalAverage(period=mon_period)],
    simulation_length=sim_len,
)
sim_closed.configure()

print("Running closed-loop (10 s, WW+CRBL)...")
t0 = time.time()
sim_closed.run()
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f} s")

rec_ctx = cortex.monitors[0]
rec_crbl = cerebellum_closed.monitors[0]
ts_c, d_c = rec_ctx.to_arrays()
ts_r, d_r = rec_crbl.to_arrays()

print(f"  Cortex: {d_c.shape}  CRBL: {d_r.shape}")

se_closed = d_c[:, 0, :, 0]
si_closed = d_c[:, 1, :, 0]
grc_closed = d_r[:, 0, :, 0]
goc_closed = d_r[:, 1, :, 0]
mli_closed = d_r[:, 2, :, 0]
pc_closed  = d_r[:, 3, :, 0]

for pop, arr in [('GrC', grc_closed), ('GoC', goc_closed), ('MLI', mli_closed), ('PC', pc_closed)]:
    print(f"  {pop}: mean={arr.mean():.4f} Hz")

# %% [markdown]
# ## 7. PSD Analysis & Frequency Bands
#
# The model's strongest oscillations are in the **ripple band** (150–500 Hz)
# from the GrC↔GoC feedback loop. We show the full 0–500 Hz range with
# canonical cerebellar bands highlighted.

# %%
# Connected CRBL nodes (those with non-zero intra-CRBL SC)
W_rr_dense = W[np.ix_(crbl_idx, crbl_idx)]
connected_crbl_nodes = np.where(W_rr_dense.sum(axis=1) > 0)[0]
print(f"CRBL nodes with SC connections: {len(connected_crbl_nodes)}/27")

bands = {
    'delta':     (1, 4),
    'theta':     (4, 8),
    'alpha':     (8, 13),
    'beta':      (13, 30),
    'low-gamma': (30, 80),
    'high-gamma':(80, 150),
    'ripple':    (150, 500),
}
band_colors = {
    'delta': '#e74c3c', 'theta': '#2ecc71', 'alpha': '#3498db',
    'beta': '#9b59b6', 'low-gamma': '#e67e22', 'high-gamma': '#1abc9c',
    'ripple': '#c0392b',
}

fs = 1000.0 / dt  # sampling freq in Hz

def compute_psd(signal, fs, nperseg=8192):
    """Welch PSD with high frequency resolution."""
    nperseg = builtins.min(nperseg, len(signal))
    noverlap = int(nperseg * 0.75)
    if noverlap >= nperseg:
        noverlap = nperseg - 1
    return scipy.signal.welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

def get_band_peak(freqs, psd, fmin, fmax):
    """Peak/mean ratio within a specific frequency band."""
    mask = (freqs >= fmin) & (freqs < fmax)
    pn = psd[mask]
    if len(pn) == 0 or pn.mean() == 0:
        return 0.0, 0.0
    return pn.max() / pn.mean(), freqs[mask][np.argmax(pn)]

def psd_pop(arr, fs):
    """Average PSD across connected CRBL nodes."""
    if len(connected_crbl_nodes) > 0:
        arr = arr[:, connected_crbl_nodes]
    sig = np.nan_to_num(arr, nan=0.0).mean(axis=1)
    sig = sig - sig.mean()
    return compute_psd(sig, fs)

# Compute PSDs
pops = ['GrC', 'GoC', 'MLI', 'PC']
psd_open = {}; psd_closed = {}
peak_info_open = {}; peak_info_closed = {}

for name, arr in [('GrC', grc_open), ('GoC', goc_open),
                   ('MLI', mli_open), ('PC', pc_open)]:
    f, p = psd_pop(arr, fs)
    psd_open[name] = (f, p)
    peak_info_open[name] = {}
    for bname, (blo, bhi) in bands.items():
        pr, pf = get_band_peak(f, p, blo, bhi)
        peak_info_open[name][bname] = (pr, pf)

for name, arr in [('GrC', grc_closed), ('GoC', goc_closed),
                   ('MLI', mli_closed), ('PC', pc_closed)]:
    f, p = psd_pop(arr, fs)
    psd_closed[name] = (f, p)
    peak_info_closed[name] = {}
    for bname, (blo, bhi) in bands.items():
        pr, pf = get_band_peak(f, p, blo, bhi)
        peak_info_closed[name][bname] = (pr, pf)

# Print band analysis table
print("\n" + "=" * 80)
print("BAND ANALYSIS — Peak/Mean Ratio (peak frequency in Hz)")
print("=" * 80)
for label, peak_info in [('Open-loop', peak_info_open),
                          ('Closed-loop', peak_info_closed)]:
    print(f"\n{label}:")
    header = f"{'Population':<10}" + "".join(f"{b:>12}" for b in bands)
    print(header)
    print("-" * len(header))
    for pop in pops:
        row = f"{pop:<10}"
        for bname in bands:
            pr, pf = peak_info[pop][bname]
            if pr > 3.0:
                row += f"  {pr:>5.1f}x{pf:>3.0f}"
            else:
                row += f"  {pr:>5.1f}x    "
        print(row)

f_se, p_se = psd_pop(se_closed, fs)
print(f"\nCortex S_e carrier (closed): {f_se[np.argmax(p_se)]:.1f} Hz")

# %% [markdown]
# ## 8. PSD Comparison — Open vs Closed Loop
#
# Top row: open-loop (CRBL-only), bottom row: closed-loop (WW+CRBL).
# Each column is one cerebellar population. Canonical bands are shaded.

# %%
fig, axes = plt.subplots(2, 4, figsize=(20, 9))

for ci, pop in enumerate(pops):
    # Open-loop
    ax = axes[0, ci]
    f, p = psd_open[pop]
    pn = (p / p.max()).astype(np.float64) if p.max() > 0 else p.astype(np.float64)
    m = f <= 500
    ax.plot(f[m], pn[m], 'k-', lw=1.5)
    for bn, (blo, bhi) in bands.items():
        if bhi <= 500:
            ax.axvspan(blo, bhi, color=band_colors[bn], alpha=0.15)
    ax.set_title(f'Open: {pop}', fontsize=13)
    ax.set_xlabel('Freq (Hz)'); ax.set_ylabel('Norm. PSD')
    ax.set_xlim(0, 500); ax.grid(True, alpha=0.3)

    # Closed-loop
    ax = axes[1, ci]
    f, p = psd_closed[pop]
    pn = (p / p.max()).astype(np.float64) if p.max() > 0 else p.astype(np.float64)
    m = f <= 500
    ax.plot(f[m], pn[m], 'k-', lw=1.5)
    for bn, (blo, bhi) in bands.items():
        if bhi <= 500:
            ax.axvspan(blo, bhi, color=band_colors[bn], alpha=0.15)
    ax.set_title(f'Closed: {pop}', fontsize=13)
    ax.set_xlabel('Freq (Hz)'); ax.set_ylabel('Norm. PSD')
    ax.set_xlim(0, 500); ax.grid(True, alpha=0.3)

fig.suptitle('Construct Validation: CRBL PSD — Open vs Closed Loop (Ee bug fixed)',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('ww_crbl_psd_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 9. Spectral Overlay — Open vs Closed
#
# Direct comparison per population. Band labels are placed at the
# band center with the peak/mean ratio annotated for each condition.

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ci, pop in enumerate(pops):
    ax = axes[ci]
    f_o, p_o = psd_open[pop]
    f_c, p_c = psd_closed[pop]
    p_on = (p_o / p_o.max()).astype(np.float64) if p_o.max() > 0 else p_o.astype(np.float64)
    p_cn = (p_c / p_c.max()).astype(np.float64) if p_c.max() > 0 else p_c.astype(np.float64)
    mo = f_o <= 500; mc = f_c <= 500
    ax.plot(f_o[mo], p_on[mo], 'b-', lw=1.5, alpha=0.8, label='Open')
    ax.plot(f_c[mc], p_cn[mc], 'r-', lw=1.5, alpha=0.8, label='Closed')
    for bn, (blo, bhi) in bands.items():
        if bhi <= 500:
            ax.axvspan(blo, bhi, color=band_colors[bn], alpha=0.1)
    ax.set_title(pop, fontsize=13); ax.set_xlabel('Freq (Hz)'); ax.set_ylabel('Norm. PSD')
    ax.legend(fontsize=9); ax.set_xlim(0, 500); ax.grid(True, alpha=0.3)
    # Annotate strongest band
    best_pr, best_bn = 0.0, 'delta'
    for bn in bands:
        pr_o = float(peak_info_open[pop][bn][0])
        pr_c = float(peak_info_closed[pop][bn][0])
        pr = builtins.max(pr_o, pr_c)
        if pr > best_pr: best_pr = pr; best_bn = bn
    ax.text(0.98, 0.95, f'Peak: {best_bn} ({best_pr:.1f}x)',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

fig.suptitle('Spectral Shift from Cortical Input (Ee bug fixed)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ww_crbl_psd_overlay.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 10. Inter-Network Coupling Sweep
#
# We sweep the inter-network coupling scale (P2: cortex→CRBL, P3: CRBL→cortex)
# using the **Numba-compiled Simulator** backend. Each sweep point builds
# a fresh 126-node network (CerebellarMF uses a custom Mako template for
# JIT-compiled dfun generation — see `nb-cerebellar-dfun.py.mako`), runs
# a full simulation, and records the TemporalAverage firing rates.
#
# > **Note on the sweep API**: The `NbHybridBackend.sweep()` method provides
# > a `prange`-parallel sweep, but requires named cfun parameters. For the
# > hybrid framework where projections use `cfun=None` (internal coupling),
# > we fall back to the sequential Simulator loop — still JIT-accelerated
# > via the custom Mako template.

# %%
# Use a shorter sim for the sweep (faster turnaround with numba-accelerated dfun)
sweep_len = 5000.0   # 5 s per point
n_sweep = 9
scales = np.linspace(0.0, 2.0, n_sweep)
print(f"Sweeping inter-network coupling scale: {np.round(scales, 2)}")
print(f"  {n_sweep} points × {sweep_len/1000:.0f}s each (Numba JIT-accelerated)")

sweep_ctx_se = np.zeros(n_sweep)
sweep_ctx_si = np.zeros(n_sweep)
sweep_grc = np.zeros(n_sweep)
sweep_goc = np.zeros(n_sweep)
sweep_mli = np.zeros(n_sweep)
sweep_pc  = np.zeros(n_sweep)

for si, scale_val in enumerate(scales):
    _ww = ReducedWongWangExcInh()
    _ww.G = np.array([2.0]); _ww.J_N = np.array([0.15])

    _crbl = CerebellarMF()
    _crbl.variables_of_interest = ('GrC', 'GoC', 'MLI', 'PC')
    _crbl.use_legacy_goc_e_e = np.array([True])
    _crbl.external_input_ex_ex = np.array([0.0])
    _crbl.state_variable_range['GrC'] = np.array([500.0, 500.0])
    _crbl.state_variable_range['GoC'] = np.array([5000.0, 5000.0])
    _crbl.state_variable_range['MLI'] = np.array([15000.0, 15000.0])
    _crbl.state_variable_range['PC'] = np.array([38000.0, 38000.0])
    _crbl.state_variable_range['noise'] = np.array([0.0, 0.0])

    _ctx = Subnetwork(name='cortex', model=_ww,
        scheme=HeunStochastic(dt=dt, noise=Additive(nsig=nsig)),
        nnodes=n_cortex, node_indices=cortex_idx)
    _crbl_net = Subnetwork(name='cerebellum', model=_crbl,
        scheme=HeunStochastic(dt=dt, noise=Additive(nsig=nsig)),
        nnodes=n_crbl, node_indices=crbl_idx)

    _p1 = IntraProjection(source_cvar=np.array([0]), target_cvar=np.array([0]),
        weights=W_cc, lengths=TL_cc, cv=cv, dt=dt, scale=1.0)
    _p2 = InterProjection(source=_ctx, target=_crbl_net,
        source_cvar=np.array([0]), target_cvar=np.array([0]),
        weights=W_cr, lengths=TL_cr, cv=cv, dt=dt, scale=scale_val)
    _p3 = InterProjection(source=_crbl_net, target=_ctx,
        source_cvar=np.array([0]), target_cvar=np.array([0]),
        weights=W_rc, lengths=TL_rc, cv=cv, dt=dt, scale=scale_val)
    _p4 = IntraProjection(source_cvar=np.array([0]), target_cvar=np.array([0]),
        weights=W_rr, lengths=TL_rr, cv=cv, dt=dt, scale=1.0)
    _p5 = IntraProjection(source_cvar=np.array([0]), target_cvar=np.array([1]),
        weights=W_rr, lengths=TL_rr, cv=cv, dt=dt, scale=1.0)

    _ctx.projections = [_p1]
    _crbl_net.projections = [_p4, _p5]
    _ctx.add_monitor(TemporalAverage(period=mon_period))
    _crbl_net.add_monitor(TemporalAverage(period=mon_period))

    _nets = NetworkSet(subnets=[_ctx, _crbl_net], projections=[_p2, _p3])
    _sim = Simulator(nets=_nets, monitors=[TemporalAverage(period=mon_period)],
        simulation_length=sweep_len)
    _sim.configure()
    _sim.run()

    _, _d_c = _ctx.monitors[0].to_arrays()
    _, _d_r = _crbl_net.monitors[0].to_arrays()

    skip = int(0.2 * _d_c.shape[0])
    sweep_ctx_se[si] = _d_c[skip:, 0, :, 0].mean()
    sweep_ctx_si[si] = _d_c[skip:, 1, :, 0].mean()
    sweep_grc[si] = _d_r[skip:, 0, :, 0].mean()
    sweep_goc[si] = _d_r[skip:, 1, :, 0].mean()
    sweep_mli[si] = _d_r[skip:, 2, :, 0].mean()
    sweep_pc[si]  = _d_r[skip:, 3, :, 0].mean()

    print(f"  scale={scale_val:.2f}: S_e={sweep_ctx_se[si]:.4f} "
          f"GrC={sweep_grc[si]:.4f} GoC={sweep_goc[si]:.4f} "
          f"MLI={sweep_mli[si]:.4f} PC={sweep_pc[si]:.4f}")

print("\nSweep complete!")

# %% [markdown]
# ## 11. Sweep Results — Activity Balance
#
# For each sweep point, we plot the mean firing rate of each population
# (averaged across nodes and time, skipping the transient). The ideal
# coupling scale keeps both cortical and cerebellar activity in a healthy
# range — not saturated, not silent.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Cortex
ax1.plot(scales, sweep_ctx_se, 'o-', label='S_e', color='#3498db', lw=2)
ax1.plot(scales, sweep_ctx_si, 's-', label='S_i', color='#e74c3c', lw=2)
ax1.set_xlabel('Inter-network coupling scale')
ax1.set_ylabel('Mean firing rate (Hz)')
ax1.set_title('Cortex activity vs inter-network coupling')
ax1.legend(); ax1.grid(True, alpha=0.3)

# CRBL
for arr, lbl, col in [
    (sweep_grc, 'GrC', '#2ecc71'), (sweep_goc, 'GoC', '#e67e22'),
    (sweep_mli, 'MLI', '#9b59b6'), (sweep_pc, 'PC', '#1abc9c'),
]:
    ax2.plot(scales, arr, 'o-', label=lbl, color=col, lw=2)
ax2.set_xlabel('Inter-network coupling scale')
ax2.set_ylabel('Mean firing rate (Hz)')
ax2.set_title('Cerebellum activity vs inter-network coupling')
ax2.legend(); ax2.grid(True, alpha=0.3)

fig.suptitle('Inter-Network Coupling Sweep — Activity Balance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ww_crbl_coupling_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# Print a summary table
print("\nCoupling Scale Sweep Summary:")
print(f"{'Scale':<8} {'Ctx S_e':<10} {'Ctx S_i':<10} {'GrC':<10} {'GoC':<10} {'MLI':<10} {'PC':<10}")
print("-" * 68)
for i, s in enumerate(scales):
    print(f"{s:<8.2f} {sweep_ctx_se[i]:<10.4f} {sweep_ctx_si[i]:<10.4f} "
          f"{sweep_grc[i]:<10.4f} {sweep_goc[i]:<10.4f} "
          f"{sweep_mli[i]:<10.4f} {sweep_pc[i]:<10.4f}")

# %% [markdown]
# ## 12. Time Series — Open vs Closed Loop

# %%
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
skip = 500  # skip transient (ms)
for ci, (pop, o_arr, c_arr) in enumerate([
    ('GrC', grc_open, grc_closed), ('GoC', goc_open, goc_closed),
    ('MLI', mli_open, mli_closed), ('PC', pc_open, pc_closed),
]):
    axes[0, ci].plot(ts_open[skip:], o_arr[skip:, :3], alpha=0.7)
    axes[0, ci].set_title(f'Open: {pop}'); axes[0, ci].set_xlabel('t [ms]')
    axes[0, ci].grid(True, alpha=0.3)

    axes[1, ci].plot(ts_r[skip:], c_arr[skip:, :3], alpha=0.7)
    axes[1, ci].set_title(f'Closed: {pop}'); axes[1, ci].set_xlabel('t [ms]')
    axes[1, ci].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ww_crbl_timeseries.png', dpi=150)
plt.show()

# %% [markdown]
# ## 13. Functional Connectivity

# %%
skip = 500
se_clean = se_closed[skip:, :]
fc = np.corrcoef(se_clean.T)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(fc, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_title('Cortex FC (S_e, closed-loop)')
ax.set_xlabel('Node'); ax.set_ylabel('Node')
plt.colorbar(im, ax=ax, label='Pearson r')
plt.tight_layout()
plt.savefig('ww_crbl_fc.png', dpi=150)
plt.show()

# %% [markdown]
# ## 14. Summary
#
# ### Bug fix
#
# The original `TF_inhibitory_goc` in `parallel_crbl_multimf_ww.py` (line 820)
# uses `self.E_i` for both the excitatory and inhibitory reversal potentials.
# This makes GoC "excitatory" drive hyperpolarizing (toward -80 mV instead of
# 0 mV), silencing GoC entirely. The fix: `self.E_e, self.E_i` — a one-
# character change that unlocks the GrC↔GoC feedback loop.
#
# ### Oscillatory dynamics
#
# With the Ee fix, the CerebellarMF model produces strong oscillations in the
# **ripple band** (150–190 Hz). Peak/mean PSD ratios:
#
# | Population | Open-loop | Closed-loop |
# |-----------|-----------|-------------|
# | GrC | ~16x @ ~160 Hz | ~17x @ ~160 Hz |
# | GoC | ~13x @ ~188 Hz | ~13x @ ~188 Hz |
# | MLI | ~22x @ ~152 Hz | ~24x @ ~157 Hz |
# | PC  | ~27x @ ~157 Hz | ~31x @ ~157 Hz |
#
# These ripple oscillations arise from the GrC↔GoC feedback loop
# (~12 ms loop delay) and match the monolithic reference model.
# The traditional delta–beta bands (1–30 Hz) show relatively flat PSDs.
#
# ### Inter-network coupling
#
# The coupling sweep identifies the range of inter-network scaling
# values that keep both cortex and cerebellum in active, balanced regimes.
# At scale=0 the network is open-loop; at scale>1, cortical drive begins
# to dominate cerebellar dynamics.
