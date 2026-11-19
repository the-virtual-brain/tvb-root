# cpp_hybrid Feature-Parity Audit vs nb_hybrid

Date: 2026-11-19 (updated from 2026-09-29). Reference surface:
`tvb/simulator/backend/nb_hybrid.py` (`_get_supported_models_classes`,
`_cfun_type`, monitor validation, stimulus plumbing). Target:
`tvb/simulator/backend/cpp_hybrid/` (`_core.cpp`, `cph_*.hpp`, `backend.py`,
`dfungen.py`, `features.py`, `notebook.py`).

## 1. Models — 29+1 = 30/30 `ModelsEnum` supported (CLOSED)

cpp_hybrid has 13 hand-written dfuns (ids 0–12, incl. `MOD_CRBL = 12` for
`CerebellarMF`) **plus** the built-in expression-driven generic table
(`dfungen.py` → `enable_generic_dfuns`, ids 100–126) and runtime g++
fallback (`ids >= 127`) for any custom `Model` subclass declaring
`state_variable_dfuns`.

With `DecoBalancedExcInh` added to both `nb_hybrid._get_supported_models_classes()`
and `dfungen._SUPPORTED_MODEL_CLASSES`, the build emits **27** built-in
expression-driven kernels (`dfungen --only-supported`, ids `100–126`), covering
all **29** classes in `_get_supported_models_classes()` (plus
`ZerlautAdaptationSecondOrder`, giving **30/30** `ModelsEnum` coverage in
native and Emscripten WASM builds).

Enumerated live (`collect_models()` vs `_get_supported_models_classes()`):

| Model | generic? | hardcoded? | cpp_hybrid status |
|---|---|---|---|
| MontbrioPazoRoxin | YES | YES | implemented |
| Generic2dOscillator | YES | YES | implemented |
| Kuramoto | YES | YES | implemented |
| SupHopf | YES | YES | implemented |
| Linear | YES | YES | implemented |
| ReducedWongWang | YES | YES | implemented |
| WilsonCowan | YES | YES | implemented |
| JansenRit | YES | YES | implemented |
| Epileptor | YES | YES | implemented |
| Epileptor2D | YES | YES | implemented |
| ZerlautAdaptationFirstOrder | no | YES | implemented |
| ZerlautAdaptationSecondOrder | no | YES | implemented |
| ZetterbergJansen | YES | YES | implemented |
| ReducedWongWangExcInh | YES | – | built-in generic (parity tested) |
| DecoBalancedExcInh | YES | – | built-in generic (30/30 coverage) |
| KIonEx | YES | – | built-in generic (parity tested) |
| EpileptorCodim3 | YES | – | built-in generic (parity tested) |
| EpileptorCodim3SlowMod | YES | – | built-in generic (parity tested) |
| EpileptorRestingState | YES | – | built-in generic (parity tested) |
| Hopfield | YES | – | built-in generic (parity tested) |
| LarterBreakspear | YES | – | built-in generic (parity tested) |
| CoombesByrne2D | YES | – | built-in generic (parity tested) |
| CoombesByrne | YES | – | built-in generic (parity tested) |
| GastSchmidtKnosche_SD | YES | – | built-in generic (parity tested) |
| GastSchmidtKnosche_SF | YES | – | built-in generic (parity tested) |
| DumontGutkin | YES | – | built-in generic (parity tested) |
| ReducedSetFitzHughNagumo | YES | – | built-in generic (parity tested) |
| ReducedSetHindmarshRose | YES | – | built-in generic (parity tested) |
| CerebellarMF | no | YES (id 12) | implemented (parity tested) |

## 2. Coupling functions — all 8 classes + variants (CLOSED)

`_cfun_type()` (nb_hybrid) distinguishes: `none`, `linear`, `scaling`,
`sigmoidal`, `sigmoidal_jr`, `sigmoidal_jr_legacy` (use_classic=0 or
single source cvar), `kuramoto`, `difference`, `tanh`, `pre_sigmoidal`,
`pre_sigmoidal_dynamic` (dynamic=1 with 2 source cvars).

| Coupling | nb_hybrid | cpp_hybrid status |
|---|---|---|
| Linear / Scaling / Sigmoidal / Difference / Kuramoto / HyperbolicTangent | yes | implemented (parity tested) |
| SigmoidalJansenRit (classic, 2 src cvars, id 8) | yes | implemented (incl. `a*a` sweep) |
| SigmoidalJansenRit legacy (use_classic=0, id 8) | yes | implemented (CLOSED) |
| PreSigmoidal dynamic (dynamic=1, id 9 incl. globalT) | yes | implemented (CLOSED) |
| PreSigmoidal static | yes | implemented |

## 3. Monitors (Batch B — CLOSED)

All supported TVB monitors now execute natively inside C++ (`cph_monitors.hpp`,
`kmon<W>` across `W in {1, 4, 8, 16}`), reading directly from reusable
per-step `observe_voi_row<W>` / `gather_global_voi_row<W>` buffers so
`chunk_size > 1` no longer forces `execution_chunk_size = 1` or raw
`tavg`/`ctavg` copies (`need_tavg=False, need_ctavg=False`):

- **Per-subnet & global monitors:** `Raw`, `RawVoi`, `SubSample`,
  `TemporalAverage` (`KMON_TAVG`), `AfferentCoupling` /
  `AfferentCouplingTemporalAverage` (`KMON_CTAVG`, reading `sub.c` directly
  with `n_modes * W`), `Bold` (`KMON_BOLD`, HRF convolution + optional
  `KMON_BOLD_BW` SIMD Balloon-Windkessel), `GlobalAverage`
  (`KMON_GLOBAL_AVG`), `SpatialAverage` (`KMON_SPATIAL_AVG`, dense & CSR),
  and `Projection` (`EEG` with `_ref_vec` subtraction, `MEG`, `iEEG` →
  `KMON_PROJECTION`). Only `BoldRegionROI` falls back to Python.
- **Per-monitor `voi` selection & multi-dt `cadence_K`:** Each `kmon<W>`
  accepts explicit `voi` indices and `cadence_K`, enabling single-pass
  per-subnet `Recorder` execution (`subnet_monitors`) in `hybrid/simulator.py`
  without a second simulation run.
- **Full `KMonState` snapshot/resume:** `get_monitor_states()` /
  `set_monitor_states()` and `get_global_monitor_states()` /
  `set_global_monitor_states()` persist `Bold`, `TemporalAverage`,
  `SpatialAverage`, `Projection`, etc. across `snapshot()` / `resume()`
  in C++ without falling back to Python.

## 4. Stimuli, Philox PRNG, Features, & SIMD Widths (Batch C — CLOSED)

- **Separable C++ stimuli (`cph_stim.hpp`):** `Stim` (`SpatioTemporalPattern`)
  with 9 analytic temporal equations (`Linear`, `Sinusoid`, `Cosine`,
  `PulseTrain`, `Alpha`, `GeneralizedSigmoid`, `Gaussian`, `DoubleGaussian`,
  `Sigmoid`) or 1D `STIM_DISCRETE_1D` fallback is lowered via
  `sim.set_stim_specs()` for both single runs (incl. multi-dt master `dt0`
  grid) and `sweep()`, retaining 5D dense `stim` fallback only for
  non-separable stimuli.
- **Philox4x32-10 PRNG & stochastic sweeps (`cph_rng.hpp`):** Preserves
  bit-identical NumPy MT19937 parity under `rng_mode="numpy"` (default for
  single runs), adds `rng_mode="philox"` (with colored OU noise `ntau >= 0`
  and full `rng_state<W>` snapshot/resume), and enables stochastic `sweep()` /
  `run_sweep()` across `W in {1, 4, 8, 16}` (`shared_noise`, `noise_seeds`,
  `ntau >= 0`).
- **SIMD feature extraction (`cph_features.hpp` + `features.py`):** Tier-1
  streaming `OnlineFeatureAccum<W>` (`O(n_voi * n_node * W)` memory,
  zero time-series storage) + Tier-2 C++ order-stats/entropy/radix-2+Bluestein
  Welch PSD exposed via `_cpp_hybrid.compute_features` and `sweep(features=...)`.
- **Portable SIMD widths (`W in {1, 4, 8, 16}`):** `notebook.default_width()`
  defaults to `4` on `emscripten` (`-msimd128`) and `arm64`/`aarch64`, and
  `8` on `x86_64`.

## 5. Priority order for implementation (ALL CLOSED)

1. M1 CerebellarMF (CLOSED: hand-written dfun `MOD_CRBL=12`, parity tested).
2. C2 PreSigmoidal dynamic (CLOSED: id 9 incl. globalT); C1 SJR legacy
   (CLOSED: id 8).
3. S1 multi-subnetwork stimuli + multi-stochastic (CLOSED: per-subnet
   noise/stim sequences + separable `cph_stim.hpp` + Philox `cph_rng.hpp`);
   L1 not a gap.
4. Per-model parity tests for all built-in models (CLOSED: 30/30 `ModelsEnum`
   incl. `DecoBalancedExcInh`).
5. Batch B all-in-C++ monitors, `KMonState` snapshot/resume, and single-pass
   `Simulator(backend="cpp")` `subnet_monitors` (CLOSED).
6. Batch C separable stimuli in `sweep()`, stochastic Philox sweeps,
   SIMD features (`cph_features.hpp`), and `W in {1, 4, 8, 16}` (CLOSED).

## 6. Multi-dt support (LOCKED 2026-09-30 — binding spec for cards 9-B..9-J)

Design source: `~/src/tvb-kh-notes/hybrid-multi-dt-concerns.md` (blocker map
A–L, semantic-decision checklist §4, sequencing §5). All subnet dts are
integer multiples of a base `dt0` (subnet j: `dt_j = k_j * dt0`, integer
`k_j >= 1`), validated in both `_check_compatibility` implementations; a
master clock ticks at `dt0` and subnet j integrates on master ticks
`t mod k_j == 0`.

Status (2026-09-30): the 12 decisions below were audited against the
concerns doc and against the committed single-dt kernels on branch `tvbkh`
(`_core.cpp` master loop / push / projection reads / `set_subnet_state` IC
prefill; `nb_hybrid.py` srcbuf init, snapshot fields, SHA-256 source-keyed
cache; `nb-hybrid-sim.py.mako` chunk loop, per-chunk RNG draws, tavg/ctavg
normalization; cpp `backend.py` projection build order, `_topology_key`,
`_build_stim_arrays`, `_stim_estimate_mb`). The earlier draft of this
section (07:21, marked CLOSED without review or a green run) was untrusted:
decisions 1, 2, 3, 6, 11, 12 are corrected here, the rest confirmed and
tightened. **This section is THE multi-dt spec: implementation cards 9-B..9-J
reference it and must not diverge.** The degenerate gate (all `k_j = 1`) must
reproduce the pre-multi-dt single-dt trajectories bit-for-bit in both
backends. Implementation: `nb_hybrid.py` (oracle) + `nb-hybrid-sim.py.mako`
(codegen), `cpp_hybrid/_core.cpp` + `backend.py` (port),
`test_cpp_hybrid_multi_dt.py` (parity + naive oracle).

### Pinned semantics (decisions 1–12)

1. **Interpolation formula (float32, identical in both implementations)**
   [concerns §4-1, blocker B]: only inter-subnet reads interpolate. Read
   position per edge, in source-step units: `tau = (t - 1) / k_src - idelay`
   at master tick `t` (1-based). Sampled steps: `i0 = floor(tau)`,
   `i1 = i0 + 1`. Blend weight `alpha = ((t - 1) mod k_src) / k_src`,
   identical to `tau - i0` because delays are integers — one alpha per
   source subnet per master tick, shared by every edge reading that source.
   Pinned value expression, evaluated left-to-right in float32 with no FMA
   contraction and no fast-math re-association in either backend:
   `tmp = x1 - x0; val = x0 + alpha * tmp`, with
   `alpha = float32((t - 1) mod k_src) / float32(k_src)`. Never
   extrapolates: `x1` is clamped to `x0` when the upper sample has not
   been pushed (decision 3), and pre-history samples come from the
   IC-prefilled ring (decision 4). Intra-subnet projections never
   interpolate (exact integer reads, decision 3; alpha not applied). If
   `k_src == 1`, `alpha == 0` at every tick and the read is the legacy
   single-slot read exactly (degenerate gate).
2. **Master-tick ordering (push / apply / integrate)** [concerns §4-2,
   blocker C]: per master tick `t` (1-based absolute step): (1) zero the
   coupling scratch `c` of every subnet; (2) apply all projections in the
   backend build order — all intra-subnet projections first (per-subnet
   order), then all inter-subnet projections (graph order) — reads per
   decision 3; (3) for each subnet in index order: add the current
   master-grid stimulus row into its `c` (decision 7); accumulate ctavg
   (decision 6); if `t mod k_j == 0`, integrate its own step with its own
   `dt_j` and its own noise slice (decision 8), then push the completed
   step into its own history ring using the per-subnet completed-step count
   `j = t // k_j` — per-subnet step counters advance by exactly one per
   push (asserted, blocker A), and each backend keeps its legacy slot map
   so ring *content* is the cross-backend invariant (C++ per-subnet slot
   `(j - 1) & (H - 1)`, oracle slot `j % H`); accumulate per-subnet tavg
   (and the oracle's spatial/proj monitor accumulators) on the master
   grid; (4) Bold and TemporalAverage sampling happen in the monitor layer
   on the master grid (a master tick is an observation point; mechanics
   unchanged). With all `k_j = 1` this reduces bit-for-bit to the
   pre-multi-dt loop order (projection apply order included).

   **Sample-time quantization (pinned 2026-10-01, review finding):** the
   per-chunk sample times are the mean of the first and last master tick
   of the chunk, scaled by `float64(float32(dt0))` — i.e. the `dt0`
   quantized to float32 once, exactly as the nb template has always done
   (`time_step = np.float32(dt0)`; per-chunk `mid_t = (t_global +
   t_global + this_chunk - 1) * 0.5 * float(time_step)`).  A partial final
   chunk uses its own bounds (nb and the naive oracle formula).  The cpp
   backend matches this bit-for-bit (`_run_compiled`'s
   `(steps_lo + steps_hi) * 0.5 * float(np.float32(dt0))`), and the C++
   monitor engines stamp Raw/SubSample rows the same way
   (`within * (double)(float)dt`); Bold engine times stay in double (their
   reference is the TVB monitor's own `_config_dt` arithmetic).  Without
   the pin the two backends' time axes differ by ~2e-8 (float32 vs float64
   `dt0`), which exact-time parity tests surface.
3. **Read semantics (incl. fast target)** [concerns §4-3, blocker C]: at
   the apply phase of master tick `t`, the newest pushed source-step index
   is `m = floor((t - 1) / k_src)` (steps are pushed at the end of ticks
   `j * k_src`; the IC is step index 0, always readable). Inter edges:
   `tau = (t - 1) / k_src - idelay`; `i0 = floor(tau)` is always readable
   (negative `i0` wraps to IC-prefilled ring slots, decision 4);
   `i1 = i0 + 1` is readable iff `i1 <= m`, which holds exactly when
   `idelay >= 1`; otherwise `x1 := x0` (clamp), which makes the
   interpolant identically `x0` — a zero-order hold of the newest pushed
   state. A fast target (small `k_tgt`) reading a slow source therefore
   sees the lagged segment between the two most recent pushed source
   states, ZOH between source pushes, never a not-yet-pushed state and
   never extrapolated; a slow target reading any source obtains exact
   bracketing reads. Intra edges (source == target subnet, `k_src = k_tgt
   = k_j`): exact integer read position `(t // k_j) - 1 - idelay` (the
   state at the start of the own step being integrated), single slot, no
   interpolation, identical to the legacy read when `k_j = 1`; on the
   target's off-ticks this value is a monitor artifact only (ctavg,
   decision 6) and never feeds integration. Integer `tau` reduces to the
   old single-slot read exactly.
4. **Startup / ring wrap** [concerns §4-4, blocker D]: ring slots wrap as
   in the legacy kernels (C++ `& (H - 1)` with power-of-two `H`; oracle
   `% H` with the per-source horizon) and every slot of every history ring
   is pre-filled with the subnetwork's initial state in both backends (C++
   `set_subnet_state` prefill; oracle IC broadcast across all horizon
   slots — both already implemented pre-multi-dt). Early reads therefore
   yield the IC exactly as in the pre-multi-dt code; no zero-padding, wrap
   semantics unchanged; an interpolating pair can only blend two pushed
   samples, or a pushed sample with the IC — never with a pre-run zero.
5. **Raw / SubSample sampling grid** [concerns §4-5, blocker E]: monitors
   operate on the master `dt0` grid: per monitor, `istep =
   max(1, round(period / dt0))` master ticks and `chunk_size = GCD(monitor
   isteps)` computed on the master grid (Raw/SubSample keep
   `chunk_size = 1`); Raw emits one sample per master tick, SubSample
   selects master ticks (`step % istep == 0`). A slow subnet's output
   between its own steps repeats its last integrated state (ZOH of the
   state trajectory) and stays meaningful as a master-grid sample. All
   monitor time axes — including chunk midpoints (legacy midpoint formula)
   — are master-tick times.
6. **ctavg on a slow subnet's off-ticks** [concerns §4-6, blocker E]: the
   coupling scratch `c` is recomputed for every subnet on every master tick
   (projections run every tick, decision 2), and ctavg accumulates that
   per-master-tick `c` on every tick — no hold, no zero, no skip. Chunk
   normalization is pinned: ctavg per chunk = (sum of per-master-tick `c`
   over the chunk) / (number of master ticks accumulated in that chunk) —
   the oracle's long-standing `tavg_count` normalization; the C++ kernel's
   fixed chunk-size denominator is replaced by the accumulated step count
   so both backends agree on every chunk (identical to legacy on full
   chunks, preserving the degenerate gate).
7. **Stimulus grid + memory policy** [concerns §4-7, blocker F]: stimuli
   are evaluated on the master `dt0` grid and indexed by the global master
   step: function-valued stimuli are called once per master tick,
   array-valued stimuli are precomputed at master resolution, and the
   layout/indexing mechanics are unchanged (per-subnet
   `(nstep_master, n_cvar, n_nodes, nm, W)`; `_build_stim_arrays` step
   indexing). Memory policy: full master-grid arrays are allocated for
   every stimulated subnet — a slow subnet's off-tick rows are unused but
   allocated; memory scales with the total master-step count, and
   `_stim_estimate_mb` plus the stimulus memory-estimate test use the
   master-grid step count so the `1/dt0` growth stays covered.
8. **RNG draw order** [concerns §4-8, blocker G]: one full-chunk draw per
   stochastic subnet per chunk, at chunk start, in subnet index order, from
   each subnet's own RNG: `randn(this_chunk, n_svar, n_nodes, n_modes)`
   scaled by `sqrt(2 * nsig * dt_j)` (each subnet's own dt) into the
   per-master-tick float32 layout. A slow subnet consumes only its due-tick
   rows; off-tick rows are drawn and unused. Each subnet's per-chunk stream
   is identical to its single-dt stream (same draw source, same order), so
   the all-`k_j = 1` noise is unchanged and both backends consume identical
   arrays bit-for-bit.
9. **Merge eligibility = equal dt/k** [concerns §4-9, blocker H]:
   `_can_merge_subnets` keeps its existing conditions (node_indices
   present; equal voi count) and additionally requires equal dt — equal
   `k_j` — for every constituent; merging across different dts is rejected
   because it would silently erase the multi-dt distinction.
10. **Merged / feature-table time axis** [concerns §4-10, blocker E]: all
    outputs — per-subnet and merged — live on the master tick grid; merged
    `SweepResult` and `features.py` tables align on the shared master-tick
    time axis, with per-subnet rows being their master-grid samples
    (decision 5; ZOH for slow subnets) — no resampling at merge. The
    equal-dt merge guard (decision 9) guarantees merged constituents share
    identical step grids, so one master axis per merged table is exact.
11. **Snapshot / resume** [concerns §4-11, blocker I]: the snapshot stores
    per-subnet state vectors, per-subnet history rings (all slots — needed
    so delayed reads after resume are exact; the oracle already stores
    them, the C++ side must add ring persistence — it currently records
    `buffers: {}`), the master step counter (oracle `_step_offset`; the
    C++ side must restore `t_abs` — resume currently re-inits states and
    resets it), per-subnet RNG states, and monitor states. Per-subnet tick
    phase is derived, not stored: subnet j's next integration tick is the
    first `t > t_abs` with `t mod k_j == 0` (its phase is `t_abs mod k_j`).
    Resume continues at master tick `t_abs + 1` and reproduces the
    continuous run bit-for-bit; multi-dt resume-parity tests round-trip at
    2:1 and 4:1 ratios.
12. **Cache-key dt vector** [concerns §4-12, blocker J]: nb_hybrid's
    in-process and disk caches are keyed by SHA-256 of the rendered source
    (existing mechanism); the codegen must bake each subnet's `dt_j` and
    its `k_j` gating / interpolation literals into the rendered source so
    that any change to the dt vector changes every rendered source and
    hence every key — the dt vector is thereby part of both nb_hybrid
    cache keys. The cpp_hybrid compiled-kernel topology key (per-subnet
    model/nodes/modes/svars/cvars/horizon + projection info; currently no
    dt term) must be extended with the full per-subnet dt vector (each
    `k_j`). A lock test asserts different dt vectors yield different keys
    in both backends.

### Blocker-map resolution (A–L)

| Blocker | Resolution |
|---|---|
| A slot aliasing under per-subnet stepping | push by per-subnet completed-step count `j = t // k_j`, per-subnet counters advance by exactly one per push (asserted); read slots from decision 3 in source-step units; ring content is the cross-backend invariant |
| B interpolation = 2× gathers | inter edges: two-slot reads `i0`/`i1` per edge, one shared per-source alpha (integer delays); the contiguous matvec fast path is skipped when `k_src > 1` (perf note below); intra edges stay single-slot |
| C direction semantics | decisions 2 + 3 (fast target = lagged ZOH segment; slow target = exact bracketing reads; intra = exact integer reads) |
| D ring wrap at startup | decision 4 (wrap + IC-prefilled rings, unchanged) |
| E monitors & chunking | decisions 5, 6, 10; chunk size = GCD of monitor isteps on the master grid |
| F stimulus arrays | decision 7 (master grid; memory estimate on master step count) |
| G RNG draw order | decision 8 (per-chunk draws in subnet index order; off-tick rows drawn and unused) |
| H merged-subnet dt guard | decision 9 (`_can_merge_subnets` requires equal dt `k_j`) |
| I checkpoint/resume | decision 11 (rings + `t_abs` + RNG + monitor states; phase derived as `t_abs mod k_j`) |
| J cache invalidation | decision 12 (dt vector in both cache keys; lock test) |
| K template surgery | Mako template + nb_hybrid plumbing changed per decisions 1–9 (this document + code) |
| L validation strategy gap | `test_cpp_hybrid_multi_dt.py`: degenerate k=1 gate, 2:1/4:1 both directions, stochastic, resume round-trips, monitors, stim, cache-key and merge-guard locks, plus an independent naive pure-Python oracle for the first ratios |

Perf note (blocker B): interpolating projections always take the slow path;
projections whose source has `k_src == 1` retain the contiguous matvec fast
path bit-for-bit.
