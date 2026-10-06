# cpp_hybrid Feature-Parity Audit vs nb_hybrid

Date: 2026-09-29. Reference surface: `tvb/simulator/backend/nb_hybrid.py`
(`_get_supported_models_classes`, `_cfun_type`, monitor validation, stimulus
plumbing). Target: `tvb/simulator/backend/cpp_hybrid/` (`_core.cpp`,
`backend.py`, `dfungen.py`).

## 1. Models — 27 supported by nb_hybrid

cpp_hybrid has 12 hand-written dfuns (ids 0–11) **plus** a generic route
(`dfungen.py` → `enable_generic_dfuns`, ids ≥ 100) that JIT-translates any
model class declaring `state_variable_dfuns` not covered by the built-in
kernels.  Since handoff item 6 (option 2), the **build** emits only the
supported model set (`dfungen --only-supported`, 25 expression-driven
kernels, ids 100–124); every in-tree model outside that set
(`DecoBalancedExcInh`) still runs through the runtime g++ fallback, so the
runtime surface below is unchanged.

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
| ReducedWongWangExcInh | YES | – | generic (needs parity test) |
| KIonEx | YES | – | generic (needs parity test) |
| EpileptorCodim3 | YES | – | generic (needs parity test) |
| EpileptorCodim3SlowMod | YES | – | generic (needs parity test) |
| EpileptorRestingState | YES | – | generic (needs parity test) |
| Hopfield | YES | – | generic (needs parity test) |
| LarterBreakspear | YES | – | generic (needs parity test) |
| CoombesByrne2D | YES | – | generic (needs parity test) |
| CoombesByrne | YES | – | generic (needs parity test) |
| GastSchmidtKnosche_SD | YES | – | generic (needs parity test) |
| GastSchmidtKnosche_SF | YES | – | generic (needs parity test) |
| DumontGutkin | YES | – | generic (needs parity test) |
| ReducedSetFitzHughNagumo | YES | – | generic (needs parity test) |
| ReducedSetHindmarshRose | YES | – | generic (needs parity test) |
| **CerebellarMF** | **no** | **–** | **MISSING** |

Gap M1: `CerebellarMF` has neither `state_variable_dfuns` nor a hand-written
dfun. Route: hand-written dfun in `_core.cpp` (preferred, matches nb_hybrid
gufunc semantics) or add `state_variable_dfuns` to the class.

## 2. Coupling functions — nb_hybrid supports 8 classes + variants

`_cfun_type()` (nb_hybrid) distinguishes: `none`, `linear`, `scaling`,
`sigmoidal`, `sigmoidal_jr`, **`sigmoidal_jr_legacy`** (use_classic=0 or
single source cvar), `kuramoto`, `difference`, `tanh`, `pre_sigmoidal`,
**`pre_sigmoidal_dynamic`** (dynamic=1 with 2 source cvars).

cpp_hybrid `_CFUN_IDS` maps the 8 class names to C dispatch cases 0–7 in
`_core.cpp` (`cfun_pre`/`cfun_post`).

| Coupling | nb_hybrid | cpp_hybrid status |
|---|---|---|
| Linear / Scaling / Sigmoidal / Difference / Kuramoto / HyperbolicTangent | yes | implemented (parity-tested: contract suite) |
| SigmoidalJansenRit (classic, 2 src cvars) | yes | implemented |
| **SigmoidalJansenRit legacy** (use_classic=0) | yes | **MISSING** — cpp_hybrid dispatches by class name only; the legacy form would silently use the classic C formula |
| **PreSigmoidal dynamic** (dynamic=1) | yes | **MISSING** — `backend.py` raises `NotImplementedError` (`dynamic=False only`) |
| PreSigmoidal static | yes | implemented |

Gap C1: SJR legacy pre-transform variant. Route: extend `cfun_pre` case 6
(+ an extra dispatch id or a parm flag) and map it in `backend.py`.
Gap C2: PreSigmoidal dynamic. Route: new cfun case (H*(Q+tanh(G*(P*(v0−w·v1)−θ))))
following the nb_hybrid formula; remove the backend rejection.

## 3. Monitors

cpp_hybrid reuses nb_hybrid's monitor plumbing directly
(`_apply_monitors`, `_validate_monitors`, `_compute_chunk_size`,
`_copy_monitor_states` imported from `backend/nb_hybrid.py`), so supported
sets match **by construction**:

TemporalAverage, Raw, SubSample, GlobalAverage, AfferentCoupling (+Temporal
Average subclass), SpatialAverage, Projection (EEG/MEG/iEEG), Bold.

### Kernel monitor engines (CLOSED 2026-09-30, handoff item 8)

Step-level monitor post-processing used to run as NumPy on the kernel's raw
output (most visibly the Bold HRF convolution: per-step Python `sample()`
calls plus a `numpy.dot` over the 5000-row HRF stock at every TR).  That is a
performance gap, **not** a parity gap — both backends shared the same
Python path, so parity held by construction.

Now the C++ step loop runs per-subnet **monitor engines** (`kmon` in
`_core.cpp`, armed via `Sim.set_monitors`):

- **Bold** — interim-stock averaging (4 ms resolution), circular HRF stock
  and the rolled `hrf · stock` convolution run inside the GIL-released
  loop, in double-precision accumulation over the same rows the kernel
  already computes for `tavg`, so the engine's inputs are bit-identical to
  what the Python reference consumes.  State (stocks + step counter)
  persists across `run()` calls like the Python monitor runtimes.
- **Raw / RawVoi** — per-step observed-row collection.
- **SubSample** — per-step row capture with the period mask applied in the
  loop.

TemporalAverage stays on the shared Python path (the kernel already
pre-accumulates `tavg`/`ctavg`, so it is cheap); `features.py` post-hoc
feature computation was left as-is (not a bottleneck).  AfferentCoupling
(its stream is the coupling, not the observed state) and BoldRegionROI
(extra Python-side region-ROI reduction) also stay on the Python path.

**Parity is no longer by construction for moved monitors**: the shared
Python `_apply_monitors` path remains the reference (nb_hybrid still drives
it; cpp falls back to it when `CppHybridBackend.enable_kernel_monitors` is
False), and every moved monitor has a real parity test in
`test_cpp_hybrid_monitors.py` (identical NetworkSet both backends and
engine-vs-reference, rtol 1e-3 / atol 1e-4), including a 210000-step Bold
run that fully fills and wraps the HRF stock, multi-`run()` continuation,
VOI-subset and non-FirstOrderVolterra HRF kernels.

Measured on the 76-node / 20000-step MPR profile: Bold runs 112.8 → 71.3 ms
(1.58×); Bold+Raw+SubSample 110.2 → 75.2 ms; unmonitored kernel runs are
unchanged.

## 4. Stimuli

Stimulus application lives in `_core.cpp` (`stim.get_coupling(step)` values
applied to coupling slots; `backend.py::_build_stim_arrays`), mirroring
nb_hybrid's per-step `sc` injection. `hybrid/stimulus_utils` provides
`constant_stim`, `pulse_stim`, `sinusoid_stim` — any `Stim` works through the
same `get_coupling` interface in both backends.

Gap S1 (CLOSED 2026-09-29): stimuli on multiple subnetworks are now
supported — `sim.run` accepts per-subnet noise/stim sequences (None where
absent); `backend.py` aligns arrays with subnetwork order. Gap N1 (also
closed): multiple stochastic subnetworks now supported the same way.

Related L1: nonzero `model_local_coupling` — **NOT a gap**: nb_hybrid raises
`NotImplementedError` for it too (line ~1942); both backends reject it
identically.

## 5. Priority order for implementation

1. M1 CerebellarMF (CLOSED: hand-written dfun MOD_CRBL=12, parity tested).
2. C2 PreSigmoidal dynamic (CLOSED: id 9 incl. globalT); C1 SJR legacy
   (CLOSED: id 8).
3. S1 multi-subnetwork stimuli + multi-stochastic (CLOSED: per-subnet
   noise/stim sequences); L1 not a gap.
4. Per-model parity tests for the generic-route models (CLOSED:
   test_cpp_hybrid_models.py, 31/31).
5. Per-monitor and per-stimulus parity tests (CLOSED:
   test_cpp_hybrid_monitors.py 10/10, test_cpp_hybrid_stimuli.py 6/6).

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
