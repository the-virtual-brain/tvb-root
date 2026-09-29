# cpp_hybrid Feature-Parity Audit vs nb_hybrid

Date: 2026-09-29. Reference surface: `tvb/simulator/backend/nb_hybrid.py`
(`_get_supported_models_classes`, `_cfun_type`, monitor validation, stimulus
plumbing). Target: `tvb/simulator/backend/cpp_hybrid/` (`_core.cpp`,
`backend.py`, `dfungen.py`).

## 1. Models — 26 supported by nb_hybrid

cpp_hybrid has 12 hand-written dfuns (ids 0–11) **plus** a generic route
(`dfungen.py` → `enable_generic_dfuns`, ids ≥ 100) that JIT-translates any
model class declaring `state_variable_dfuns`.

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

No functional gap expected. Work needed: **explicit parity tests** per
monitor class (same config → identical sample streams, float32 tolerance).

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
