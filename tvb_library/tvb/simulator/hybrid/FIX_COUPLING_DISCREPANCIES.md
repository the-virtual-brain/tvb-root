# Remaining Coupling Discrepancies: Hybrid vs Classic TVB

This file documents coupling-function discrepancies between the hybrid simulator
(and its Numba / CUDA backends) and the classic TVB simulator.

**Last updated:** 2026-05-13

---

## Fixed Discrepancies (Phase 2 + Phase 3)

### 1. SigmoidalJansenRit — multi-cvar pre now matches classic

**Status:** FIXED in Python, Numba CPU, and CUDA

**What was wrong:** Hybrid used `e0`/`v0` single-cvar formula; classic uses `cmin`/`cmax`/`midpoint` with two source cvars (`x_j[:,0] - x_j[:,1]`).

**What was fixed:**
- `SigmoidalJansenRit` gained `cmin`, `cmax`, `midpoint`, `use_classic` flag.
- When `use_classic=True` (default), `pre()` reads two source cvars and computes `cmin + (cmax-cmin)/(1+exp(r*(midpoint-(x_j0-x_j1))))`.
- `_cfun_type()` returns `"sigmoidal_jr"` for classic mode and `"sigmoidal_jr_legacy"` for the old formula.
- `_cfun_params()` packs `[a, cmin, cmax, r, midpoint]` for classic and `[a, e0, r, v0]` for legacy.
- Numba CPU and CUDA templates were updated with the new `pre_ct` branches and 2-cvar source access.

---

### 2. PreSigmoidal — dynamic threshold mode now matches classic

**Status:** FIXED in Python, Numba CPU, and CUDA

**What was wrong:** Hybrid only supported static threshold; classic supports dynamic threshold using two source state variables (`P*x_j[:,0] - x_j[:,1]`).

**What was fixed:**
- `PreSigmoidal` gained `dynamic` and `globalT` flags.
- When `dynamic=True`, `pre()` uses two source cvars.
- `_cfun_type()` returns `"pre_sigmoidal_dynamic"` for dynamic mode and `"pre_sigmoidal"` for static.
- `_cfun_params()` packs `[H, Q, G, P, theta]` for static and `[H, Q, G, P, 0]` for dynamic.
- Numba CPU and CUDA templates were updated with the new `pre_ct` branches and 2-cvar source access.

---

### 3. Kuramoto — `1/N` normalization now applied

**Status:** FIXED in Python, Numba CPU, and CUDA

**What was wrong:** Hybrid post was `a * sin(wsum)` without dividing by the number of coupling variables.

**What was fixed:**
- Classic `post()` divides by `gx.shape[0]` (number of coupling variables).
- Hybrid now computes `sin(x_j - x_i)` **per-edge** in `pre()`, then post applies `a * (1/N) * wsum`.
- `_cfun_params()` packs `[a, 1/N]` where `N = source_cvar.shape[0]`.
- `_cfun_type()` returns `"kuramoto"` for the pre phase and `"kuramoto_norm"` for the post phase.
- Numba CPU and CUDA templates were updated with `kuramoto_norm` post branch.

---

### 4. Difference — per-edge `x_j - x_i` now computed

**Status:** FIXED in Python, Numba CPU, and CUDA

**What was wrong:** Hybrid mapped Difference to Scaling (`a * wsum`) without subtracting target state `x_i` per edge.

**What was fixed:**
- `_cfun_type()` now returns `"difference"` instead of `"scaling"`.
- `pre()` computes `x_j - x_i` per-edge before weighting.
- `_needs_xi()` returns `True` for Difference, and `tgt_state` is plumbed into the Numba CPU / CUDA per-edge loops.
- Numba CPU and CUDA templates read `tgt_state[target_cvar[ic], j, mode]` inside the edge loop.

---

### 5. HyperbolicTangent — `b` parameter now included

**Status:** FIXED in Python

**What was wrong:** Hybrid `pre()` used `a * (1 + tanh((x - midpoint)/sigma))` ignoring the `b` parameter.

**What was fixed:**
- `pre()` now computes `a * (1 + tanh((b*x - midpoint)/sigma))`.
- `_cfun_params()` packs `[a, b, midpoint, sigma]`.
- Numba CPU and CUDA templates already had the correct `tanh` formula with the `b` parameter.

---

## Remaining Minor Discrepancies

### R1. Numba/CUDA templates — `tgt_state` is always passed even when unused

**Severity:** LOW (no functional impact)

**What is the case:** To avoid complex Mako conditionals, every `compute_coupling_*` device function now unconditionally receives `tgt_state` as a parameter, and every call site unconditionally passes it. For coupling types that do not need `x_i` (Linear, Scaling, Sigmoidal, etc.), the parameter is simply unused.

**Impact:** None — Numba and CUDA JIT compilers ignore unused parameters.

---

### R2. Classic TVB `SigmoidalJansenRit` — `a` is applied in `post`, not `pre`

**Severity:** LOW (formula is mathematically equivalent)

**What classic does:** `pre()` computes `cmin + (cmax-cmin)/(1+exp(...))`, then `post()` multiplies by `a`.

**What hybrid does:** `pre()` computes the full expression including `a` scaling; `post()` is identity.

**Impact:** None — the overall result is identical because `a * (cmin + (cmax-cmin)/(1+exp(...)))` is mathematically the same as computing the sigmoid first and multiplying by `a` after.

---

## Summary

| # | Issue | Python | Numba CPU | Numba CUDA | Root Cause |
|---|-------|--------|-----------|------------|------------|
| 1 | SigmoidalJansenRit multi-cvar pre | ✓ | ✓ | ✓ | Fixed — 2-cvar per-edge support |
| 2 | PreSigmoidal dynamic threshold | ✓ | ✓ | ✓ | Fixed — 2-cvar per-edge support |
| 3 | Kuramoto `1/N` normalisation | ✓ | ✓ | ✓ | Fixed — per-edge sin + norm post |
| 4 | Difference per-edge `x_j - x_i` | ✓ | ✓ | ✓ | Fixed — `tgt_state` plumbing |
| 5 | HyperbolicTangent `b` parameter | ✓ | ✓ | ✓ | Fixed — formula corrected |
| R1 | Unused `tgt_state` param | — | ✓ | ✓ | Intentional simplification |
| R2 | `a` in pre vs post (SJR) | ✓ | ✓ | ✓ | Mathematically equivalent |

**Legend:** ✓ = fixed / works as intended, — = not applicable
