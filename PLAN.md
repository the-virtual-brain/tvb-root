# C++ backend delivery plan

## Purpose and operator direction

Replace the previous GUI plan with this recovery-ready plan for the C++ hybrid
backend. Close the existing BW investigation, deliver usable improvements on
`tvbkh`, then support global and local Balloon–Windkessel (BW) monitors and
parameter sweeps, with unit tests and executable demo notebooks.

Execute one phase at a time with **one subagent per phase**, no child delegation
and no concurrent writers. The supervisor owns review, acceptance, this progress
log, and publication. A child must stop at a phase gate and request supervisor
review before the workflow advances. Reuse the same phase child for corrections
where possible rather than creating extra workers/reviewers. Never equate a
child's success claim, a job ID, or a green subset of CI with acceptance.

## Recovery checkpoint

- Initial code baseline: `bb819dc35f31029064b1e8200587aee57569d0cc`.
- Initial root tree: `eb90a95bb118de55307b9b331df1f2d8a8c91e25`.
- Initial local branch: `recovery/monitor-complete-windows`.
- Live GitHub `tvbkh` was verified at that same commit; no unpublished code divergence.
- PR #801 (`tvbkh` -> `master`) was verified open, draft and mergeable; this is not
  publication/release approval or authority to merge.
- `origin` is a stale recovery remote with pushing disabled. Do not change its
  URLs or use its stale `origin/tvbkh` reference as GitHub truth.
- Initial worktree had only untracked `HANDOFF-monitor-windows.md` and
  `cpp-handoff.md`; this plan is now the intended tracked change.
- `cpp-handoff.md` remains the detailed historical evidence inventory. Its
  broad operating/research agenda is superseded by this delivery plan.
- Outstanding Slurm job `5216500` has now been observed **COMPLETED, 0:0**,
  elapsed 00:04:04, node g2. Remote receipt index and both record hashes verify;
  installed provenance reports the expected baseline. Phase 1 has now been
  independently reviewed and accepted with the scoped findings recorded below.

### Phase board

| Phase | Deliverable | State | Evidence / acceptance |
|---|---|---|---|
| 1 | Close existing installed-native investigation | Complete; accepted; report errata finalized | Existing job 5216500; hashes and key raw records independently checked |
| 2 | Green tvbkh baseline, permanent coverage, current-capability demo | Complete; published and accepted at ef1d60b8d | Seven workflows + both Sonar checks pass; dedicated suites/demo and Windows notebook pass |
| 3 | Public local BW Recorders, including independent SIMD lanes | Installed-native qualification passed; publication/exact-SHA CI pending | Job 5216533: 546 passes + demo; sole child held before final acceptance |
| 4 | Compatible multi-subnetwork global BW and shared placement semantics | Pending | Wait for Phase 3 acceptance |
| 5 | Explicit BW parameter sweeps through backend and notebook APIs | Pending | Wait for Phase 4 acceptance |
| 6 | Integrated demos and exact-candidate qualification | Pending | Wait for Phase 5 acceptance |

Current workflow/child IDs and report paths are recorded in the progress log
when available. On recovery, inspect those runs and pending supervisor requests
before launching anything; do not duplicate an active child or Slurm job.

## Fixed scientific and product scope

- Reuse the shipped BW equations, 10 ms held-drive windows and two 5 ms RK4
  steps. Keep current double-precision reductions and hemodynamic arithmetic.
- BW is fractional BOLD change, not the HRF convolution `Bold` monitor.
  Existing `monitor="bold"` sweep behavior must not silently change.
- Regional, single-source-mode BW initially; independent per-region/per-lane
  state with SIMD widths 1, 4, 8 and 16. Width one is not a product limit.
- Explicit finite drive baseline and signed scale; existing nominal shared
  hemodynamic parameter set. Per-region parameter fitting/lowering is deferred.
- Complete windows only, midpoint timestamps, release at the closing boundary,
  no incomplete-tail flush, chunk-independent phase, fresh-run reset.
- Local BW counts eligible source updates on its physical clock. Global BW
  gathers on the master clock and holds slower sources between updates.
  Compare global/local traces only when their effective drives match.
- Preserve public output ordering and established result conventions; make any
  additive output/API contract explicit and test it before implementation.
- Preserve partial drive/BOLD windows and hemodynamic state in compatible
  compiled checkpoint/restore. No new public append/resume API or cross-binary
  snapshot guarantee; no claim of whole-simulator rollback after a failed run.
- Native Linux venv qualification is the development target. Keep existing
  hosted CI enabled. Separate browser/WASM experimentation remains deferred.
- No solver replacement, precision campaign, CUDA work, noisy/surface projection
  expansion, unrelated refactoring or automatic PR readiness/merge decision.

## Phase 1 — Close the existing investigation

**Owner:** one evidence-review child; no production edits and no new jobs.

1. Read the applicable instructions, handoff and r3 launch report.
2. Fetch job 5216500's existing receipt into a new private local evidence
   directory outside the checkout. Preserve originals; verify input/archive,
   receipt and raw-record indexes from their documented working directories.
3. Confirm job identity/terminal status, installed wheel/extension provenance,
   exact-source identity, two completed cases and their lane audits.
4. Inspect reference agreement, midpoint/release grids, repeats, public versus
   compiled equality and chunk independence. Corroborate summary values against
   raw records; inspect harness interpretation rather than trusting its labels.
5. Summarize whole-entry timings and repeat dispersion. Separate source-derived
   memory scaling from process-wide allocator/RSS observations. Do not turn
   BW-minus-TemporalAverage into a kernel ablation or a copy-cost measurement.
6. Conclude: retain RK4 and double precision; proceed with monitor functionality
   unless a concrete correctness failure blocks it. Record unresolved questions
   as deferred, not as automatic follow-up experiments.

**Exit:** supervisor accepts a concise final disposition and evidence reference,
or records the exact first failure/blocker. No automatic rerun, tolerance
relaxation, fixture rewrite or benchmark redesign.

## Phase 2 — Establish the next green tvbkh baseline

**Owner:** one phase child for CI/test/demo changes; supervisor handles Git.

1. Align the local branch with the live GitHub `tvbkh` tip without rewriting
   history; preserve recovery references and handoffs. Inspect fresh exact-tip CI.
2. Audit actual collection of the dedicated complete-window/BW suites. Several
   are absent from the explicit list in `.github/workflows/cpp-hybrid.yml`;
   establish where they run, then add missing permanent gates without duplicating
   expensive work unnecessarily or reducing existing coverage.
3. Add/update a small paired `.py`/`.ipynb` demo of existing public regional BW,
   alongside TemporalAverage, explicit calibration, midpoint times and incomplete
   tails. Explain the difference from HRF `Bold`.
4. Execute the demo through public APIs and make a bounded CI smoke check.

**Exit:** current functionality is demonstrated, relevant regression suites are
permanent CI gates, and the supervisor has reviewed the exact diff and evidence.
Keep this increment separate from local BW implementation.

## Phase 3 — Implement local BW Recorders, tests first

**Owner:** one phase child, serial test/native/public seams with supervisor gates;
no overlapping component writers. The clock/continuation contract ties these
seams together. Agree the test contract before broad implementation.

Implement `Recorder(monitor=BalloonWindkessel(...))` on C++ subnetworks using the
existing native BW engine. C++ owns accumulation, held-drive advancement, BOLD
reduction and release; Python owns configuration, validation and packaging.
Remove blanket local rejection only as the corresponding route becomes supported.

Required tests:

- Public Simulator/Recorder attachment and single-subnetwork local/global
  agreement with matching drives/clocks.
- Nontrivial source cadence; no observations or drive advances on ineligible
  source ticks; independent physical source clocks.
- Chunks crossing drive/output boundaries and no incomplete-tail emission.
- Independent regions and lanes, widths 1/4/8/16, distinct inputs, permutation
  control, independent W=1 comparisons and finite/geometry validation.
- Fresh-run rearm; checkpoint/restore with both partial windows; identity and
  incompatible-state rejection; validate participating Recorder sets before commit.
- Legacy/native siblings and existing global monitors coexist.
- Preserve explicit classic/Numba rejection for unsupported BW execution.

Relevant seams: `hybrid/recorder.py`, `hybrid/simulator.py`, `monitors.py`,
`cpp_hybrid/backend.py`, `_core.cpp`, `cph_monitors.hpp` and existing BW reference
and Recorder tests. Check allocation scaling/overflow across region and lane count.

**Exit:** public local BW works with documented supported profiles, not merely
through a native test hook; focused installed-native tests pass.

## Phase 4 — Complete global BW placement and shared semantics

**Owner:** one phase child, serial gather/native/public validation seams.

Extend the global route to compatible regional multi-subnetwork layouts using
established gather machinery, not a general relaxation of merge rules.

- Require complete/disjoint canonical node mappings and compatible ordered
  observations, clocks and valid cadence. Reject unsupported geometry explicitly.
- Global observations are gathered on every master tick after eligible updates;
  slower sources contribute initial/held state. Local eligibility is unchanged.
- Keep independent regional hemodynamics: do not average neural regions first
  and then treat BW of that average as equivalent to regional BW.
- Support wide independent lanes in this bounded profile.
- Test node/source permutations, mixed cadences, global/local coexistence,
  chunk independence, fresh reset, checkpoint identity and continuation.
- Require global/local parity under matching effective drives; use separate
  oracles for genuinely different mixed-rate sampling semantics.

**Exit:** global and local placements have coherent, tested contracts; supported
profiles and deliberate restrictions are documented without false equivalence.

## Phase 5 — Add BW parameter sweeps end to end

**Owner:** one phase child; get supervisor approval of the additive API contract
before implementing it across backend, result assembly and notebook wrapper.

Today's sweep uses string monitor modes and special BOLD options; merely deleting
BW guards is not support. Define explicit BW configuration (VOI, period,
calibration), global/per-subnetwork selection, output axes/time grids/node order,
and sweep-row ordering. Preserve existing HRF `bold` behavior.

Wire the supported contract through `CppHybridBackend.sweep` and
`cpp_hybrid/notebook.py`. Preserve legacy `run_sweep` behavior unless explicitly
extended; unsupported requests must fail rather than be silently ignored.

Required tests:

- Widths 1/4/8/16 versus independent non-sweep runs.
- Partial final lane groups, padded-lane exclusion, lane/parameter permutations,
  worker-count independence and repeated calls.
- Paired and Cartesian grids retain parameter/result order.
- Local BW across different source clocks and compatible global BW output.
- Explicit configuration errors and unchanged raw/tavg/HRF sweep behavior.

**Exit:** BW sweeps are usable from public APIs, including the notebook wrapper,
with interpretable results and no native-internals workaround.

## Phase 6 — Integrated demos and qualification

**Owner:** one phase child; supervisor reviews exact integrated source/evidence.

Add or extend focused paired notebooks to show:

1. Global and local BW in one hybrid network.
2. Different source clocks with accurate timing explanations.
3. A small neural/coupling parameter sweep with genuinely distinct drives.
4. Selected sweep members checked against independent runs.

Use small deterministic fixtures, no external downloads, explicit calibration,
shape/time/finite-value assertions and scientifically meaningful plotted duration.
Keep executable smoke tests bounded; avoid unsupported performance claims.

Qualify the exact candidate in fresh installed-native venvs on Slurm, including
reference traces, cadence, lanes, lifecycle, invalid state, relevant backend/hybrid
regressions and notebook execution. Make one bounded baseline/candidate time and
memory comparison with repeats; do not reopen allocation/copy attribution research.

**Exit:** supervisor accepts the integrated diff and exact-candidate evidence;
exact-SHA hosted CI is green for the published increment, and residual platform
limits are explicit. No automatic merge or release.

## Execution, evidence and publication rules

- Read applicable `AGENTS.md` before edits. The operator's current direction is
  incremental delivery to `tvbkh`; children may not commit, push, merge, rebase,
  change remotes, publish comments, or weaken gates. Supervisor retains Git
  authority and checks any conflict with repository policy before publication.
- Keep one active phase child, no nested delegation and no silent model/CLI
  fallback. Parent review is the independent phase gate; do not add reviewer
  agents contrary to the one-child-per-phase instruction.
- Children do not edit PLAN.md. At a checkpoint they stop writing and contact
  the supervisor with changes/evidence/blockers. The parent reviews and updates
  PLAN.md while the child is explicitly quiescent, then accepts or requests fixes.
- Tests are developed first, but failing intermediate states stay local. Publish
  reviewed, locally qualified increments only to `tvbkh`; inspect their exact-SHA
  CI and fix failures before advancing. No force, merge, remote rewrite or auto-merge.
- Any approved push uses explicit `git@github.com:the-virtual-brain/tvb-root.git`
  and `HEAD:refs/heads/tvbkh`, with fresh ref/ancestry/diff and dry-run checks.
- Do not stage local handoffs, raw receipts, machine settings, credentials or
  generated benchmark outputs. Keep operational evidence outside the source tree.
- Heavy builds/tests only on Slurm via `inslurm-proxy`, private exact-source
  inputs, node-local builds and fresh installed environments with verified imports.
  One explicit submission at a time; supervisor approves a concrete job request.
- No new job is authorized for Phase 1. Later jobs require a bounded case/test
  matrix and resource request approved by the supervisor before submission.
- Verify actual completed tests, selectors and expected skips, not collection
  counts. Keep raw bytes before reduction; diagnose failures before any retry.
- Async child completion wakes the supervisor. Yield rather than sleep/poll.
  Stop and report launch/runtime failures; preserve partial changes and evidence.

## Progress log

### Plan established

- Operator explicitly requested overwriting PLAN.md, one subagent per phase,
  supervisor review and ongoing progress updates.
- Previous GUI plan replaced; it remains recoverable from Git history.
- No production code changed and no additional Slurm job submitted.
- Sequential async workflow launched: `f65dea8b-85fd-424d-a5a1-5d143b64ec48`;
  mission `3e979de0-5354-4b3c-9820-f2d3df354bb6`.
- Stable child keys: `cpp-phase-1` through `cpp-phase-6`. Reports are bound to
  managed workflow artifacts `cpp-delivery/phase-N.md`, not the source tree.
- Child model: `blablador/deepseek-v4.1-flash-at-hzdr:medium`, matching the
  handoff's worker routing. No fallback is authorized.
- Each phase pauses for a blocking supervisor review before returning accepted;
  the workflow stops on failure/blocked status. Children cannot edit this plan
  or publish. Supervisor updates the board at each gate before advancement.
- Phase 1 review and Phase 2 authorization are recorded in the next entry.

### Phase 1 — supervisor acceptance and investigation closure

- Child run: `5e39cb94-dc18-49cd-aef4-659b75d8d5a7` (workflow key `cpp-phase-1`).
- Candidate report: managed workflow artifact `cpp-delivery/phase-1.md`.
- Evidence: `../recovery/verify/monitor-phase2-native-installed-bb819dc-r3/receipt-5216500-20261126T162115Z/`.
- Parent independently verified all **31 receipt-index entries** and both raw
  record hashes; receipt index SHA-256
  `1d3f60d041c3e225b07a3e65622102daae2e3a51cf5b693013d32af35375bcfb`,
  records index `1c371e919076e66615eb530c10994f462c1d83332f372ce7d441b7300b21ffe7`.
- Parent read the raw reference/chunk/entry records and recomputed timing medians
  and contrasts from raw timings. No simulator/oracle replay or new job.
- Scoped accuracy finding: 8 regions, 60 simulated seconds, 6000 drive windows,
  30 BOLD rows; maximum difference from the recorded float64 DOP853 reference
  `4.5598143143255054e-10`, maximum `0.499206949` float32 ulp. All 240 values
  match the reference's float32 casts. This is agreement with that numerical
  reference on this fixture, not a general exact-solution rounding guarantee.
- Midpoint/release grids agree; public/compiled output digests agree; default,
  1000-step and 7-step chunk outputs agree. End-state partial-window diagnostic
  was unavailable due to harness ordering, so it supplies no continuation proof.
- Compiled BW median `29.619563 ms` versus matched TemporalAverage `11.252251 ms`;
  difference `18.367312 ms` for this whole entry, not isolated monitor/kernel cost.
  Public BW median `29.898517 ms`. No copy-cost or per-emitted-row attribution.
- Decision: **investigation closed; retain shipped RK4 and double precision**.
  No blocking product finding or new experiment is required before feature work.
  Phase 3 must test partial-window continuation directly.
- Report-only errata requested: receipt count is 31, not 29; error is below but
  near half an ulp, not far below; scope cast agreement to the numerical oracle;
  qualify cost-scaling observations rather than asserting causal attribution;
  do not infer simulator throughput from a job dominated by build/reference work.
- While the phase child was quiescent, supervisor created local `tvbkh` at the
  unchanged baseline commit, preserving the recovery branch and all handoffs.
  No commit, stage, push, remote change or history rewrite occurred.
- Fresh GitHub check: live `tvbkh` still `bb819dc35`; all reported applicable PR
  checks pass (release upload intentionally skipped). The child must inspect
  exact-SHA CI collection, not treat a green list as exhaustive test coverage.
- Phase 1 child finalized the requested report-only errata and returned accepted.
  Supervisor read the amended saved result; no rerun was needed.

### Phase 2 — startup contract and bounded validation authorization

- Sole phase child: `623cdd9a-1924-49f0-81e8-6b9ddc83a673`, key `cpp-phase-2`.
- Confirmed base is local/live `tvbkh` at `bb819dc35`; no branch or publication
  prerequisite remains before source edits. Publication stays supervisor-owned.
- Audit finding: seven BW/window suites already run in the broad Python workflow
  on supported Python versions. This is dedicated-gate visibility/durability work,
  not a claim of previously absent coverage. Python <3.12 deliberately ignores
  the native test glob; the reference-only suite still runs there.
- Approved CI changes: correct stale cpp-hybrid workflow header; add a separate
  named step with the seven explicit reference/BW/window module paths; preserve
  all existing selectors; add matplotlib and a bounded demo-script smoke step.
  Prefer explicit selectors over a widening glob for this increment.
- Approved new paired demo: `tvb_documentation/demos/cpp_hybrid_balloon_windkessel.py`
  and `.ipynb`. Public C++ regional BW plus TemporalAverage, finite explicit
  calibration, deterministic three-region input and no downloads/noise. Explain
  BW versus HRF Bold and show calibration from 10 ms means.
- Supervisor adjustment: the main plot must span a meaningful 30–60 simulated
  seconds with a sensible 1–2 s BOLD period and bounded drive, not just the
  proposed 200 ms initial transient. Keep short exact-grid/tail/zero-scale checks
  separately (including the 195 ms incomplete-tail case). Do not add a precision
  oracle or performance experiment to this demo.
- No extra pytest module required for identical smoke coverage: existing suites
  plus executable notebook/script assertions cover this phase. The Windows
  notebook runner automatically executes this new top-level notebook as well.
- Approved **one** Slurm submission after the edits and lightweight static
  checks: partition run, one task, eight CPUs, 16 GB, 30 min, no pinned node;
  node-local fresh build/test venvs and a noneditable wheel. Run the seven named
  installed-package suites, demo script and notebook, capture completed outcomes,
  actual allocation, package versions and installed binary/source provenance.
- Archive actual candidate bytes using an explicit allowlist and hashes; include
  tracked package sources plus the approved new demo pair and necessary runner
  inputs. Exclude handoffs, ignored build products, credentials and arbitrary
  untracked files. Bind provenance to baseline + candidate diff/file manifest,
  not a falsely claimed committed Git tree. Freeze functional inputs during job.
- Test working directory/sys.path and notebook kernel must resolve the installed
  test-venv package, not the extracted source or another/global Python kernel.
  Use a bounded notebook execution timeout and record kernel/import provenance.
- Fetch-back of one executed notebook is approved for review, provided source
  cells match the tested input, `.py`/`.ipynb` cell pairing remains exact and
  outputs/metadata contain no private paths or credentials. Record input/output
  hashes separately; output-only changes do not claim identical whole-file bytes.
- Stop/report the first build/test/runtime failure; no automatic retry, skip,
  gate/tolerance relaxation or second submission. No commit/push authorized to
  the child. The first submitted job failed during setup; see the recovery entry
  below. Final diff and exact receipt review are still required for Phase 2.

### Phase 2 — first validation failure and correction gate

- Exactly one Phase 2 job submitted so far: `5216501`, node g2, exit 127 at
  `cmake --version`. Build dependencies include cmake 4.4.5 and ninja 1.13.2,
  but the harness omitted exporting the build-venv bin directory onto PATH.
  No wheel build, install, selected suite, demo or notebook execution occurred.
- Failed receipt preserved as-is:
  `../recovery/verify/monitor-phase2-demo-bb819dc-r1/failed-5216501/`.
  Supervisor independently checked its 12-entry preserved-files index and all
  five frozen input hashes; submitted input-index SHA-256 is
  `852ad354c536413ecdfaf944e6d693d60eaf4bc878ec8b4932166dfa92d600e8`.
- Library archive/source identity remains the baseline; new demo source and
  notebook still match the submitted copies. This failure is harness setup,
  not evidence of a product/test failure or a successful validation.
- Parent found a deterministic next-stage conflict before retry: demo prints
  its absolute installed path, while the executed-notebook text scan rejects
  private /tmp or home paths. Imported warning text can also contain paths.
  Keep actual interpreter/import provenance in private receipts; make published
  notebook output safe without hiding warning messages or execution errors.
- Parent also found demo wording that incorrectly calls ten emitted 20 ms rows
  over 200 ms ten drive windows: there are twenty 10 ms drive windows. Correct
  that explanation before re-freezing the candidate.
- Authorized correction preparation only: restore build-venv PATH, resolve
  notebook path-output/privacy conflict, and fix drive-window versus row wording.
  Preserve original failed harness/inputs and receipt; use a fresh r2 directory.
- **No second submission authorized yet.** Same phase child must present the
  compact correction diff and lightweight static/pairing/privacy checks before
  supervisor approves one corrected job. Do not repeat broad scouting, alter
  solver/precision, weaken numerical tests, blanket-suppress warnings, relax the
  privacy requirement or automatically retry. Report changed input hashes
  honestly; the retry will not have byte-identical demo inputs after these fixes.

### Phase 2 — corrected r2 submission approved

- Supervisor read r2 job script, notebook provenance helper and demo correction
  diff; independently verified all six r2 frozen inputs, all five original r1
  inputs and the preserved r1 failed-receipt hashes. No simulator/test replay.
- Corrected input-index SHA-256:
  `0700d02ab951b66d8cff64b383e75a2591093115d8bc9e73806c8d91038f801e`.
  Library archive remains unchanged. Demo source SHA-256:
  `5870d0779254e7cbd5f1c16aa94006839b7e208425f6e151383613c985ce5443`;
  source-only notebook `fded0c9449e4dc4cfe4637140c7c17ffeb80b28f503d460a67bb4744bc547d43`;
  job script `21d6137fc988a1fbb13d3c6a63bb2a8f3cc0272c0eb286760cf6b22ecade1e8a`.
- PATH defect corrected; public path print removed; drive-window/row explanation
  corrected. Runtime matplotlib cache stays node-local. Notebook kernel is the
  test-venv interpreter with a portable python3 kernelspec.
- Raw executed notebook is preserved separately. Publication normalization only
  replaces declared scratch/receipt/home prefixes in text outputs; tested sources
  must stay unchanged. Published output scan has no prefix allowances. This
  helper checks specific text/metadata fields, not a general privacy guarantee;
  supervisor will inspect final notebook metadata and outputs at acceptance.
- Authorized exactly one corrected r2 staging/submission in a fresh private
  remote directory after verifying the above hashes remotely: same partition,
  task/CPU/memory/time request and unchanged seven-suite/demo/notebook scope.
  This is the second Phase 2 submission, not authorization for further retries.
- Child must report the new job ID and receipt path (or first failure), preserve
  r1 unchanged, and present final diff/receipts before Phase 2 acceptance. No
  tolerance/selector changes, no publication, no Phase 3 advancement yet.

### Phase 2 — r2 suite success, harness count failure, bounded continuation

- Second Phase 2 job `5216502` ended FAILED 1:0, elapsed 49 s, node g2.
  Supervisor read remote accounting, run/input/provenance records, wheel hash
  and suites log. Build, installed import and selected suites succeeded:
  **440 passed, 3 warnings in 9.16 s**, with no reported skips/fails/errors.
- The harness then rejected that result because its earlier hosted-log audit
  had incorrectly set EXPECTED_TESTS=387. Wrapped log IDs were miscounted; the
  correct selected-suite baseline count is 440. This is a harness-count error,
  not a product failure; do not change tests/selectors/tolerances to match 387.
- No demo script or notebook ran; no DONE marker exists. Preserve original job
  failure and receipt as-is. Do not relabel the whole job successful.
- Wheel artifact:
  `tvb_library-3.0.0a1-cp312-abi3-linux_x86_64.whl`, SHA-256
  `8b313a54f014586c28a486c09bb4fd6db723d4f8639037a34f277716fea465aa`.
  Embedded extension SHA-256 matches earlier qualified baseline binary:
  `9e7a6532c6d5098ab4b091e9a6cf0a47844c740054aea83f49c1efd5e877c3ee`.
  Suites log SHA-256:
  `6288cde391e7eaecef56bd96c5ad9d64b1f0700c41699705122662a5fe4cefc7`.
- Supervisor rejects a full rebuild/test replay solely to fix the count.
  Authorize one **continuation-only** third job for the unexecuted demo script,
  notebook, source pairing and publication hygiene checks. Install the exact
  preserved wheel into a fresh node-local test venv; no source build, selected
  suite replay, collect-only campaign, solver work or benchmark expansion.
- Because the saved wheel uses -march=native from g2, this artifact-continuation
  job may explicitly request **g2** (a documented change from prior unpinned
  jobs, not a portability/performance qualification). Other resource limits
  remain run / one task / eight CPUs / 16 GB / 30 min. No alternate-node fallback.
- Freeze/hash the exact saved wheel, unchanged r2 demo pair and pairing/privacy
  tools, and minimal continuation runner. Keep r2 receipt/input binding and the
  new receipt linked by hashes, not by a rewritten success marker. Use recorded
  dependency versions where practical and report actual resolved environment.
- Fetch/preserve the partial r2 receipt outside the source tree before treating
  its suite evidence as the accepted component of the final phase gate. Final
  acceptance combines r2's existing 440 passes with continuation results; it
  must explicitly say these came from separate jobs/environments.
- If the continuation fails, stop/report it; no further submission is preapproved.
  Phase 2 was not yet accepted at this checkpoint; see the publication gate below.

### Phase 2 — local acceptance and publication gate

- Continuation job `5216503`: COMPLETED 0:0, 18 s on requested g2. Same saved
  wheel installed into a fresh venv; installed ELF hash equals wheel entry;
  recorded dependency comparison is empty. Demo and notebook completed.
- Supervisor inspected workflow/source diff and full notebook text/metadata;
  pairing is exact (14 cells, source hash
  `7905e0601c057c6b19d5ced43d8f693a3051cb5e60950c87f0553ec6320283db`).
  All 23 continuation receipt-index entries verify. No errors in cell outputs,
  no private paths in publication text/metadata, portable python3 kernel.
- Publication notebook SHA-256:
  `72098edce87f7d4304e134043068618cf0c00f6dd5c6dfd607a1e05a59d8a271`;
  byte-identical to reviewed saved publication artifact. Observed 40 s fixture
  peak absolute fractional BOLD `0.00790291465818882`; exact midpoint/tail/rest
  assertions pass. No additional oracle claim is needed or accepted here.
- Evidence directories:
  `../recovery/verify/monitor-phase2-demo-bb819dc-r2-receipt-5216502/` and
  `../recovery/verify/monitor-phase2-demo-bb819dc-r3-receipt-5216503/`.
- Fetch-index defect recorded, not hidden: r2's local
  `remote-receipt-files.sha256` has 18 payload entries that match, plus an invalid
  stale self-hash entry. The original index is preserved; a blanket successful
  shasum verification is NOT claimed. Parent independently checked suite/wheel
  hashes against remote originals; this does not invalidate those payloads or
  justify a test replay. r3's 23-entry in-job index fully verifies.
- Accounting independently confirmed: 5216501 FAILED 127:0 / 4 s (PATH);
  5216502 FAILED 1:0 / 49 s (count gate after 440 passes);
  5216503 COMPLETED 0:0 / 18 s (remaining demo/notebook). Do not conflate them.
- Local source/evidence accepted for publication under the operator's direction
  to keep delivering to tvbkh. Parent publishes only PLAN.md, the workflow and
  paired demo; no handoffs, receipts, harnesses or production source changes.
- Phase 2 remains gated on the resulting exact commit's applicable hosted CI.
  Child stays quiescent except report-only errata. No Phase 3 advancement, new
  Slurm submission, merge or release is authorized until this gate is resolved.

### Phase 2 — published; hosted CI in progress

- Parent committed and pushed the four reviewed files as
  `ef1d60b8d3d21f74b723023d17820bcabd98185d`
  (parent `bb819dc35f31029064b1e8200587aee57569d0cc`),
  "Demonstrate regional Balloon-Windkessel and name its CI gates".
- Publication used explicit GitHub URL and HEAD:refs/heads/tvbkh after fresh
  remote identity/ancestry checks and successful dry run. Live GitHub tip was
  verified equal to the new commit. No force, merge, remote edit or handoff staging.
- Exact-SHA Actions runs observed in progress:
  cpp-hybrid `45863926002`, Test Py `45863926025`, Test Notebooks `45863926353`,
  Test PG `45863926213`, Test Win `45863925843`, contrib `45863926024`,
  Wheels `45863937226`. Hosted/Sonar acceptance remains pending.
- Phase 2 child remains the sole phase worker and may monitor these existing
  checks read-only; no new jobs/source changes or Phase 3 advancement. On a
  failure, inspect exact-SHA failing logs and report before proposing fixes or
  reruns. Parent independently dispositions final results before acceptance.
- Attention reviews: the child is intentionally running `gh pr checks --watch
  --fail-fast`, not stuck in a build or source write. The first 2400 s watch
  timed out while CI was still running; this is not a hosted check failure.
  Latest parent exact-SHA check: Wheels, contrib, cpp-hybrid, Test PG and
  Test Win succeeded. In Test Py, Python 3.13 has passed; Python 3.11/3.12
  remain in progress. Test Notebooks is running its notebook-execution step.
  Dedicated C++ logs show the new seven-suite step (440 passes) and BW demo
  step passed. Subsequent watch attention reviewed: still an intentional CI
  watch, no failed check observed; no source intervention, CI rerun or new worker.
- These log updates are local operational progress after the published commit;
  they do not change the tested workflow or demo bytes.

### Phase 2 — exact-SHA acceptance; Phase 3 authorized

- Parent read the saved Phase 2 report and independently queried exact commit
  `ef1d60b8d3d21f74b723023d17820bcabd98185d`: all seven workflows above are
  completed/success. SonarCloud check `140098397352` and SonarCloud Code
  Analysis check `140112711734` are completed/success. Release-upload job is
  deliberately skipped, not a missing required qualification.
- Parent independently verified both added dedicated C++ steps succeeded and
  found the new notebook's successful execution line in Windows run 45863926353.
  The dedicated suite recorded 440 passes with three existing warnings; all
  three broad Python matrix jobs also succeeded.
- Live GitHub tvbkh and PR #801 head still match the exact reviewed commit.
  PR remains open and draft; acceptance does not authorize merge/release.
- **Phase 2 fully accepted.** Existing failures and manifest caveat stay on
  record; no test/CI rerun or evidence relabeling was needed for acceptance.
- Sole Phase 2 child may finalize its saved report and return accepted. The
  existing sequential workflow is authorized to start its sole Phase 3 child.
- Phase 3 starts from the accepted published commit. Before broad implementation
  the child must present local Recorder tests/clock semantics, per-region/lane
  isolation, Python/native seams, continuation identity and bounded validation
  matrix for parent review. No Slurm job or future-phase API expansion is
  preapproved. Keep RK4/double arithmetic fixed; no new precision investigation.
- Local PLAN.md progress remains the only tracked uncommitted change at this
  gate. Include reviewed progress with the next functional delivery instead of
  creating a documentation-only push that needlessly restarts the full CI suite.

### Phase 3 — tests-first contract reviewed; implementation authorization

- Sole child `cdca7fa2-7605-4ab2-b004-042160ea351b`, key `cpp-phase-3`, starts
  from accepted ef1d60b8d. Initial changes are the new local BW test module and
  scoped updates to prior public BW guard tests; no production code/job yet.
- Supervisor read the startup report, new tests and existing clock/restore
  seams. Approve public C++ local Recorder support, source-eligible observations,
  10 ms physical drive windows, shipped two-step RK4/double arithmetic, complete
  output windows, independent regions/lanes and compatible monitor-set restore.
- Kind 11 local BW and a BW-specific version-2 continuation identity are in
  scope. Reuse the current BW engine/state structure where practical; do not
  introduce a second solver. Preserve version-1 identities and existing width
  restrictions for local average monitors; do not loosen global/sweep guards.
- Distinct-lane proof through native Sim plus broadcast-lane public runs is an
  acceptable Phase 3 gate. Label that scope honestly; public distinct-parameter
  sweep routing and independent-run parity remain Phase 5 deliverables.
- Adding the new module to the dedicated CI step is approved (seven -> eight).
- Parent identified mandatory corrections in the initial tests, not production
  behavior changes to satisfy erroneous expectations:
  * Same physical duration and same 20 ms output period yield the same output
    count on fast/slow sources when both source clocks divide the drive window.
    Slow sources contribute fewer eligible observations, not half as many rows.
  * Master dt is min(subnet dt). A lone slow subnet has cadence one. Use a fast
    companion to establish a nontrivial source cadence, including restore tests.
  * Compute the global parity reference before installing the tracing wrapper
    that deliberately forbids global-output draining; trace local work only.
  * At dt=2.5 ms, 13 steps span 32.5 ms and emit one 20 ms row, not zero.
    Six native steps span 15 ms and cannot emit three 20 ms rows; use 24 steps
    (six drive windows) for that lane test, with integer step counts.
  * At seven source updates, the second drive window has three pending samples;
    its drive sum is nonzero. Assert the actual partial sum rather than zero.
    Calls with return_snapshot and subnet_monitors return three values, not two.
- Strengthen all-set restore coverage with multiple monitors and a valid changed
  earlier state followed by an invalid later state; use a wide-lane corruption
  case. Add a late-region/lane close failure check showing no partial BW close
  commits, without promising neural-state or whole-simulator rollback.
- Resources/scope approved in principle: one run-partition job, one task,
  eight CPUs, 16 GB, 30 min, no node pin, exact-source installed wheel/fresh venv.
  Include proposed eight modules, existing global co-clock window regressions,
  and the current demo script. No rerun of Phase 1 benchmarks.
- **Submission still gated:** implement serially after correcting test logic;
  present corrected test/implementation diff and frozen runner inputs for parent
  readiness review before the single submission. Use pytest exit status and
  per-test outcome evidence; do not hardcode a guessed count from wrapped logs.
  Preserve failures; no retry or tolerance/selector relaxation preapproved.

### Phase 3 — first readiness review; submission withheld

- Child reports implementation complete, scientific engine unchanged, all six
  startup corrections addressed, and local syntax/header-level checks passing.
  These header-level checks are not installed-native qualification or acceptance.
- Parent read the actual production diff, all new tests, managed readiness
  report and full runner. `git diff --check` passes; HEAD remains ef1d60b8d.
  No Slurm submission, native package build or test replay by parent.
- Frozen first package is preserved at
  `../recovery/verify/monitor-phase3-local-bw-ErAJgc/`. Parent independently
  matched reported hashes for archive, demo, runner and candidate patch:
  archive `e398e52e796076ca4709d227fb42ea37d333c3889fa03ba6669bec8566b2a4ad`,
  runner `0c39e4f877277d6c7bd0edd9c08af66063e231c3a8e22e146c7847a91d7872f0`,
  patch `d91a954c690fd7ab5f73d524e2e0d34a17cee0f65727705497aec08792a4b97f`.
  These are reviewed candidate inputs, not a qualified or committed revision.
- **Do not submit this package.** Parent found concrete remaining blockers:
  * BW output reservation omits pending `bw_drive_count`. From a seven-update
    state at dt=2.5 ms / period=20 ms, one eligible update emits a row but reserves
    zero rows. Include pending drive observations so output allocation cannot
    first happen after clock/state commit; add a bounded capacity assertion.
  * BW restore bypasses `dict_to_checked_window_state` and therefore weakens
    exact-integer/float64 accumulator checks of local Recorder snapshots. Reuse
    strict validation (empty extent for BW) plus its BW-specific state checks;
    add malformed type/empty-accumulator regression cases.
  * Fresh-rearm test reads its supposed 13-step warm state after the second,
    nine-step run, then asserts 13. Capture warm state before the second run.
  * Two tests expect `requires native monitor collection`, while the new local
    error says `Recorders require native monitor collection`. Keep the existing
    diagnostic substring consistently; no guard or selector relaxation.
- Also require integer restore step counts and accurate physical-duration
  comments (48 master steps are 120 ms, not 300 ms). Keep intended coverage,
  independently sampled clocks and unchanged numerical tolerances.
- Runner matrix/resources are suitable. Strengthen provenance before re-freeze:
  verify installed extension resides in the fresh test environment and hashes
  exactly to its wheel member; include/check candidate.patch in frozen inputs
  instead of only logging its expected hash from an environment variable.
- Authorize only these bounded fixes and a fresh frozen package. Preserve the
  first package unchanged; re-present corrected diff/hashes before submission.
  No new investigation, broad refactor, retry, job, publication or next phase.

### Phase 3 — corrected frozen job approved once

- Same sole child presented corrections and paused before submission. Parent
  reviewed correction.patch, full corrected runner and managed report; checked
  working-tree status, unchanged HEAD and `git diff --check`, plus `bash -n`.
  No build/test/oracle replay was performed by parent.
- Pending-observation reservation is corrected; saved capacity logs show the old
  header reserves zero rows and the corrected header reserves one row / three
  values for the seven-update/one-more-tick case. Strict local snapshot parsing
  now uses the existing checked helper before parsing BW state. Five malformed
  snapshot-type cases were added. Warm state is captured before the second run;
  disable diagnostic, integer horizons and duration comments are corrected.
- Runner now verifies the installed extension's fresh-venv location and exact
  wheel-member hash. Candidate patch is in the frozen index and separately
  checked against the supplied patch identity. Nine-suite JUnit/outcome gate,
  zero skips and existing demo remain unchanged.
- Fresh private package:
  `../recovery/verify/monitor-phase3-local-bw-qXlR3a/`.
  Parent independently verified all four indexed inputs and correction hash:
  * archive `2436fb1c161f365b4271ca558d0578bbdb2e74c3392b194b4e9e9f7342be0fa4`
    (5,339,648 bytes);
  * unchanged demo `5870d0779254e7cbd5f1c16aa94006839b7e208425f6e151383613c985ce5443`;
  * runner `e4b4e9e75998768bcdf6dcc58e1e401f38e2ac5f1410bfadce2b992c755123cd`;
  * candidate patch `c4e27648c006906f44875a2392097d5ed99c1a7500a05e8d606e5a43a18e9b74`;
  * input index `71ad80130e11926dbe6d2948f297de848e2c0e13d73630660ac219abcdeb02f0`;
  * correction `eae5383a510df815d2ad28ccfe40d8d0ee655127b8ec6e2f8abbc1186eee5635`.
- Parent compared every archive source payload byte with the checkout: all 326
  tracked library files plus the approved new test match, no missing/unexpected
  source payload. Recomputed tracked/new-test diffs concatenate exactly to the
  frozen candidate.patch. Old package archive/runner/patch hashes are unchanged.
- File-count clarification: the archive contains **327 source payloads plus
  327 generated AppleDouble metadata entries**, 654 regular entries total.
  Parent checked the metadata magic; these are tar-generated metadata, not
  extra scientific source. Keep private frozen inputs unchanged and report
  actual extracted manifest counts; do not describe 327 as total archive files
  or publish these metadata-bearing private artifacts.
- **Approve exactly one submission** of the above frozen job via inslurm-proxy:
  run partition, one task, eight CPUs, 16 GB, 30 min, no pinned node. Verify the
  approved identities remotely before submission; freeze functional source.
  Report job ID and receipt path, preserve all evidence, stop/report first
  failure with no automatic retry, gate relaxation or harness/source mutation.
- This is validation authorization only: no Phase 3 acceptance, commit/push,
  future phase, extra local numerical checks or additional job is authorized.
  Parent reviews installed-native receipts and final diff before publication.

### Phase 3 — first installed-native failure; corrections-only authorization

- Approved job `5216532` ran once, node vep4, reported FAILED 1:0 after 58 s.
  Receipt: `../recovery/verify/monitor-phase3-local-bw-qXlR3a/receipt-5216532/`;
  remote `inslurm-proxy:~/verify/monitor-phase3-local-bw-qXlR3a/local-bw-5216532/`.
  Child stopped without edits/retry. No demo execution or Phase 3 acceptance.
- Parent read saved report, failure log, outcome/provenance/input records and
  verified all 21 local receipt-index entries. Index SHA-256:
  `02e721e36e562bd1c079764c43655f3765436facd68138d7448d7c96209b8ba0`.
  Source manifest `d618afc7c45f277a511e6ee5cf0dedb15a042139f36ab3e29e6d8eab11679f01`
  has 654 entries, with the metadata distinction recorded above.
- Build/install/provenance succeeded. Wheel SHA-256:
  `e0db5b8ef969612ae527df0556f482ac63cb4391709f11a8307cd5f060a43352`.
  Parent independently extracted native wheel-member hash
  `a549b8d578329fd26f43e5ab57a541d379569a3c8ce1abbd934109678c190ffd`,
  matching recorded installed hash. All nine selected wheel test files match
  checkout bytes. No simulator/test execution was repeated by parent.
- Actual result: **537 passed, 7 failed, zero skipped**. JUnit has empty
  classnames, so the intended per-module gate cannot attribute its cases.
  Parent instead mapped case base names to AST definitions in the exact wheel
  source: every case has a unique owner, no ambiguity. Existing modules passed:
  reference 53, BW stream 79, complete windows 86, global co-clock 64, native
  monitors 22, public BW 57, public windows 42, Recorder windows 59. New local
  BW module has 75 passes / 7 failures (82 cases). This is forensic attribution,
  not acceptance of the broken classname-based runner gate.
- Logs/source support the child's fixture diagnosis: parity and mixed-sibling
  comparisons use one too few global-output nesting levels; three distinct-lane
  controls pass strided slices into a flat-buffer native API; NaN is rejected
  by traits at construction rather than at the intended runtime assertion; a
  multi-mode model fixture hits an earlier dfun fingerprint guard and prevents
  its combined native-disable assertion from running. Those seven intended
  behaviors remain unqualified until corrected tests pass.
- Authorize test-only corrections: correct global nesting, use/assert contiguous
  float32 lane inputs with unchanged exact comparisons, use trait-accepted
  nonfinite calibration for the runtime guard, and test the Python multi-mode
  spec seam explicitly without claiming this substitutes for public integration.
  Also add a direct native non-one-mode attachment rejection and split disabled
  collection into its own test so it always runs. Do not touch production code,
  numerical tolerances or existing suite selectors for these fixture fixes.
- Replace fragile classname attribution with nine separate invocations in the
  same job, each explicitly labeled by its selector and own log/JUnit/status.
  Verify each selector resolves to installed test-venv code, each has completed
  cases and zero skips/failures/errors; stop at the first failed invocation.
  Preserve the nine-suite matrix and final demo; no guessed test count or
  reliance on name-only attribution as the permanent runner gate.
- Pre-existing hazard recorded, not silently dismissed: native resolve_per_sn
  accepts strided noise then uses flat-buffer indexing. Deferred outside this
  Phase 3 correction; revisit contiguity validation before Phase 5 public lane
  sweep qualification. The corrected lane tests must explicitly use contiguous
  buffers; their evidence will not establish support for strided native inputs.
- **No resubmission yet.** Preserve failed receipt and frozen packages; prepare
  fresh exact-source inputs, compact test/runner diff, hashes and static checks
  for readiness review. Same resources remain suitable. One concrete retry may
  be approved after review; no automatic rebuild/job/retry or future phase.

### Phase 3 — test/harness correction package; one retry approved

- Same child implemented the approved fixture corrections, split the disabled
  collection test, added both Python-spec and native multi-mode rejection tests,
  and paused before submission. Parent read the compact diff, full runner and
  saved report. Exact parity/lane/permutation/sibling comparisons are retained;
  production finite/geometry guards and numerical tolerances are unchanged.
- Parent verified all indexed inputs and correction hash, `git diff --check`
  and runner `bash -n`. Compared every archive source payload to the checkout,
  checked the approved allowlist and recomputed the full candidate patch. Against
  qXlR3a, the **only changed source payload is the new local BW test module**;
  all production code and eight earlier suites remain byte-identical.
- Retry inputs: `../recovery/verify/monitor-phase3-local-bw-QNKBhL/`:
  * archive `fc4311ff8cea872b4d86e53e83a7870bd8006107223a79c59b52df85b6f578d8`
    (5,342,208 bytes; 327 source payloads + 327 AppleDouble metadata entries);
  * unchanged demo `5870d0779254e7cbd5f1c16aa94006839b7e208425f6e151383613c985ce5443`;
  * runner `214bd783090d875721053fab5c47a1edc7c1c1cd7154a72cb4e7152e6e83c4d6`;
  * patch `793c613dc68687ef4c7e3dd24ef31f6c35fff1c4a111adefa2f4665ec51cbe8c`;
  * index `d39169b3a21761e11dcfb84cb8ae2b8ac670499b8f7c791a3e5168cfa343a818`;
  * correction `b65f0f0ebd1ffd803824804757ce980cfd867ece87b0d0df98117101275df860`.
- Runner uses nine explicit serial module invocations, each with installed-origin
  verification, own selector/log/JUnit/status and zero-failure/error/skip gate.
  Any unclean invocation stops the job; only nine clean invocations reach the
  aggregate gate and unchanged demo. No guessed test count or JUnit classname
  attribution. Fresh exact-source wheel build/test environments remain required.
- **Approve exactly one retry submission** of these frozen bytes via
  inslurm-proxy, run/one task/eight CPUs/16 GB/30 min/no pin. Verify identities
  remotely, freeze source, report job ID/receipt promptly, fetch full evidence
  and pause for parent review. No automatic follow-on retry or source changes.
- Prior failed receipts/packages stay intact. No installed-native replay by
  parent, acceptance, publication, next phase or new precision work authorized.
  This will be the second Phase 3 job, not a relabeling of the failed first job.

### Phase 3 — installed-native qualification passed; publication gate

- Retry `5216533` COMPLETED 0:0, node vep4, 73 s, eight CPUs / requested 16 GB,
  batch MaxRSS 532,796 K. Scheduler accounting is preserved beside the receipt.
  No additional job/retry occurred and the first failure remains intact.
- Parent read the saved phase report, raw outcome/provenance/demo records and
  scheduler accounting. Independently verified all **73 receipt-index entries**:
  `../recovery/verify/monitor-phase3-local-bw-QNKBhL/receipt-5216533/`;
  index SHA-256 `1b6533d34293e53a9833d02732e4a48b05477a1f370c4c380383a1f4a089b729`.
  The index intentionally excludes itself and the final exit.txt; parent also
  separately read exit.txt (exit=0). Index entries are not total fetched files.
- All four frozen inputs verify. Extracted source manifest SHA-256
  `00e86d2a9bad24f23c0acf559e774d5a182caea9271a8a13e7b76d3a5d1acb72`
  has 654 entries with the source/AppleDouble distinction retained.
- Actual retry wheel SHA-256:
  `8fe0e0ca550b8bf3fe8f18e044ecea337cfa26e5fae361e4652c1da87c841766`.
  Parent extracted native member SHA-256
  `a549b8d578329fd26f43e5ab57a541d379569a3c8ce1abbd934109678c190ffd`,
  matching the installed fresh-venv import record. All nine wheel test modules
  and changed Python production files match the checkout; all archive source
  payloads and the tested candidate.patch also match the checkout exactly.
  The child's initial report accidentally reused the first attempt's wheel hash;
  report-only correction requested. Raw artifacts are correct and unchanged.
- Parent independently parsed all nine JUnit files and checked selector/status/
  installed-origin records: **546 passed, zero failures/errors/skips**. Counts:
  reference 53, BW stream 79, local BW 84, complete windows 86, global co-clock
  64, native monitors 22, public BW 57, public windows 42, Recorder windows 59.
  No solver, test or oracle execution was repeated by the parent.
- Exact array parity tests now pass for matching-drive global/local output,
  per-region controls, distinct native lane controls/permutation at widths
  1/4/8/16, and sibling coexistence. Public broadcast lanes, independent clocks,
  chunking/tails, fresh reset, both partial-window continuation, strict/atomic
  restore and failure guards are qualified by the 84-case local suite.
- Existing demo ran after all nine suites and passed: forty 1 s BOLD rows over
  40 s, midpoint grid, incomplete-tail checks and exactly zero zero-drive BOLD.
  Experimental/geodesic warnings and the existing pytest deprecation remain
  visible; no suppression or gate relaxation occurred.
- **Native qualification accepted for publication**, not final Phase 3/workflow
  advancement. Parent owns the reviewed functional commit/push and exact-SHA
  hosted CI review; child remains held before returning accepted. A fresh remote
  query still shows tvbkh at ef1d60b8d; no competing upstream change detected.
- Keep native-only distinct-lane proof scoped honestly until public Phase 5
  sweeps. Strided native input hazard and public multimode fingerprint issue
  remain explicitly deferred. Global multi-source BW/sweeps remain guarded.
- Stage only the nine reviewed functional files plus this plan; preserve both
  untracked handoffs and all private metadata-bearing receipts outside Git.
  No merge/release or Phase 4 work is authorized before exact-SHA CI acceptance.
