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

## Current iteration policy — operator update (supersedes historical CI holds)

The operator explicitly stopped per-increment CI blocking. Iterate using relevant
`tvb_library` tests in a suitable dedicated local environment or a bounded Slurm
job. Passing local/installed-native tests plus supervisor diff/evidence review
are sufficient to accept a phase and start the next; pending hosted CI is NOT a
sequencing gate. Keep CI enabled in the background, inspect it at meaningful
phase/delivery checkpoints, and require final-plan qualification without turning
that into repeated `gh --watch` waits during development. Investigate actual CI
failures when reported; do not confuse pending checks with failures or claim they
passed. Earlier progress entries and child prompts requiring all CI green before
advancing are historical and superseded by this policy.

- Use focused affected library tests for the inner loop, then relevant broader
  library/backend regressions at phase boundaries. Reuse completed evidence for
  unchanged inputs; no redundant jobs just to occupy a CI wait.
- Prefer the already working Slurm native setup when no suitable local installed
  extension exists; local execution is now explicitly allowed by the operator.
  Keep environment/source provenance, scientific tests and numerical tolerances.
- Preserve one active implementation child per phase, supervisor review/publication
  ownership, and concrete Slurm resource approval. No CI watcher tasks/loops.
- Operator-approved timeout fallback: if Blablador times out, use Terra 6.1,
  exact registry model `openai/gpt-6.1-terra`. This now authorizes replacing the
  failed, zero-tool Phase 4 worker. If Terra stalls or cannot deliver, the operator
  also authorizes restarting its native worker as `openai/gpt-6.1-sol`. Stop the
  old writer first and preserve its candidate/evidence. No external CLI or other
  model fallback. Use Terra while it is responsive, not repeated provider probes.
- Phase 3 is accepted from reviewed published code, job 5216533's 546 passing
  installed-native cases and successful demo. Phase 4 is now accepted from
  independently verified job5216536 (568 passes + demo); parent publication is
  in progress. Phase5 starts only after that publication and Phase4 closeout.

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
| 3 | Public local BW Recorders, including independent SIMD lanes | Complete; published bae00b766; accepted under updated iteration policy | Job 5216533: 546 passes + demo; CI no longer blocks advancement |
| 4 | Compatible multi-subnetwork global BW and shared placement semantics | Complete; independently accepted; parent publication in progress | 5216536:568 clean passes +40s demo; two failed fixtures preserved |
| 5 | Explicit BW parameter sweeps through backend and notebook APIs | Authorized after Phase4 publication/closeout; startup contract next | No CI wait; same one-worker/local-or-Slurm test gates |
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

Deliver a user-friendly executable notebook in **tvb_documentation/demos/**,
with its paired script and clear setup, configuration, axis/time/units explanations
and meaningful plots. The operator explicitly requires a combined demonstration
of **mixed timescales, genuine SIMD lanes and BW**, not just tests or broadcast
lanes. Add or extend focused paired notebooks to show:

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
  reviewed, locally qualified increments only to `tvbkh`. Passing relevant local
  or Slurm `tvb_library` tests and parent review permit phase advancement without
  waiting on hosted CI. Inspect CI at delivery checkpoints and address actual
  failures; no force, merge, remote rewrite or auto-merge.
- Any approved push uses explicit `git@github.com:the-virtual-brain/tvb-root.git`
  and `HEAD:refs/heads/tvbkh`, with fresh ref/ancestry/diff and dry-run checks.
- Do not stage local handoffs, raw receipts, machine settings, credentials or
  generated benchmark outputs. Keep operational evidence outside the source tree.
- Run library tests locally in a suitable dedicated environment or on Slurm via
  `inslurm-proxy`; the operator now explicitly permits both. For Slurm use private
  exact-source inputs, node-local builds and fresh installed environments with
  verified imports. One explicit submission at a time; supervisor approves a
  concrete job request. Do not rebuild locally merely to duplicate passed native
  evidence when the existing Slurm setup is faster.
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

### Phase 3 — reviewed increment published; exact-SHA CI underway

- Parent committed exactly the nine reviewed functional files and PLAN.md as
  `bae00b76645648d82753a2173dd3f37fc515f020`, parent ef1d60b8d:
  `Support local Balloon-Windkessel Recorders on the C++ hybrid backend`.
  Both untracked handoffs remain untracked; no private evidence was staged.
- Fresh remote check showed ef1d60b8d, ancestry/parent checks passed, tracked
  worktree/index were clean, explicit push dry-run passed, then parent pushed
  `HEAD:refs/heads/tvbkh` to the authoritative GitHub URL without force. Live ref
  independently confirms bae00b766; origin/remotes were not changed.
- Parent discovered seven in-progress workflows at the exact published SHA:
  cpp-hybrid 45883418359; Test Py 45883418686; Test Notebooks 45883418412;
  Test PG 45883418372; Test Win 45883418451; Check tvb-contrib 45883418497;
  Wheels 45883426216. PR #801 is still open/draft and points to this SHA.
- Same sole Phase 3 child is authorized to watch these checks read-only, record
  matrix/explicit local-BW gate/demo outcomes and the two applicable Sonar checks,
  and pause again. No CI rerun, source edit or acceptance return preapproved.
- Exact-SHA CI remains the final acceptance gate. Phase 4 stays blocked until
  parent independently verifies all applicable outcomes and explicitly advances.
  Operational log edits after publication remain local for the next functional
  increment, avoiding an unnecessary documentation-only CI restart.
- Watch attention at about four minutes was inspected, not treated as failure:
  active command is `gh pr checks 801 --watch --fail-fast`, with trailing output
  buffered by tail. Parent independently queried exact SHA bae00b766: Wheels
  and Check tvb-contrib have succeeded, including Linux/macOS/Windows wheel
  jobs; release-upload is deliberately skipped. Five other workflows remain in
  progress, with no failure observed. PR remains open/draft at the same SHA.
  No supervisor request is pending; no interrupt, rerun, replacement worker or
  Phase 4 advancement is justified. Revisit on native child completion/request.
- Subsequent attention also concerns the expected watch. Its prior 1500 s tool
  timeout did not cancel/rerun CI; the child inspected statuses and continued
  read-only watching. Parent independently confirms cpp-hybrid 45883418359 has
  now completed successfully at bae00b766, including all wheel/sdist jobs,
  reference/backend/buildgen/battery checks, **524 passes** in the dedicated
  eight-suite BW/window step, and the 40-row public BW demo. Wheels and contrib
  remain successful; Test Py, Test PG, Test Win and Test Notebooks still run.
  No check failure or pending supervisor decision was found; continue the
  existing watch without interrupt/retry or Phase 4 advancement.
- Later watch attention: parent independently confirms Test PG, Test Win and
  now Test Notebooks succeeded at bae00b766 (six of seven workflows successful).
  Test Py's 3.13 job also succeeded; 3.11 and 3.12 are still running framework
  tests. No failures are posted; Sonar check runs have not appeared yet and are
  not assumed satisfied. The expected gh watch remains active, not stuck or
  waiting for a supervisor decision. Keep waiting for remaining exact-SHA gates,
  with no intervention, duplicate watch, CI rerun or premature advancement.

### Operator correction — stop CI blocking; Phase 3 accepted, Phase 4 authorized

- Operator explicitly requires fast iteration on local/Slurm `tvb_library` test
  results rather than repeated hosted-CI watches. This supersedes the preceding
  historical Phase 3 holds and CI-before-advancement instructions.
- Parent queued the new policy to the existing Phase 3 child and terminated only
  its verified local gh-watch process (PID 45567). Hosted CI jobs were NOT
  cancelled, and no test/job was rerun. Do not restart a watch.
- **Phase 3 accepted now**, from the independently reviewed source/diff, published
  bae00b766, all 546 installed-native tests passing in job 5216533, and the passed
  demo. Existing dedicated hosted gate also passed 524 tests; no new validation
  is required just to close this phase. Remaining hosted results are not claimed
  green and no longer block Phase 4.
- Same Phase 3 child should finalize its report with this acceptance basis and
  return accepted. Existing sequential workflow may launch its sole Phase 4
  child from bae00b766. No extra child, fresh Phase 3 run or CI watcher.
- Phase 4 follows its existing scientific scope: compatible global regional
  gather on master ticks with held slower sources, complete/disjoint mappings,
  independent regions/lanes, continuation and public coexistence. Agree a concise
  tests/clock/gather contract, then implement/test; do not repeat broad audits or
  solver/performance investigations. Use local/Slurm library suites as the work
  gate; inspect CI at milestones without blocking implementation on pending jobs.
- PLAN.md operational changes remain local until the next functional increment;
  no documentation-only publication or CI restart is needed for this policy.

### Workflow timeout recovery — accepted work preserved; continue at Phase 4

- Original workflow `f65dea8b-85fd-424d-a5a1-5d143b64ec48` is now terminal
  failed. Phase 3 child `cdca7fa2-7605-4ab2-b004-042160ea351b` timed out and
  did not emit its required structured closeout. Preserve that operational
  failure and missing structured result honestly; do not fabricate a child
  acceptance object or overwrite the failed receipt.
- Parent inspected the workflow receipt/status, saved partial output and durable
  Phase 3 evidence report. The report is intact, with corrected wheel hash and
  all 546 native passes/demo. Product Phase 3 remains **accepted by parent** as
  recorded above; worker finalization failure is not a failed scientific test.
- Checkout remains bae00b766 with only parent PLAN.md changes and the two
  untracked handoffs; no gh-watch process remains. No source loss, active writer
  or partial Phase 4 implementation was found.
- Do not revive a long-running finished phase merely to manufacture its missing
  JSON closeout, and do not rerun Phases 1–3, tests or CI. Continue the already
  authorized remaining Phases 4–6 in a fresh native sequential workflow, one
  worker per phase, same explicit worker model, no provider/CLI/model fallback.
  Parent remains the sole reviewer and publication owner at phase gates.
- New phase tasks must read the current top-of-plan iteration policy. Focused
  local/Slurm library tests gate iteration and acceptance; pending hosted CI
  never blocks the next implementation phase. Report actual failures promptly.
- Native follow-up workflow launched asynchronously:
  `b4c301f4-2e42-4ba2-8165-e6dc6ca50bc2`, attached to the existing mission
  `3e979de0-5354-4b3c-9820-f2d3df354bb6`. Only stable keys cpp-phase-4, -5,
  -6 are scheduled, serially, same worker/model. Each must obtain parent review
  and explicit acceptance before the next child starts; no CI-watching child.
- Outputs are bound to managed `cpp-delivery/phase-4.md` through phase-6.md
  under the new workflow artifact root. Child IDs arrive with startup updates.
  Old reports/receipts remain provenance; their failed run state is not rewritten.
- New child contracts explicitly forbid rerunning finished phases, publication,
  PLAN edits, model fallback and CI waits. They carry the current local/Slurm
  testing policy and a checkpoint before their run deadline.

### Phase 4 startup provider failure — stopped for routing decision

- Follow-up workflow `b4c301f4-2e42-4ba2-8165-e6dc6ca50bc2` failed in its first
  child, `e99bd0bd-078d-457a-a5bc-75c570c58372` (cpp-phase-4), with
  `Request timed out` and no required structured output. Phases 5–6 never ran.
- Parent read failed-child status/transcript/output and event records: **zero
  tool-execution events**, six failed model turns with built-in provider retry
  events. The configured blablador/deepseek-v4.1-flash-at-hzdr worker did not
  reach its initial reads or startup contract. This is a provider/runtime
  failure, not a native test failure or incomplete source implementation.
- Parent verified HEAD remains bae00b766 and only its PLAN.md updates are
  tracked changes; both handoffs remain untracked. No active gh-watch, source
  write, test execution, Slurm submission or partial Phase 4 change was found.
- Preserve failed native records; do not launch repeated automatic replacements
  against the same timed-out provider or silently change models/CLI/ownership.
  Phase 3 stays accepted. Phase 4 is authorized but technically blocked pending
  operator decision: approve a replacement worker on another available model,
  or explicitly retry the existing provider. No fallback had been selected at
  that checkpoint; the operator's authorization below resolves the decision.

### Operator-approved timeout fallback — Terra 6.1

- Operator: "if blablador provider times out, use terra 6.1". The condition is
  met by the documented Phase 3 closeout and fresh Phase 4 startup timeouts.
- Parent queried the native model registry and resolved the exact permitted
  model as **openai/gpt-6.1-terra**. Continue with the native worker protocol,
  not an external CLI, and preserve one active implementation child per phase.
- Replace the failed Phase 4 startup (zero tools/edits) with a fresh Terra worker;
  continue remaining phases serially on this authorized fallback rather than
  repeatedly probing the failed provider. No rerun of Phases 1–3 or CI watching.
- Keep the same scope, local/Slurm validation and parent checkpoints/publication
  ownership. This changes routing, not scientific equations, acceptance evidence
  or the tests.
- Native async workflow launched: `783dd1e9-dcce-4a1f-a4df-a9611627351f`, same
  mission `3e979de0-5354-4b3c-9820-f2d3df354bb6`. Phases 4–6 are serial,
  worker model `openai/gpt-6.1-terra:high`, one active worker, parent acceptance
  before each next phase. Output reports are managed cpp-delivery/phase-4.md
  through phase-6.md under this workflow's artifact directory. The two failed
  workflows and their records remain preserved; no completed tests were rerun.

### Phase 4 — startup contract approved; implementation authorized

- Active sole worker `abd84257-e724-4231-8afa-64ca989e143e`, Terra 6.1,
  workflow 783dd1e9…, starts from bae00b766. Parent confirms no source edits
  yet; only PLAN changes and preserved handoffs. Startup contract is in scope.
- Approve serial tests/native/public changes in backend.py, _core.cpp and
  cph_monitors.hpp, with focused tests and supported-profile documentation.
  Preserve the solver and accepted single-source ordering/API. Multi-source
  mappings must be explicit, complete/disjoint canonical 0..N-1; compatibility
  means ordered observation definitions, not merely matching counts.
- Use established gathered-global return nesting with one gathered stream per
  global monitor, shape (n,1,N,W), float64 times / float32 samples. Local Recorder
  output stays source-local. Do not silently canonicalize existing single-source
  BW ordering or loosen average-monitor width/merge rules.
- Global BW uses the master clock, which must divide 10 ms under the existing
  binary32 clock rule. A slower source needs valid integral cadence, not its own
  10 ms divisor unless a local BW is also attached there. Include cadence-3
  (2.5 ms master / 7.5 ms source) initial/held-drive and window-crossing tests;
  assert actual compiled clocks instead of trusting fixture labels.
- Extend wide state/reservation and snapshot identity over every source, mapping,
  clock/cadence, ordered VOI, width, calibration and hemodynamic configuration.
  Stage all regions/lanes before close commits; reject late corrupt state or
  sibling without partial monitor-set commits. Preserve both partial windows;
  no neural/whole-simulator rollback claim or cross-binary snapshot guarantee.
- Qualify matching-drive global/local parity, mixed-rate held-drive reference,
  source/node/lane permutations, supported widths, chunk/reset/continuation and
  siblings. Update old unsupported-route assertions only for genuinely supported
  profiles, retaining invalid mappings/modes/native-disable/classic/Numba/sweep
  guards. Distinct native-lane plus public broadcast proof remains acceptable
  until Phase 5 public sweeps; use contiguous float32 native inputs.
- Proposed ten-suite gate (new Phase 4 suite plus prior nine) and unchanged demo
  approved in principle: run/one task/eight CPUs/16 GB/30 min/no pin. Local
  library test iterations are allowed in a suitable existing environment.
  Present exact frozen inputs/runner readiness before one Slurm submission;
  no job or retry approved by this contract decision alone.
- Native/local test evidence + parent review gate this phase. No hosted-CI watch,
  prior-phase rerun, numerical pilot or future-phase API expansion is authorized.

### Phase 4 — first frozen readiness review; corrections before submission

- Worker paused with no installed-native tests or submitted jobs. Static compile,
  diff whitespace and runner syntax checks passed. Parent reviewed all native /
  Python changes, new tests and runner. Frozen readiness package preserved at
  ../recovery/verify/monitor-phase4-global-bw-9cmVDO (never submitted).
- Parent verified four indexed input hashes and index digest
  6f80397b7e8843c279ea69595e67caab5f98950b8e6f7d6ce3f0c1a6cea62cd8.
  Archive 7268e796f5b76666bf7e3e5adbc8443f6bfccfe2cc8b2c8e379e846b75cf8343
  is 5,017,600 bytes, 328 regular allowlisted source files, no metadata/cache /
  build/link/path hazards; every payload equals checkout, no tracked source
  omitted. Patch 45acc221e5182744cb14e086f05c8233cd2161bc8d5497b0ffe74187624518f0
  equals functional checkout diff including the new untracked suite; demo bytes
  match. Runner 8fcb31d2e3b8efbf97677b1fac8c7682102bb8770920bc622539792c458c521f
  correctly uses installed origins and dynamic per-suite outcomes/aggregate.
- Do not submit this package. Strengthen the mixed-clock reference assertion:
  <50 ms plus atol=2e-7 has not established sensitivity to missing BW response or
  wrong hold/update semantics. Use a bounded resolved response and assert that
  zero/pre-update/always-update controls fail the existing suitable reference
  tolerance; no solver/tolerance relaxation or numerical investigation.
- Native distinct-lane proof currently uses one source and identical region
  increments, while multi-source proof only broadcasts lanes. Add combined
  multi-source/mixed-clock asymmetric source/region/lane controls vs scalar
  runs, plus wide late-lane close atomicity and mixed global/local continuation.
  Exercise new native multi-source identity mismatches, not just public metadata.
- Global BW reserve now includes W but still reserves only new rows, ignoring
  retained pending output. Fix this touched branch with checked old-size +
  additional rows/elements and a continued-call regression; no unrelated reserve
  refactor. Update stale public BW supported-profile documentation.
- Operator reiterates Phase 6 user-friendly mixed-timescale/SIMD/BW notebook and
  permits Sol 6.1 restart if Terra fails delivery. Terra is responsive at its
  first review; corrections are normal review, not a provider failure. No model
  restart yet. Preserve one active worker. A corrected frozen readiness review
  and concrete parent job approval are still required.

### Phase 4 — corrected candidate reviewed; one job approved

- Parent reviewed corrected tests/docs/reservation branch and independently
  verified ../recovery/verify/monitor-phase4-global-bw-6AbUOh. Four indexed
  hashes pass; index d1e5d191ccc0d05d5777305514304fbb207765909e25cf788fba86de404999ac.
- Archive bc6ae39b64d792642d0ef768e1c5d590a886187c690b5c0ab59aa424050e0411,
  5,027,840 bytes, exactly 328 regular source payloads; every payload equals
  checkout and no tracked source is omitted. No links, traversal, AppleDouble,
  cache/build products. Functional patch including new suite equals checkout:
  4eab07a61ae819cc700a456f35f4fc6738210758807fda6f870f7398c7678ba9.
  Demo unchanged. Runner byte-identical to reviewed predecessor, SHA
  8fcb31d2e3b8efbf97677b1fac8c7682102bb8770920bc622539792c458c521f;
  bash syntax and git diff whitespace checks pass.
- Changes from first readiness limited to BW header reservation, monitor docstring,
  README and new global test module. Global reservation now checks remaining
  limits and adds current pending-output sizes. New reference uses 203 ticks,
  strict existing 1e-7/1e-10 envelope and zero/pre-update/always-update controls.
  Combined native mixed-clock source/region/lane controls, permutations, wide
  domain-close atomicity, native identity/sibling mismatches, global/local/average
  continuation and pending-output rejected-restore checks are present.
- Approve EXACTLY ONE submission of these immutable inputs: run partition,
  one task, eight CPUs, 16 GB, 30 min, no node pin; transfer/hash verification,
  allocation-only build/install/test/demo, status and receipt fetch for that job.
  Ten independently gated installed suites; counts derived from actual JUnit.
  Stop/preserve/report first failure; no retry, selector/tolerance change,
  source edit, extra allocation, login-node scientific work or CI watching.
- This is submission readiness, not Phase 4 acceptance. No installed-native test
  has run yet. Caveat: new pending-output test checks rejected-restore retention;
  it does not directly exercise appending another output with an undrained buffer
  or expose vector capacity. Do not overstate that proof; reservation arithmetic
  is independently code-reviewed. Raw outcomes and parent final review remain
  required before acceptance/publication/Phase 5.

### Phase 4 — job 5216534 first failure preserved; bounded test correction

- FAILED 1:0, 52 s, vep4, run partition/eight CPUs/requested 16 GB;
  batch MaxRSS 975484K. Fresh build/install and installed provenance succeeded.
  Parent read scheduler/exit/raw failure, independently verified all four suite
  selectors/statuses/origins/JUnit cases: reference 53, BW stream 78, local BW84
  pass; global BW20 pass/one fail. Total235 passes/one failure, zero skips/errors.
  Six later selectors/demo/aggregate/DONE did not run; no full gate claimed.
- Only failure: test_native_global_wide_last_lane_close_is_atomic expected a
  RuntimeError from a huge positive observation with positive drive_scale=0.25.
  No exception occurred. Unlike the accepted local fixture's negative scale,
  this does not establish a negative-inflow intermediate or invalid domain.
  Do not change production to reject arbitrary large finite positive drives.
- Parent verified all328 pre-build source-manifest payload hashes against the
  frozen archive and patch/index copies. Wheel SHA
  466801356e727bb1972a33823b0bd19c3a28605c20fa25c3c3ade2aa829003da;
  native wheel member equals asserted installed hash
  9e2c435db32e0a9e49ec4d169d324f4a88bb93277cf4d58ba79f90aaec413803.
  Packaged source members checked against archive where present.
- Raw receipt-5216534 and original inputs remain unchanged. Runner writes its
  receipt index only on success, so this failed receipt has NO job-generated
  receipt index. Parent independently indexed41 fetched files in the external
  parent-fetched-receipt-5216534.sha256 (not inside the raw receipt), SHA
  7678a798b1d324168ee6310648fdd6893d9a87f07e1e2486362b47ae2e99f6db.
  Do not mislabel that local fetch manifest as a job-generated receipt.
- Authorize test-only correction in the global suite: deliberately negative
  finite calibrated drive in the final canonical region/lane, after a successful
  output and with the next drive window open; require the real RK4 domain error,
  unchanged monitor set/tick and preserved pending output vs untouched control.
  Negative drive at this magnitude makes the first RK4 inflow stage nonpositive.
- Also close the existing pending-append coverage caveat by extending the
  rejected-restore/pending-output case across another output close WITHOUT first
  draining its pending row, and compare both rows with untouched control.
  Preserve production, solver, tolerances, remaining tests and runner byte-for-
  byte. New exact-source package/readiness required before any retry approval.
- Terra is responsive and stopped correctly on first failure. This is one
  bounded fixture correction, not a provider timeout or grounds to discard the
  productive candidate. Sol6.1 fallback remains available if delivery stalls.
  No retry job, source/selector relaxation, CI watch or Phase5 advancement yet.

### Phase 4 — lifecycle clarification; preserve per-call output semantics

- Worker correctly identified an error in the parent's correction direction:
  simw::run calls monitor_begin_call before reservation/integration, and BW
  begin_call swaps out t_out/d_out. Native output buffers are PER CALL, not
  undrained streams retained across calls. Parent read and confirmed both paths.
- WITHDRAW the requested cross-call two-row assertion. Do not change begin_call,
  output lifecycle or public API. Earlier parent language calling additive reserve
  a demonstrated pending-output bug was incorrect: old sizes are zero on this
  normal Sim.run path. The submitted additive formula is harmless defensive
  arithmetic, not proof of a reachable data-retention fix. Do not overclaim it.
- Authorize a test-only, single-call atomicity proof: identical fresh candidate
  and untouched prefix-control, same 12-tick inputs, final canonical region/lane
  receives large negative increments only at ticks9–12. Control runs11 ticks;
  candidate runs12 in ONE CALL and fails on its last close. Compare candidate
  monitor snapshots/tick to prefix-control after11; both should retain the
  earlier tick8 output, produced within the same call. Compare nonempty buffered
  outputs byte-for-byte. No neural rollback or partial-state retry claim.
- Keep rejected-restore pending-output test unchanged: restore itself does not
  start another run and its current proof is valid. Only the domain-fixture
  function needs correction. Preserve production/runner/tolerances and failed
  evidence; new frozen identities still required before retry submission.
- Terra's prompt detection of this contract conflict is appropriate supervision,
  not stalled delivery or a reason to switch models.

### Phase 4 — corrected fixture frozen; one concrete retry approved

- Parent reviewed exact failed-vs-retry diff: ONLY the failed domain atomicity
  function changed in test_cpp_hybrid_global_balloon_windkessel.py. It now uses
  negative finite drive at the final canonical region/lane in one12-tick call,
  compares the entire monitor set/tick to an untouched11-tick prefix, and checks
  their nonempty earlier tick8 outputs. Existing rejected-restore case unchanged.
- Retry package ../recovery/verify/monitor-phase4-global-bw-tWquiK:
  archive e284df08b0381173aaa72eb6005e27dab916a8ae397e73e709e7b1307daee75a,
  5,027,840 bytes; patch42ad8af7a75d7b727a6fa376138978a08289189f7fe8c4acdc43cf12cddd70f3;
  index c6dbf4f22c49aa72e924fef81905f82092389ab1c2a7afafe7527f503bca896e.
  Parent verified all input hashes, archive safety/328 payloads/exact checkout,
  functional patch equality and that only this test payload differs from6AbUOh.
  Every production source byte, runner and demo match first failed submission.
  Reviewed runner SHA remains8fcb31d2e3b8efbf97677b1fac8c7682102bb8770920bc622539792c458c521f.
  Whitespace/runner syntax checks pass.
- Approve EXACTLY ONE retry with these frozen inputs and unchanged ten-suite+
  demo outcome gates/resources: run partition/one task/eight CPUs/16GB/30min,
  no pin. Fresh installed-native envs/provenance; hash-verified transfer and
  receipt/accounting fetch. No source edits, selector/tolerance relaxation,
  automatic extra retry, CI watching or future-phase work. Stop/report failure.
  This is job authorization, not Phase4 acceptance; raw results remain pending.

### Phase 4 — job 5216535: scientific/global suites pass; public guard correction

- FAILED1:0 at selector8,71s,vep4,run/eight CPUs/requested16GB,
  batch MaxRSS961700K. No edits/retry. Parent inspected raw failure/accounting,
  exact source manifest328, all eight selectors/statuses/origins/JUnit records.
  Seven suites clean: reference53,stream78,localBW84,globalBW21,complete86,
  co-clock64,native monitors22. PublicBW56 pass/one fail. Total464 passes/one
  failure, no skips/errors. Last two suites and demo/DONE absent.
- The corrected wide atomicity case and all21 new global tests passed. Only
  failure is a negative public geometry fixture: both sources explicitly map
  onto the SAME permutation(2,0,1), but its name/regex call them unmapped.
  Actual precise rejection is complete/disjoint node_indices, not the missing-
  declaration message. Production rejects the invalid geometry correctly.
- Parent verified frozen manifest/patch/index and wheel source members; wheel
  SHA57f774366e5441aa7b3a35ec90d6469ae4b2ca5a2373f242c133a45ad39b11ce.
  Native member AND asserted installed hash9e2c435db32e0a9e49ec4d169d324f4a88bb93277cf4d58ba79f90aaec413803,
  byte-identical native module to first job. First failed receipt untouched.
- This failed runner again produced no job receipt index. Parent-generated
  external local fetch manifest parent-fetched-receipt-5216535.sha256 covers65
  files, SHA f2177e34282e25d462d8cfbe00b2ede28afb627ac5b9b196d5e7e662131e1595.
- Authorize bounded TEST-ONLY correction in public BW module: separate absent-
  mapping and overlapping-mapping cases with actual explicit fixtures and precise
  corresponding messages; isolate multimode rejection so an earlier mapping
  assertion cannot mask it. Use _network(indices=None) for missing declarations,
  default(2,0,1) on both for overlap. No broad alternative regex, weakened
  geometry/numerical assertions, production/runner/solver changes or CI watching.
- New frozen test-only diff/input review required for one further job approval.
  No job authorized by this correction decision alone. This is meaningful native
  progress, not a provider failure; continue sole Terra worker, keep Sol fallback
  available if delivery stalls. Phase4 not yet accepted, Phase5 not started.

### Phase 4 — public guard correction reviewed; one further job approved

- Frozen ../recovery/verify/monitor-phase4-global-bw-PJ8Rn0 independently
  verified: all indexed hashes pass; index
  452500bebd0a94c5d1d48c6ee759b5ad153eb99a69efde6dd6f3c8892a223cca.
  Archive fa770cfea2520bd60e9e3db583f3b5ebb013632db21c8843e6ad158f5949a4cf
  is5,027,840 bytes,328 safe regular payloads exactly equal checkout with no
  tracked-source omissions. Functional patch equals checkout:
  ac07dc967b77ee96200c5e0bac7498334f2590132d4e67bfe343015fc1e8d405.
- Only the public test module differs from tWquiK: absent and overlapping
  mappings are distinct parameterized cases with precise errors; multimode is
  independent. No weakened assertions. All production, global suite, runner and
  demo byte-identical. Runner remains8fcb31d2e3b8efbf97677b1fac8c7682102bb8770920bc622539792c458c521f.
  Static whitespace/runner syntax checks pass.
- Approve EXACTLY ONE submission of PJ8Rn0 with unchanged10-selector+demo gate,
  run/one task/eight CPUs/16GB/30min/no pin. Verify transfer and fresh installed
  provenance, derive actual counts, fetch raw receipts/accounting. Preserve both
  prior failed inputs/receipts/local fetch indices. No source edits, automatic
  retry, CI watch, selector/tolerance relaxation or Phase5 until parent review.
  This is submission approval only; full native outcomes remain pending.

### Phase 4 accepted — job 5216536 independently verified

- COMPLETED0:0,81s,vep4,run/eight CPUs/requested16GB; batch MaxRSS836976K.
  Parent verified all79 job-indexed payloads, index
  b60bd43512660e836a16453b15a291d727f7f0097c08b00ef0b96b9c91066bc4;
  81 fetched files including index/exit. Exit0/DONE and exact328 source-manifest
  payloads, frozen patch/index and checkout all match before final prose edit.
- Ten independently checked installed selector/origin/status/JUnit gates:
  reference53,stream78,localBW84,globalBW21,complete86,co-clock64,native22,
  publicBW59,public windows42,Recorder59 = **568 passed,0 failed/errors/skips**.
  Counts are actual completed cases, not collection or a guessed constant.
- Existing unchanged demo passes its40s/40row response, midpoint and dropped-tail
  checks and exact-zero zero-drive assertion. Meaningful response peak0.0079029.
  Wheel SHA8645717637cde76aa89734ce1b90f3ea2b8e40048eb46db6ef39c792cc2d49fe;
  native wheel member AND fresh installed extension SHA
  9e2c435db32e0a9e49ec4d169d324f4a88bb93277cf4d58ba79f90aaec413803.
  Native binary also matches both failed jobs; only tests changed on retries.
- Parent final diff/evidence review accepts compatible multi-source global BW
  and widths1/4/8/16, master-held slower observations including cadence3, canonical
  mappings, independent lanes, strict staged continuation, global/local/average
  coexistence and unchanged legacy guards. Solver/precision/HRF/sweeps unchanged.
  Atomicity claims are monitor-set restore/all-region-lane close, not neural
  rollback; native outputs remain per-call. Both failed jobs stay private/intact.
- ONE post-qualification README-only clarification scopes the old average-gather
  paragraph correctly; it previously contradicted the newly documented BW
  profile. Entire executable/test/native candidate remains byte-identical to
  the qualified archive. This prose edit was manually reviewed, not claimed to
  be part of the frozen wheel; no redundant scientific test job for prose.
- At this delivery checkpoint, single non-watching GitHub query confirms all
  Phase3 bae00b766 workflows and BOTH Sonar checks completed successfully;
  release upload intentionally skipped. Live tvbkh/PR801 head still bae00b766,
  PR open/draft. No hosted-CI waiting or requalification of accepted Phase3.
- Parent will stage only ten functional files plus PLAN, commit/push explicit
  GitHub tvbkh with fresh ancestry/ref/dry-run checks. Phase4 accepted on local
  native evidence; Phase5 authorized following publication/worker closeout.
  Phase6 retains the operator-required user-friendly mixed-timescale/genuine-
  SIMD/BW notebook in tvb_documentation/demos. No merge/release authority.
