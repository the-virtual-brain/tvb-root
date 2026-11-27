# Public classic / C++ trajectory checks

`test_classic_cpp_trajectories.py` compares the public classic `Simulator` with
public hybrid `Simulator(backend="cpp")`, not the Python/Numba hybrid backend
or direct backend internals. No production changes are required for these
three fixtures. The hybrid directory is already included in C++ CI.

## Scientific and timing contract

- Synthetic three-node single-subnet networks; asymmetric weights are indexed
  `[target, source]` in both simulators. No transposition, row normalization,
  whole-network scaling, or fitted coupling compensation is applied.
- Same Heun deterministic scheme, `dt=0.0625 ms`, explicit initial state,
  weights, lengths, speed, and coupling parameters. Linear/Kuramoto run for
  8 ms (128 steps); Jansen–Rit for 16 ms (256 steps).
- Constant prehistory only: classic receives all `H` identical samples, while
  hybrid fills history from its one supplied state. This does **not** validate
  arbitrary time-varying prehistory, which the hybrid public API cannot accept.
- Classic naturally starts at `(H-1)*dt`; hybrid starts at zero. Do not reset
  `current_step`. Each absolute timestamp array is checked against its own
  independently specified sequence before comparing elapsed timestamps.
- The delayed JR network quantizes propagation times in units of dt:
  `[[0.2,1.2,3.8],[2.8,0.2,8.2],[5.2,6.8,0.2]]` to
  `[[0,1,4],[3,0,8],[5,7,0]]`. These avoid half-step ties. Both classic dense
  delay indices and hybrid CSR delay indices are asserted. Maximum delay 8
  yields a 0.5 ms classic clock offset, one full averaging period, aligning
  averaging windows without discarding the delayed startup transient.
- Classic Kuramoto's sparse coupling divides by the number of coupling
  variables (one here), not the three nodes. Both APIs use amplitude 0.375.
- JR uses the same classic pre-summation sigmoid of source `y1-y2`, not the
  legacy hybrid `e0/v0` convention. Source state indices `[1,2]` map to target
  coupling slot `[0]`. All six JR states are observed at every step.
- Monitors, in order: `TemporalAverage(period=0.5)` and
  `TemporalAverage(period=dt)`. The latter preserves each integrated state,
  timestamped at its bin midpoint rather than Raw's endpoint. JR's first
  monitor selects VOIs `[2,0]`. Shapes, finiteness, both absolute clocks,
  elapsed clocks, trajectory values, and window averages/VOI ordering against
  the per-step output are asserted independently.
- CPP float32 versus classic float64: trajectory tolerance `rtol=3e-5,
  atol=3e-6`; within-implementation window checks use `rtol=3e-6, atol=3e-7`.
  No output fitting, transient removal, or tolerance changes after observing
  the numerical comparison were used.

## Isolated cluster validation

Base production revision: `22f3a6817c08096c25777296709d2cacfdff3a97`.
All execution was submitted from the Slurm head through
`ssh -o BatchMode=yes inslurm-proxy`; all build/test jobs ran on c4.
The existing c5 editable environment was never imported.

Build job **5213835** succeeded (58 s): archived the base into
`/tmp/phase2-classic-5213835/src`, created a fresh `/usr/bin/python3.12` venv,
used separate node-local build/cache/tmp directories, and non-editably
installed with:

```sh
$HOME/.local/bin/uv venv --python /usr/bin/python3.12 "$ROOT/venv"
$HOME/.local/bin/uv pip install --python "$ROOT/venv/bin/python" \
  "$ROOT/src/tvb_library" pytest h5py -Cbuild-dir="$ROOT/build"
```

Python 3.12.3, GCC 13.3.0, pytest 9.2.0. Provenance checks confirmed both classic
Python and `_cpp_hybrid.abi3.so` load from
`/tmp/phase2-classic-5213835/venv/lib/python3.12/site-packages/tvb/`, with no
editable `.pth`. Tests were copied to fresh node-local run directories;
`PYTHONPATH`/`VIRTUAL_ENV` were unset and caches were node-local.

Test job **5213842**: **3 passed, 0 skipped**, 2.24 s.
Repeat checked job **5213844**: **3 passed, 0 skipped**, 2.42 s; an external
pytest plugin printed errors without modifying assertions or production.
The checked module SHA256 is
`a6a10d4e189cd1ccc12bd251002af173fa1c420099b9415ef9aa82d9115f6108`.
Maximum absolute classic/CPP trajectory errors:

| Model | 0.5 ms average | Every-step state |
|---|---:|---:|
| Linear | 4.42736860506e-08 | 6.02180458831e-08 |
| Kuramoto | 1.10418770660e-06 | 1.25728549616e-06 |
| Delayed Jansen–Rit | 2.79047224172e-06 | 7.26450934607e-06 |

All timestamp comparisons were exact. Warnings were the expected unavailable
optional geodesic-distance module and experimental hybrid notice; classic
also logs its non-stochastic random-state warning. Full-suite validation and
other platforms are not claimed.

Persistent evidence on the head: `~/phase2-classic-evidence/` contains
`build-5213835.log`, `tests-5213842.{log,xml}`,
`checked-5213844.{log,xml}`, and reproducible `run-checked.sh`.
The checked test command was:

```sh
/tmp/phase2-classic-5213835/venv/bin/python -m pytest \
  -p metrics -s -vv -ra test_classic_cpp_trajectories.py \
  --junitxml=$HOME/phase2-classic-evidence/checked-5213844.xml
```

## Earlier failures and separate Raw compatibility gap

- Probe job 5213833 succeeded, verifying c4's Python 3.12.3.
- Build job 5213834 failed before building: `uv` was absent from batch PATH.
  Work stopped and parent approved retry with the verified absolute uv path.
- Test job 5213838 failed before CPP execution because the initial test
  chained hybrid `configure()` (which returns None). The test was corrected.
- Test job **5213840** reached the public CPP simulator and all three initial
  Raw fixtures raised `ValueError: Raw monitor requires chunk_size=1; pass
  chunk_size=1 to run_network()`. At the base revision,
  `tvb/simulator/backend/cpp_hybrid/backend.py:1602-1606` rejects Raw at chunk
  sizes other than one. The public call in
  `tvb/simulator/hybrid/simulator.py:336-342` does not supply chunk_size and
  reaches the backend default eight. This is a separate reachable public-API
  compatibility gap, not evidence of numerical mismatch. Parent approved
  replacing Raw with the per-step TemporalAverage above and owns Raw triage.
  These tests do **not** claim Raw coverage. Failure log/JUnit remain as
  `tests-5213840.{log,xml}`. No production workaround was introduced.
