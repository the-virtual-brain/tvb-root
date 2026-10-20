# AGENTS.md

Guidance for AI coding agents working on tvb_library package.

## Package overview

`tvb_library` is the standalone scientific core of The Virtual Brain (TVB). It provides brain simulation, scientific datatypes, models, coupling, integrators, monitors, analyzers, and related utilities.

Main packages:

* `tvb.basic` – shared infrastructure used across TVB, including traits, logging, configuration, profiles, and common utilities. Changes here can have broad impact across the codebase.
* `tvb.datatypes` – scientific data structures such as connectivity, surfaces, sensors, time series, and other objects exchanged between simulation, analysis, and visualization components.
* `tvb.simulator` – the core brain simulation functionality, including neural mass models, coupling functions, integrators, monitors, stimuli, connectivity handling, and hybrid simulations.
* `tvb.analyzers` – scientific analysis algorithms applied to TVB data and simulation results.

Avoid modifying `tvb.basic` unless necessary, as changes there can affect most of TVB.

## Environment and commands

Use a dedicated Python environment.

Install for development:

```bash
cd tvb_library
pip install -e ".[test]"
```

Run tests:

```bash
python -m pytest tvb/tests/library
```

Run a focused test when possible:

```bash
python -m pytest path/to/test_file.py
```

Build the package:

```bash
python -m build
```

## Code style

* Follow existing TVB conventions and surrounding code style.
* Prefer clear scientific naming over unnecessary abstraction.
* Keep public APIs backward compatible unless a breaking change is intentional.
* Add docstrings for public classes, methods, and scientific parameters.
* Do not perform unrelated refactoring while implementing a focused change.

## Development guidelines

* Understand the scientific meaning of code before modifying simulation logic.
* Preserve existing simulator behaviour unless the task explicitly changes it.
* Be careful with array shapes, state variables, nodes, modes, delays, and numerical precision.
* Consider both standard and hybrid simulation paths when changing shared simulator functionality.
* Avoid introducing unnecessary dependencies.

## Configuration and secrets

* Do not commit credentials, tokens, passwords, private URLs, or machine-specific configuration.
* Scientific defaults and reusable configuration may be committed when appropriate.
* Keep local environment settings outside the repository.

## Data

* Do not commit large generated datasets or simulation outputs.
* Use `tvb-data` or existing test fixtures when suitable.
* Keep test data small and deterministic.
* Do not modify reference scientific data unless required by the task.

## Testing expectations

* Add or update tests for behaviour that changes.
* Run focused tests during development and the relevant library test suite before considering the task complete.
* Scientific/numerical changes should include checks for expected shapes, values, and numerical consistency.
* Bug fixes should preferably include a regression test.
* For hybrid simulation changes, test both Python and Numba backends when applicable.

## Build and release

`tvb-library` is built with Hatchling and distributed as a Python package.

Before release-related changes:

* run the relevant test suite;
* build the package successfully;
* avoid manually changing generated artifacts;
* keep package metadata in `pyproject.toml` consistent.

## Git and collaboration

**Do not commit or push any changes.**

The agent may inspect files, modify the working tree, run tests, and prepare changes for review, but all commits and pushes must be performed manually by the developer after reviewing the changes.

Additionally:

* Do not create commits, amend commits, rebase, merge, or force-push.
* Do not rewrite Git history.
* Keep changes focused on the requested task.
* Inspect existing tests, issues, and surrounding implementation before changing established APIs.
* Clearly identify any API or behavioural changes for developer review.

## License

TVB is distributed under the GNU General Public License v3 or later.

Preserve existing copyright, license, and citation headers when modifying files.
