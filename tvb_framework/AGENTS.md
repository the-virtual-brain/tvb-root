# AGENTS.md

Guidance for AI coding agents working on tvb_framework package.

## Package overview

`tvb_framework` contains the application and service layer of The Virtual Brain. It connects the scientific functionality from `tvb_library` with the TVB web application, persistence layer, forms, adapters, project operations, and user workflows.

Main areas:

* `tvb.adapters` – connects scientific algorithms and datatypes to application workflows. Includes simulator, analyzer, visualizer, uploader, and datatype adapters.
* `tvb.core` – shared framework services, entities, project operations, storage, configuration, and application infrastructure.
* `tvb.interfaces` – application interfaces such as the web layer, controllers, forms, and related user-facing workflows.
* `tvb.config` – framework-level configuration and initialization utilities.

Scientific simulation logic should normally remain in `tvb_library`; `tvb_framework` should expose and orchestrate that functionality for application use.

## Environment and commands

Use a dedicated Python environment and install the relevant TVB packages in editable mode.

Example:

```bash
pip install -e ../tvb_library
pip install -e ".[test]"
```

Run focused tests when possible:

```bash
python -m pytest path/to/test_file.py
```

Run the relevant framework test suite before considering a task complete.

## Code style

* Follow existing TVB conventions and nearby implementation patterns.
* Prefer small, focused changes.
* Keep controllers thin; place reusable logic in appropriate services or adapters.
* Reuse existing forms, traits, entities, and utilities instead of duplicating functionality.
* Preserve backward compatibility unless a breaking change is explicitly required.
* Avoid unrelated refactoring.

## Development guidelines

* Understand the existing UI and backend workflow before adding new functionality.
* Keep scientific logic in `tvb_library` and application orchestration in `tvb_framework`.
* Consider persistence, form validation, adapter configuration, and datatype compatibility when adding UI features.
* For simulator changes, verify how parameters move from UI/forms to adapters and finally to `tvb_library`.
* New hybrid-model UI functionality should build on the public hybrid APIs from `tvb_library` rather than duplicating simulation behavior.
* Avoid unnecessary dependencies.

## Configuration and secrets

* Never commit credentials, tokens, passwords, private URLs, or machine-specific settings.
* Do not hard-code deployment-specific configuration.
* Use existing TVB configuration mechanisms.

## Data and persistence

* Use existing TVB datatypes and database entities where possible.
* Be careful when changing stored entities or serialization, as changes may affect existing projects.
* Avoid committing generated project data, database files, or large simulation outputs.
* Database or migration-related changes require focused tests and backward-compatibility consideration.

## Testing expectations

* Add or update tests for changed behavior.
* Prefer regression tests for bug fixes.
* Test form validation and adapter configuration for UI-facing changes.
* Test both successful and invalid user input when applicable.
* For hybrid simulation UI changes, verify that the generated configuration matches the expected `tvb_library` hybrid API.
* Run focused tests first, followed by the relevant framework suite.

## Build and integration

* Keep dependencies between `tvb_framework` and `tvb_library` explicit.
* Do not copy scientific implementations from `tvb_library` into the framework.
* Verify integration with existing TVB simulator workflows when changing simulation-related functionality.
* Preserve compatibility with the wider `tvb-root` application.

## Git and collaboration

**Do not commit or push any changes.**

The agent may inspect files, modify the working tree, run tests, and prepare changes for developer review. All commits and pushes must be performed manually by the developer.

Additionally:

* Do not create or amend commits.
* Do not merge, rebase, force-push, or rewrite Git history.
* Keep changes limited to the requested task.
* Inspect existing implementation and tests before introducing new patterns.
* Clearly identify API, persistence, or UI behaviour changes for review.

## License

TVB is distributed under the GNU General Public License v3 or later.

Preserve existing copyright, license, and citation headers when modifying files.
