# Hybrid simulation GUI implementation plan

## Goal and first-version scope

Add a separate **Hybrid Simulation** workflow to the Simulator top overlay menu, next to **Simulation Cockpit** and **Phase plane**. The workflow starts from one existing region `Connectivity`, partitions all of its nodes into subnetworks, derives and edits intra- and inter-subnetwork projections, configures one model and integrator per subnetwork, configures global monitors and simulation length, and launches `tvb.simulator.hybrid.Simulator` through the normal framework operation machinery.

The first version should deliberately stay small:

- Region simulations only; surface simulations, stimuli, branching, parameter-space exploration, importing/exporting configurations, and reopening/copying a hybrid setup are out of scope.
- Keep the in-progress configuration only in the CherryPy session. At launch, serialize a framework view model into the operation folder so asynchronous execution receives a stable snapshot.
- Use the hybrid simulator's default `backend="python"`; do not add Numba/GPU controls.
- Build every intra- and directed inter-subnetwork projection automatically from the selected parent connectivity. Use the hybrid `Linear` coupling with default parameters and expose only weights and tract lengths.
- Initially support deterministic Heun and Euler integrators. Each subnetwork owns its integrator, but all integration time steps must match because the hybrid simulator requires a common `dt`.
- Initially support `Raw` and `TemporalAverage` monitors. Select one observable per subnetwork so a result can be merged into one standard `TimeSeriesRegion` in the original connectivity-node order.
- Use the first coupling variable of the source and target models for each projection. Show this derived mapping in the summary but do not expose coupling-variable, scale, mode-map, conduction-velocity, or coupling-function controls yet. Use a fixed documented conduction velocity (the existing projection helper default, `3.0`).

## User flow

Implement a linear five-step wizard at `/burst/hybrid/`, with Previous/Next navigation and server-side validation at each transition.

1. **Connectivity and subnetworks**
   - Select an existing `Connectivity` from the current project.
   - Create, rename, and remove subnetworks, then assign regions using the same 3D connectivity selection and text-grid interaction used by **Set up Region Model**.
   - Give each subnetwork a stable generated ID independent of its editable name.
   - Display an explicit **Unassigned** group. Require at least two non-empty subnetworks and require every connectivity node to belong to exactly one subnetwork before continuing.
   - When connectivity changes, reset all later steps because their node indexes and matrix dimensions are no longer valid.

2. **Projections**
   - Generate one `IntraProjection` definition for each subnetwork and one directed `InterProjection` definition for every ordered pair of distinct subnetworks. For `N` subnetworks this produces `N` intra projections and `N * (N - 1)` inter projections.
   - Initialize each matrix from the parent connectivity using `weights[np.ix_(target_indices, source_indices)]` and `tract_lengths[np.ix_(target_indices, source_indices)]`. Rows are target nodes and columns are source nodes, matching `extract_connectivity_subset()` and `BaseProjection`.
   - Present a source/target projection list and an editable weights/tract-length matrix for the selected projection. Zero weights represent absent edges; retaining a zero-weight entry does not create a separate enable/disable concept.
   - Reuse the Connectivity Viewer matrix styles, color-scale utilities, labels, and cell-edit behavior, but post edits to the hybrid session instead of launching `ConnectivityCreator`. The hybrid matrix component must support rectangular inter-projection matrices; do not create temporary or persisted `Connectivity` datatypes.
   - Validate matrix shape, finite non-negative tract lengths, and finite weights on save. Preserve full precision in the posted JSON.

3. **Subnetwork dynamics**
   - Show a compact list or accordion with one section per subnetwork.
   - For each section, select a model using `ModelsEnum` and render its existing form from `get_form_for_model()`. Reuse the existing model view-model classes and form parsing rather than defining duplicate parameter forms.
   - Select deterministic Heun or Euler and render the existing integrator form from `get_form_for_integrator()`.
   - Select exactly one entry from that model's `variables_of_interest`. This gives the merged output one consistent variable slot while still allowing different models to expose different observables.
   - On model changes, reset only that subnetwork's model parameters and selected observable. On integrator changes, reset only that subnetwork's integrator parameters.
   - Validate that every model has at least one coupling variable and that all subnetworks use the same `dt`. Report errors beside the affected subnetwork before allowing the next step.

4. **Monitors and run settings**
   - Choose one or both of `Raw` and `TemporalAverage`, using the existing monitor view models/forms for the period field. Default to `TemporalAverage`.
   - Enter simulation name and simulation length using the existing final simulator form fields and validation style.
   - Reject monitor periods greater than simulation length and require positive length/period values.

5. **Review and launch**
   - Summarize the parent connectivity, node count per subnetwork, model, integrator and `dt`, selected observable, all projection shapes, monitor periods, fixed linear coupling, fixed conduction velocity, and simulation length.
   - Provide Edit links back to each step and one Launch action. Disable repeat submission after the first accepted launch.
   - Launch asynchronously through `SimulatorService.async_launch_and_prepare_simulation()` and `OperationService`, create a normal `BurstConfiguration`, and return the burst ID so the existing progress/history UI can track it.
   - Clear the hybrid session state after the launch snapshot has been stored successfully. A browser refresh before launch must retain the current wizard state.

## Framework-side data model

Create view models under `tvb_framework/tvb/core/entities/file/simulator/hybrid_view_model.py`:

- `HybridSubnetworkViewModel`: stable ID, name, node-index array, model view model, integrator view model, and selected observable name.
- `HybridProjectionViewModel`: projection kind (`intra` or `inter`), source and target stable IDs, dense weight matrix, and dense tract-length matrix. Dense arrays keep web editing and H5 serialization simple; convert them to CSR only when building the library objects.
- `HybridSimulatorAdapterModel`: parent connectivity GID, list of subnetworks, list of projections, monitor view models, simulation length, and the inherited operation metadata from `ViewModel`.

Use real nested `ViewModel`/`List(of=...)` traits rather than an opaque JSON configuration string. `ViewModelLoader` already recursively stores referenced `HasTraits`, which allows the operation launch to receive model, integrator, monitor, subnetwork, and projection values without inventing another persistence format. The view model is persisted only as an operation input; no new database datatype or migration is needed.

Add a dedicated `HybridSimulatorContext` in `tvb_framework/tvb/interfaces/web/entities/context_hybrid_simulator.py`. It should have only the hybrid view model and current-step keys, plus initialize/reset helpers. Do not add hybrid fields to `SimulatorContext`, because classic and hybrid wizards can be open independently and the classic copy/branch assumptions do not apply.

## Adapter and library bridge

Add `tvb_framework/tvb/adapters/simulator/hybrid_simulator_adapter.py` and register its module name in `tvb_framework/tvb/adapters/simulator/__init__.py` so normal framework introspection creates an algorithm entry.

The adapter should:

1. Load the parent `Connectivity` once from its GID.
2. Convert every configured model, integrator, and monitor view model with the existing `view_model_to_has_traits()` mechanism.
3. Construct each library `Subnetwork` with `name`, `model`, `scheme`, `nnodes`, and the original global `node_indices`.
4. Convert projection matrices to `scipy.sparse.csr_matrix` and call `create_intra_projection()` or `create_inter_projection()` from `tvb.simulator.hybrid.projection_utils`. Use the first configured model coupling-variable name for both the source and target mapping, the fixed `Linear()` function, fixed conduction velocity, and the shared integrator `dt`.
5. Attach intra projections to `Subnetwork.projections`, put inter projections in `NetworkSet.projections`, then construct `NetworkSet` and `tvb.simulator.hybrid.Simulator(backend="python")`.
6. Call `configure()` during adapter configuration so library validation failures become concise `LaunchException`/`InvalidParameterException` messages before execution starts.
7. Estimate memory, disk, and runtime from total nodes, model state-variable counts, common `dt`, simulation length, and monitor periods. Keep estimates conservative and simple; no Numba compilation estimate is needed.
8. Run the hybrid simulator and store one standard `TimeSeriesRegion` per selected monitor, linked to the original connectivity. Because each model contributes one selected observable and every subnetwork has `node_indices`, the hybrid simulator returns `(time, 1, original_node_count, mode)` in original connectivity order. Label the variable dimension `Hybrid observable`, and place a compact mapping from subnetwork name to actual observable in the time-series title or user tag.
9. Return only the time-series indexes in the MVP. Do not create `SimulationHistory`, because branching/resume is not supported and the classic history object cannot represent several models and integrators.

Keep construction and validation in small pure helper methods (`build_subnetworks`, `build_projections`, `build_simulator`) so controller tests do not need to execute a simulation and adapter tests can verify the exact library objects.

## Controller, forms, templates, and menu integration

Add the following framework components:

- `tvb_framework/tvb/interfaces/web/controllers/simulator/hybrid_simulator_controller.py`: index, step GET/POST handlers, matrix-data JSON endpoint, reset, review, and launch.
- `tvb_framework/tvb/interfaces/web/controllers/simulator/hybrid_simulator_wizard_urls.py`: named URLs for the five steps; avoid scattering route strings through Python and JavaScript.
- `tvb_framework/tvb/adapters/forms/hybrid_simulator_forms.py`: connectivity, subnetwork dynamics, monitor/run, and final validation forms composed from existing model/integrator/monitor forms.
- `tvb_framework/tvb/interfaces/web/templates/jinja2/burst/hybrid/`: a small wizard shell and one template per step. Include existing connectivity viewer/region selector fragments where their contracts fit.
- `tvb_framework/tvb/interfaces/web/static/js/hybrid_simulator.js`: subnetwork assignment, projection selection/editing, JSON submission, and launch behavior. Reuse existing global request/message helpers and matrix color utilities.
- `tvb_framework/tvb/interfaces/web/static/style/hybrid_simulator.css`: only layout rules not already supplied by simulator, connectivity, and region-selector styles.

Mount `HybridSimulatorController` at `/burst/hybrid/` in `tvb_framework/tvb/interfaces/web/run.py`. Add a `SUB_SECTION_HYBRID_SIMULATION` title in `tvb_framework/tvb/interfaces/web/structure.py`, then add the menu item to `BaseController.burst_submenu` in `tvb_framework/tvb/interfaces/web/controllers/base_controller.py` immediately after Simulation Cockpit.

The controller should obtain connectivity rendering data through `ConnectivityViewer.get_connectivity_parameters()`, as `RegionsModelParametersController` already does. Reuse `TVBUI.RegionAssociatorView` for synchronized 3D/text-grid selection, but supply hybrid callbacks that assign selected nodes to a subnetwork. If the current associator cannot display arbitrary group names/colors without model-specific assumptions, add the smallest optional callback/label extension to `region_associator.js`; keep its existing Region Model behavior unchanged.

For projection editing, extract only the reusable matrix behavior needed from `matrixScript.js` or wrap it with hybrid-specific initialization. Do not call `saveSubConnectivity()` and do not modify `ConnectivityCreator`; its persistence and square-matrix assumptions are not part of this flow.

## Validation rules

Centralize cross-step validation in the hybrid view model or a small validator used by both controller and adapter. The adapter must repeat all launch-critical checks because session requests are not a trust boundary.

- Connectivity GID resolves to a region connectivity in the current project context.
- There are at least two subnetworks; names are non-empty and unique for display.
- Node indexes are integers in range, appear exactly once, and cover all connectivity nodes.
- Exactly the expected set of intra and directed inter projection definitions exists.
- Every projection shape is `(target.nnodes, source.nnodes)` and both matrices contain finite values; tract lengths are non-negative.
- Every subnetwork has a supported model, supported deterministic integrator, one valid observable, and at least one coupling variable.
- All `dt` values are positive and identical.
- At least one supported monitor exists; monitor periods and simulation length are positive, and periods do not exceed the length.
- Before storage, verify the result arrays have one observable, the original connectivity node count, and compatible monitor time axes.

Return actionable messages naming the subnetwork or projection, for example: `Projection Cortex -> Thalamus has weights shape (4, 3); expected (5, 3)`.

## Implementation sequence

1. **View model and validator**
   - Add the three view models, session context, reset semantics, and validation helpers.
   - Add H5 round-trip tests proving nested subnetworks, models, integrators, projections, and monitors survive operation-input serialization.

2. **Backend adapter**
   - Register the adapter, implement library-object construction, validation, estimates, execution, and `TimeSeriesRegion` storage.
   - First test with a programmatically built two-subnetwork configuration before connecting any web pages.

3. **Wizard shell and node grouping**
   - Mount the controller, add menu/structure entries, implement connectivity selection and session reset, and adapt Region Model selection for named subnetworks.
   - Do not proceed to projection configuration until full, unique node coverage is enforced.

4. **Projection editor**
   - Generate default intra/inter slices, implement the rectangular weights/tract-length editor, save edits to session, and add server-side matrix validation.

5. **Per-subnetwork dynamics**
   - Compose the existing model and deterministic integrator forms for each subnetwork, add the single-observable selection, and enforce shared `dt`.

6. **Monitors, review, and launch**
   - Reuse the simple monitor/final fields, render a complete review, connect asynchronous launch to a burst, prevent duplicate launches, and clear session state after successful preparation.

7. **End-to-end polish**
   - Add navigation/error handling, verify menu highlighting and project switching/reset behavior, and run the full browser flow on a small connectivity.

## Tests and proof of completion

Add focused tests alongside the existing framework suites:

- `tvb_framework/tvb/tests/framework/core/entities/file/simulator/hybrid_view_model_test.py`
  - nested view-model H5 round trip;
  - validation for node coverage, duplicate membership, matrix shapes, unsupported integrators, common `dt`, and monitor periods.
- `tvb_framework/tvb/tests/framework/adapters/simulator/hybrid_simulator_adapter_test.py`
  - exact intra/inter CSR slices and source/target orientation;
  - library object construction with original `node_indices`;
  - synchronous two-subnetwork launch with different model classes;
  - one `TimeSeriesRegionIndex` per monitor with shape `(samples, 1, connectivity_nodes, 1)` and the original connectivity reference;
  - invalid configurations fail before operation execution.
- `tvb_framework/tvb/tests/framework/interfaces/web/controllers/hybrid_simulator_controller_test.py`
  - initial session/reset and connectivity change;
  - grouping completeness and uniqueness;
  - generated projection count and session updates;
  - per-subnetwork form updates without modifying siblings;
  - review data and one-shot asynchronous launch.
- Existing classic simulator, Region Model, Connectivity Viewer/Creator, and Jinja template tests must remain green.

Manual acceptance scenario:

1. Open **Simulator > Hybrid Simulation** from the overlay menu.
2. Select a small classic connectivity and divide all nodes between two named subnetworks.
3. Edit at least one intra and one directed inter projection in both weights and tract lengths.
4. Assign different models and supported integrators with the same `dt`; select one observable for each.
5. Select `TemporalAverage`, set a short simulation length, review, and launch.
6. Observe normal burst progress, then open the resulting `TimeSeriesRegion` and verify that all nodes are present in the original connectivity order.
7. Run the equivalent configuration directly with `tvb.simulator.hybrid.Simulator` and compare times and data numerically to the framework result.

The feature is complete when the automated tests pass, the manual scenario succeeds without GPU/Numba, and the classic Simulation Cockpit, Set up Region Model, and Connectivity Viewer flows are unchanged.

## Explicit follow-up work

After the complete MVP is stable, separate changes can add saved/reopenable hybrid configurations, more coupling functions and variable mappings, stochastic integrators, additional monitor types, stimuli, multiple observables, Numba selection, GPU execution, branching/history, and parameter-space exploration. None of these should complicate the initial implementation.
