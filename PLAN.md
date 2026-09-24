# Hybrid Simulator GUI – Implementation Plan

## Goal

Expose Hybrid Simulation in TVB Web as a separate simulator workflow.

The first version should allow a user to:

1. select a Connectivity;
2. divide its regions into Subnetworks;
3. configure a Model and Integrator for each Subnetwork;
4. create the required IntraProjections and InterProjections;
5. configure basic global simulation parameters;
6. launch one Hybrid simulation.

Keep the first implementation small and validate each step before moving to the next one.

---

## Status

Last updated 2026-09-24.

| phase | status |
|---|---|
| Phase 0 – Understand the existing Simulator workflow | **Done** |
| Phase 1 – Hybrid Simulator entry and base layout | **Done** |
| Phase 2 – Configure Subnetworks | **Done** |
| Phase 2 – Refine Subnetwork Configuration | **Done** |
| Phase 3 – Configure each Subnetwork | **Done** |
| Phase 3 addition – Set up region Model | **Done** |
| Phase 4 – Generate Projections | **Done** |
| Phase 5 – Global Hybrid Simulator configuration | **Done** |
| Phase 6 – Launch one Hybrid simulation | **Done** |
| Follow-up features | Not started |

A user can currently select a Connectivity, group its regions into Subnetworks, configure a Model and an
Integrator (with Noise and its Equation) for each of them, place saved Dynamics on a Subnetwork's regions
to give a Model parameter one value per region, save all of it, inspect the Projections generated from the
grouping, choose the Monitors and the simulation length, configure each Monitor, read what the resulting
output will look like, and **launch it**. The simulation runs as an ordinary TVB operation and its
TimeSeries land in the project like any other result.

### Verified by

161 hybrid tests — 67 service, 82 controller, 6 render checks, 6 adapter. The suites the work touches also
pass unchanged: the classic Simulator Cockpit, the burst service, the simulator adapters, and the view
model, forms and serialization suites.

```bash
python -m pytest tvb/tests/framework/core/services/hybrid_simulator_service_test.py \
                 tvb/tests/framework/interfaces/web/controllers/hybrid_simulator_controller_test.py \
                 tvb/tests/framework/interfaces/web/controllers/hybrid_render_check_test.py \
                 tvb/tests/framework/adapters/simulator/hybrid_simulator_adapter_test.py
```

There is no JavaScript test infrastructure in this repository, so the client is covered indirectly: the
render checks render every fragment the wizard puts on the wire and assert on its markup.

### Outstanding

* **Initial conditions.** Settled: no phase exposes them, and the library's random draw from each
  Model's `state_variable_range` is what runs. A seed, and explicit per-Subnetwork arrays, are
  follow-up work.
* **Per-Subnetwork `variables_of_interest`.** Phase 3 makes these editable per Subnetwork. Phase 5's
  Monitors step *reports* whether the counts agree — and so whether output is connectome-ordered or
  concatenated — rather than reconciling them. Phase 6 packages both layouts, and refuses only the one
  combination that cannot mean anything: a projection or Spatial average Monitor over concatenated
  output.
* **Stale artifact:** `interfaces/web/controllers/simulator/__pycache__/hybrid_simulator_wizard_urls.*.pyc`
  has no source any more, left over from an earlier iteration.
* **A substring match waiting for PSE.** `OperationService.initiate_prelaunch` tests
  `'SimulatorAdapter' in operation.algorithm.classname`, which also matches `HybridSimulatorAdapter`.
  Nothing creates a Hybrid operation group today, so it cannot fire; it has to become an exact
  comparison before Hybrid PSE lands. See the Phase 6 summary.
* **A past Hybrid simulation cannot be re-opened.** The history lists them and its entries carry a
  'Load this hybrid simulation' link with nothing behind it.

---

## Phase 0 – Understand the existing Simulator workflow

**Status: Done.** Findings are recorded in the phase summaries below.

Before implementing new UI:

* inspect the current Simulator Cockpit controller, forms, adapters, templates, and JavaScript;
* inspect the existing Setup Region Model functionality;
* identify which components can be reused;
* trace how Simulator Cockpit configuration reaches `tvb_library`;
* identify how simulation configuration is persisted.

### Result

Document the relevant files/classes and decide where the Hybrid Simulator entry point and controller should live.

### Validation

No functional changes.

Discuss the proposed architecture before starting Phase 1.

---

## Phase 1 – Hybrid Simulator entry and base layout

**Status: Done.**

Add **Hybrid Simulator** as a separate option next to the existing
Simulator Cockpit / Phase Plane entry points.

Reuse the existing Simulator Cockpit three-column layout:

- left: Simulation History;
- center: Hybrid Simulator configuration;
- right: Simulation Results.

Only the center configuration area should initially differ from the
classic Simulator Cockpit.

For the first version of the center panel, expose:

- Connectivity selection;
- a way to continue to Subnetwork configuration.

Do not add simulation logic yet.

### Investigation

Before implementation, determine whether the existing three-column
layout/history/results components can be reused directly or should be
extracted into shared components.

### Tests

- Hybrid Simulator entry is accessible;
- existing three-column layout is rendered correctly;
- Simulation History remains functional;
- Connectivity selection works;
- invalid or missing Connectivity is handled correctly;
- classic Simulator Cockpit behavior is unchanged.

### Checkpoint

Review the reused layout, navigation, and initial Hybrid configuration
panel before implementing Subnetwork configuration.

---

## Phase 2 – Configure Subnetworks

**Status: Done** — see the implementation summary below, and the refinement that followed it.

Implement the UI for dividing the selected Connectivity regions into Subnetworks.

After selecting a Connectivity, the Hybrid Simulator configuration should provide a **Configure Subnetworks** action that opens the Subnetwork configuration step.

### Initial behaviour

When Subnetwork configuration is opened:

* create one default Subnetwork, initially named **Subnetwork A**;
* assign all Connectivity regions to Subnetwork A;
* allow the user to rename a Subnetwork;
* allow the user to create additional empty Subnetworks;
* allow the user to remove a Subnetwork when this does not leave the configuration in an invalid state.

Example initial state:

```text
Subnetwork A
  Region 1
  Region 2
  Region 3
  Region 4
  ...
```

After creating another Subnetwork:

```text
Subnetwork A              Subnetwork B
  Region 1
  Region 2
  Region 3
  Region 4
```

### Region assignment

Implement a simple visual way of moving Connectivity regions between Subnetworks.

Preferred interaction:

* drag and drop regions from one Subnetwork to another;
* support selecting multiple regions and moving them together, since Connectivities can contain many nodes;
* clearly indicate the selected regions and the destination Subnetwork;
* preserve the original Connectivity node indices internally.

Every Connectivity region must belong to **exactly one Subnetwork**.

A region must never:

* belong to multiple Subnetworks;
* disappear from all Subnetworks;
* change its original Connectivity index.

The UI representation can use region names, but the stored configuration should rely on the original Connectivity node indices.

### UI technology investigation

Before implementing drag and drop:

1. inspect existing TVB Web JavaScript/components for reusable region-selection or list-management behaviour;
2. inspect **Setup Region Model** in particular;
3. determine whether native HTML5 drag-and-drop and the existing TVB frontend stack are sufficient;
4. avoid introducing a new frontend dependency unless it provides a clear benefit, especially for multi-selection and drag-and-drop.

If a new dependency appears necessary, document:

* why existing TVB functionality is insufficient;
* which dependency is proposed;
* where it would be integrated;
* whether it introduces additional build/runtime dependencies.

Discuss this before adding the dependency.

### Large Connectivities

The interaction should remain usable for Connectivities with many regions.

At minimum, investigate:

* multiple region selection;
* moving all selected regions at once;
* scrolling within Subnetwork region lists.

Do not implement advanced filtering/search unless it becomes necessary for usability.

### State

Store enough information in the Hybrid Simulator configuration to reconstruct the Subnetwork assignments, for example conceptually:

```text
Subnetwork A
    name
    node_indices = [0, 1, 4, 7, ...]

Subnetwork B
    name
    node_indices = [2, 3, 5, 6, ...]
```

Do not create `tvb.simulator.hybrid.Subnetwork` objects yet. That mapping will be handled in the next phases.

### Tests

Test that:

* all Connectivity regions initially appear in Subnetwork A;
* a Subnetwork can be created;
* a Subnetwork can be renamed;
* regions can be moved between Subnetworks;
* multiple selected regions can be moved together;
* region assignments preserve their original Connectivity indices;
* one region cannot belong to multiple Subnetworks;
* no region can remain unassigned;
* invalid Subnetwork removal is prevented or handled correctly;
* configuration survives navigation between the Hybrid Simulator steps;
* classic Simulator Cockpit behaviour remains unchanged.

### Checkpoint

Stop after the Subnetwork grouping UI is functional.

Review:

* the drag-and-drop interaction;
* multiple selection;
* behaviour with a large Connectivity;
* how Subnetwork assignments are stored;
* whether a 3D visualization would improve usability.

Do not start Model or Integrator configuration yet.

---

## Phase 2 – Implementation Summary

### Where the configuration lives

Subnetworks are stored as `HybridSubnetworkViewModel` (`name`, `node_indices`) on
`HybridSimulatorAdapterModel.subnetworks`, kept in the CherryPy session.

* `node_indices` are the **original Connectivity indices**, never renumbered, so the grouping can be
  sliced straight out of `weights`/`tract_lengths` in Phase 4.
* Nothing is written to the database or H5 yet. This matches the classic Simulator Cockpit, which
  also keeps the in-progress configuration in session until launch. Persistence belongs with the
  operation in Phase 6.
* Deliberately *not* `tvb.simulator.hybrid.Subnetwork`: the UI only needs a name and a set of nodes,
  and building library objects would drag in Model/Integrator before Phase 3.

### The server owns the state

Two request shapes, distinguished by the controller decorator:

| decorator | used for | returns |
|---|---|---|
| `@expose_fragment` | moving between wizard steps | rendered HTML |
| `@expose_json` | editing the grouping | the **complete** new state |

Every edit (add / rename / remove / move) round-trips to the server, which validates it through
`HybridSimulatorService` and answers with the whole configuration — not a delta. The browser discards
its copy and redraws. A rejected change returns the *unchanged* state plus an error message, so the
screen can never show a grouping the server does not hold.

### Layering

* `HybridSimulatorService` — all grouping rules and invariants; no web concepts.
* `HybridSimulatorController` — thin; every mutation funnels through one `_change_subnetworks` helper.
* `hybrid_subnetworks.js` — rendering and interaction only; it decides nothing.

### Invariants enforced by the service

Every Connectivity node belongs to exactly one Subnetwork; at least one Subnetwork always exists;
names are non-empty and unique. `prepare_subnetworks` re-creates the default grouping whenever the
stored one is not an exact partition of the current Connectivity, which is what makes a Connectivity
change self-heal instead of corrupting the configuration.

Removing a non-empty Subnetwork moves its regions into the first remaining one rather than refusing,
so no region can be orphaned. Empty Subnetworks are allowed *on the board* (they are the drop target
being prepared) and discarded when the grouping is saved.

### UI structure

Connectivity and Subnetworks are cockpit wizard steps. The grouping board lives in the third column, as
the configuration belonging to the Subnetworks step — see the refinement summary below. The wizard
accumulates steps read-only, mirroring `wizzard_submit` / `previousWizzardStep` in `bursts.js`.

### No new frontend dependency

Native HTML5 drag-and-drop plus the existing jQuery / `doAjaxCall` / `displayMessage` stack.
`TVBUI.RegionSelectComponent` (used by Setup Region Model) was evaluated and not reused: it models one
selection over one flat checkbox grid whose DOM order *implies* the node index, which does not carry
over to N containers owning disjoint region sets. Its interaction idiom — click / Ctrl-click /
Shift-range, then apply — was reused instead.

### Styling

The hybrid step reuses TVB's own `fieldset` / `fieldset dt` rules rather than styling from a blank
slate; overriding them is what produced mismatched insets, separators and text colours. Two traps
worth remembering: the configuration column is a near-transparent cream over a **light** page (light
text is invisible), and `base.css` styles the bare `<header>` element as the site's fixed top nav, so
an in-box title bar must not be a `<header>`.

### Tests

Service and controller are covered by Python tests (partition invariants asserted after every
operation). There is no JavaScript test infrastructure in this repository, so the client is verified
out of tree.


---

## Phase 2 – Refine Subnetwork Configuration

**Status: Done** — see the implementation summary below.

Refactor the Phase 2 UI so Subnetwork configuration remains on the main **Hybrid Simulator** page.

Remove the separate Subnetwork configuration page and reuse the current third column, which is reserved for Results / Visualization.

Workflow:

* user selects a Connectivity in the main Hybrid Simulator configuration;
* after pressing **Next**, show the existing Subnetwork configuration UI in the third column;
* keep the current Subnetwork operations and drag-and-drop behaviour unchanged where possible;
* remove the **Configure Subnetworks** button, since navigation to a separate page is no longer needed;
* add a **Save Configuration** action for the Subnetwork configuration;
* update the Subnetwork summary in the main simulator column only after the configuration is saved;
* when the user presses **Next** to continue to the following Hybrid configuration step, clear the third column.

The third column should become the contextual configuration area for the current Hybrid Simulator step.

Future phases can reuse the same area for Model / Integrator configuration, Projections, and other Hybrid settings.

### Tests

* Subnetwork configuration is displayed in the third column;
* existing drag-and-drop and multi-selection behaviour still works;
* saved Subnetwork configuration updates the summary correctly;
* navigating between Hybrid configuration steps preserves the expected state;
* the separate Subnetwork configuration page and navigation are removed;
* classic Simulator Cockpit behaviour remains unchanged.

### Checkpoint

Review the new single-page Hybrid Simulator workflow before continuing with Phase 3.

---

## Phase 2 Refinement – Implementation Summary

### The third column is now step-contextual

Every wizard fragment declares the configuration it wants in the third column:

```html
<form ... data-hybrid-context-url="/burst/hybrid/configure_subnetworks"
          data-hybrid-context-title="Subnetworks">
```

After any wizard render, `_afterHybridRender` in `hybrid_simulator.js` reads that attribute off the
**last** form in the stack (the step being configured) and either loads it into
`#hybrid-context-column`, hiding `#hybrid-results-view`, or empties the column and hands it back to the
Results view. The Connectivity step declares nothing, which is what clears the column when stepping
back; a Phase 3 step gets the same behavior for free by declaring its own url, or none.

The full-width detour is gone with it: `colscheme-1` swapping, `resetLayout`,
`hybridBackToSimulator`, `hybridLoadFragment`, the *Configure Subnetworks* button and the
`is_subnetwork_fragment` branch were all removed.

### Save Configuration, and why a draft was needed

The summary must change only on save, but every grouping edit still has to round-trip so the server can
validate it. So the board edits a **draft** held in its own session slot
(`HybridSimulatorContext.KEY_SUBNETWORKS_DRAFT`); `save_subnetworks` is the only writer of
`HybridSimulatorAdapterModel.subnetworks`.

| where | holds | changed by |
|---|---|---|
| `hybrid_simulator.subnetworks` | what the wizard step lists | `save_subnetworks` only |
| session draft | what the board shows | add / rename / remove / move |

Drag-and-drop, multi-selection and every `HybridSimulatorService` rule are untouched — only the list
they are applied to changed. `copy_subnetworks` keeps the two apart: the grouping operations mutate
`HybridSubnetworkViewModel` instances in place, so sharing them would let a drag silently rewrite the
summary.

The draft survives stepping away and back (unsaved work is not thrown away), and is dropped together
with the grouping when the Connectivity changes or the configuration is reset. Empty Subnetworks stay
on the board as drop targets and are discarded on save.

Each answer carries `is_modified`, computed against `discard_empty_subnetworks(draft)` so that an empty
Subnetwork prepared as a drop target is not reported as a pending change. The board shows it as
*Unsaved changes* / *Configuration saved*, which is what stops the board and the summary next to it
from disagreeing silently.

### Fixed in passing

`main_hybrid_simulator.html` called `displayBurstTree`, which lives in `bursts.js` and is not loaded on
this page — the Results tab raised a `ReferenceError`. Replaced by a local `displayHybridResultsTree`
built on `updateTree` from `projectTree.js`, which the page does load.

### Tests

Python: 29 controller tests and 25 service tests, asserting the partition invariants after every
operation plus the new draft/save split — that editing leaves the saved grouping alone, that saving
updates the summary and discards empty Subnetworks, and that both saved and unsaved groupings survive
navigation. The classic Simulator Cockpit suite (41 tests) still passes untouched.

Client: there is no JavaScript test infrastructure in this repository, so the board was driven out of
tree in jsdom against the **real** `hybrid_simulator.js` / `hybrid_subnetworks.js`, the **real**
Jinja-rendered fragments and TVB's own jQuery — covering multi-selection (click / Ctrl / Shift / Select
all), dragging a multi-region selection between Subnetworks, save refreshing the summary, and the third
column filling and clearing as steps change. Worth adding to the repository if a JS test dependency is
acceptable.

---

## Phase 3 – Configure each Subnetwork

**Status: Done**, checkpoint included — the translation into `tvb.simulator.hybrid.Subnetwork` objects
it asks for is implemented in Phase 4, which needs those same objects. See the implementation summary
below, and Phase 4's.

For each Subnetwork allow selection and parameter editing of:

* Model;
* Integrator (including its nested Noise and, for Multiplicative Noise, Equation sub-fragments).

Start from the defaults supplied by the selected Model and Integrator classes, then reuse the classic
Simulator Cockpit's own forms and rendering to display and edit their parameters.

### Investigation findings

These constrain the design and are the reason for the decisions below.

1. **The classic Cockpit never renders these sub-fragments inline.** It splits Model/Integrator/Noise
   across six sequential wizard steps (`simulator_controller.py`: `set_model`, `set_model_params`,
   `set_integrator`, `set_integrator_params`, `set_noise_params`, `set_noise_equation_params`) and
   actively suppresses nesting — `SimulatorModelFragment.model` carries no subform at all, and
   `set_integrator` sets `form.noise.display_subform = False`.

2. **Inline nesting would need new plumbing.** With `display_subform = True`,
   `form_fields/select_field.html` emits an inline script calling `refreshSubform` and
   `setEventsOnFormFields`. `setEventsOnFormFields` is *not* global — it is defined per page in
   `bursts_dynamic.js`, `spatial/model_parameters.js` and `spatial/transfer_function_apply.js`, none of
   which the Hybrid page loads, so it would raise a `ReferenceError` (the same class of bug as the
   `displayBurstTree` one fixed in Phase 2). `flow_controller.refresh_subform` is equation-specific
   (it calls `spatial_model.get_equation_information()`) and cannot refresh an Integrator→Noise
   subform. Nested display would require a hybrid-specific refresh endpoint plus `session_key` /
   `form_key` on the SelectFields.

3. **`is_dt_disabled` already exists.** `IntegratorForm.__init__(self, is_dt_disabled=False)` disables
   `dt` from `fill_from_trait`, which is exactly what the shared `dt` needs.

4. **The selection fragments are reusable as they are.** `SimulatorModelFragment` and
   `SimulatorIntegratorFragment` are duck-typed on `.model` / `.integrator`, so they operate on a
   `HybridSubnetworkViewModel` without modification.

5. **`SimulatorIntegratorFragment.fill_trait` replaces the Integrator unconditionally**
   (`datatype.integrator = self.integrator.value.instance`), unlike `SimulatorModelFragment.fill_trait`
   which guards on a class change. Reusing it verbatim would discard edited Integrator parameters on
   every re-submission of that step.

6. **The form POST namespace is flat.** `FormField.fill_from_post` hands the same unprefixed POST dict
   to its subform, and field names are unprefixed (`dt`, `noise`, `a`, `tau`). Only one Subnetwork's
   form can therefore be on screen and submitted at a time.

### Decisions

| decision | choice | why |
|---|---|---|
| layout | the steps are ordinary wizard steps of the **configuration column**, stacked under a Subnetwork dynamics step | no new subform plumbing (finding 2); reuses the classic step chain almost verbatim (finding 1), in the column the rest of the wizard already lives in |
| editing model | draft in session + explicit **Save Configuration** | consistent with the Phase 2 board; keeps the wizard summary honest |
| Model form scope | the full `ModelForm`, `variables_of_interest` **editable** per Subnetwork | the Cockpit's own rendering, unmodified |
| Subnetwork identity | a stable generated `id` on `HybridSubnetworkViewModel` | index identity would silently reattach a configuration to the wrong Subnetwork after a rename or removal |

### UI flow

The Subnetworks step is followed by a **Subnetwork dynamics** step, in the configuration column with the
rest of the wizard. That step holds:

* the shared `dt`;
* a read-only summary of what every Subnetwork is configured with;
* a **Subnetwork selector** listing every saved Subnetwork with its region count and its Model. The
  first Subnetwork is selected by default.

Unlike Phase 2's Subnetworks step, it declares **no** `data-hybrid-context-url`, so the third column is
emptied and handed back to the Results view — the Model and Integrator are configured in this same
column, not next to it.

Pressing **Next** on it applies the shared `dt` and opens the configuration of the selected Subnetwork
underneath, as further wizard steps that mirror the classic chain:

| # | step | form | notes |
|---|---|---|---|
| 1 | Model class | `SimulatorModelFragment` | reused unmodified |
| 2 | Model parameters | `get_form_for_model(cls)()` | full `ModelForm`, `variables_of_interest` editable |
| 3 | Integrator class | `SimulatorIntegratorFragment`, `integrator.display_subform = False` | reused unmodified |
| 4 | Integrator parameters | `get_form_for_integrator(cls)(is_dt_disabled=True)`, `noise.display_subform = False` | `dt` read-only, sourced from the shared value |
| 5 | Noise parameters | `get_form_for_noise(cls)()`, `equation.display_subform = False` | only for an `IntegratorStochasticViewModel` |
| 6 | Noise Equation parameters | `get_form_for_equation(cls)()` | only for a `MultiplicativeNoiseViewModel` |

Steps 5 and 6 are skipped exactly as the classic controller skips them, by branching on the configured
Integrator's and Noise's types. They accumulate read-only like every other wizard step, so the whole
per-Subnetwork configuration stays visible while it is built. After the last one, a **Save
Configuration** action commits the draft.

Because every `display_subform` is `False`, `select_field.html` never emits its inline script, so no
`refreshSubform` endpoint and no page-local `setEventsOnFormFields` are needed. That is the point of
reusing the step chain rather than nesting the sub-fragments.

Selecting another Subnetwork drops every step stacked under the dynamics step and starts that
Subnetwork's chain in their place, seeded from its own draft. Whatever was edited for the Subnetwork
being left is kept — the draft holds them all.

### Client-side reuse

There is one wizard stack, so `hybrid_simulator.js` keeps working as it is: the new steps are appended,
locked and stepped back through exactly like the Connectivity and Subnetworks ones. Two additions:

* **the Subnetwork selector must survive locking.** Once the user moves on, the dynamics step is locked
  like every finished step — buttons hidden, fieldsets disabled — but switching Subnetwork is what
  rebuilds the steps below it. The selector is marked `data-hybrid-keep-enabled` and `_lockHybridForm`
  leaves anything inside such a marker alone;
* **switching Subnetwork drops the steps under the dynamics step** and appends the answer in their
  place, then re-reads that step so its selector and summary stop being stale.

Also to update, listed because they are easy to miss:

* `next_button_enabled=False` in `_subnetworks_step_rules` — the Subnetworks step's **Next** is
  currently dead on purpose and is what opens this phase;
* the `HYBRID_WIZARD_STEPS` array, which the stack rebuild walks.

### The server owns the selection

The selected Subnetwork lives in the session (`HybridSimulatorContext.KEY_SELECTED_SUBNETWORK`, holding
a Subnetwork `id`), not in the url. A `select_subnetwork` endpoint sets it and answers with the first
configuration step of that Subnetwork. This keeps every step url static — which is what the id-based
previous-step lookup needs — and keeps the client deciding nothing, the same division Phase 2
established.

### Shared simulation `dt`

`dt` is not a Hybrid Simulator field of its own — it lives on each Subnetwork's Integrator, and
`tvb.simulator.hybrid.Simulator.validate_dts` raises a `ValueError` when Subnetworks disagree
(it compares every Subnetwork against `subnets[0].scheme.dt`).

Avoid the mismatch by construction with a single shared value:

* it is stored as `HybridSimulatorAdapterModel.dt`, defaulting to `IntegratorViewModel.dt`'s own
  default, and is exposed as a `FloatField` **on the Subnetwork dynamics wizard step itself** — so it
  sits in the accumulating wizard record and locks read-only with that step, like every other setting.
  Submitting that step is what applies it, on the way into the steps configuring the first Subnetwork;
* it is applied to the **saved** Integrators as well as to the ones being edited. It is a
  simulation-wide setting applied on its own step, not a pending per-Subnetwork edit, so leaving the
  saved ones behind would report an unsaved change that cannot be saved away;
* every Subnetwork's Integrator is created and kept with that value; changing it rewrites `scheme.dt`
  on every already-configured Subnetwork;
* each Subnetwork's Integrator parameters form is built with `is_dt_disabled=True`, so `dt` shows but
  cannot be edited there.

**A disabled input is not submitted.** `$(form).serialize()` drops it, and `hybridSubmit` serializes
directly — unlike the classic `wizzard_submit`, which strips `disabled` off the fieldset first. The
Integrator parameters handler must therefore inject the shared value into the POST data before
`fill_from_post`, exactly as `set_integrator_params` already does for a branch:

```python
data['dt'] = str(hybrid_simulator.dt)
```

Without it, `FloatField` validation fails on the missing key.

### Model and Integrator scope

`tvb.simulator.hybrid.Subnetwork` accepts any `tvb.simulator.models.Model` and any
`tvb.simulator.integrators.Integrator` generically; only the numba execution backend enforces a fixed
whitelist (a fixed set of Model classes, and only Heun/Euler Integrators, deterministic or stochastic).

Keep selection **fully generic** here: reuse the same class lists the classic Cockpit uses, unfiltered.
Backend selection is a single global choice for the whole `NetworkSet` and is out of scope until
Phase 5/6; compatibility between the chosen backend and the configured Models/Integrators is validated
at launch (Phase 6), where an unsupported combination must fail with a clear error rather than silently
succeed.

Reuse, rather than rebuild:

* `tvb.adapters.forms.model_forms` — `ModelsEnum`, `get_form_for_model`;
* `tvb.adapters.forms.integrator_forms` — `get_integrator_name_list`, `get_form_for_integrator`;
* `tvb.adapters.forms.noise_forms` — `get_form_for_noise`;
* `tvb.adapters.forms.equation_forms` — `get_form_for_equation`;
* `tvb.adapters.forms.simulator_fragments` — `SimulatorModelFragment`, `SimulatorIntegratorFragment`;
* the corresponding `*ViewModel` classes in `tvb.core.entities.file.simulator.view_model`.

#### Model parameters

The full `ModelForm` is rendered, `variables_of_interest` included and editable per Subnetwork.

*Forward dependency:* Phase 5 configures monitors globally, so it must reconcile Subnetworks that
choose different variables of interest. Record the decision there; do not pre-empt it here.

Model parameters are `ArrayField`s. A value must broadcast onto *this Subnetwork's* nodes, so on save a
parameter array's length must be either `1` or that Subnetwork's `nnodes`; anything else — notably an
array sized to the whole Connectivity — is rejected with a message naming the Subnetwork, the parameter
and both acceptable lengths. (Assumption, not derived from the Cockpit: classic relies on the *Setup
Region Model* page to size these against the whole Connectivity, which does not carry over to a
Subnetwork owning a subset of nodes.)

The *Setup Region Model*, *Configure Spatial Vector* and *Configure noise* buttons are **not** rendered
in this column — all three operate on the whole Connectivity and would target the wrong node set.
`FormWithRanges` range parameters are likewise not registered: Hybrid PSE stays a follow-up feature.

#### Switching classes

Switching a Subnetwork's Model or Integrator **class** resets that Subnetwork's parameters to the new
class's defaults. Re-submitting the step without changing the class must *preserve* the edited
parameters — which means the Integrator class step cannot simply delegate to
`SimulatorIntegratorFragment.fill_trait` (finding 5). The handler compares the submitted class against
`type(subnetwork.integrator)` and assigns a fresh instance only when they differ. `SimulatorModelFragment`
already guards this way and is delegated to as-is.

Neither shared fragment is modified, so the classic Cockpit is unaffected.

### Persisted configuration

Extend `HybridSubnetworkViewModel` (Phase 2: `name`, `node_indices`) with:

* `id` — a generated, stable identifier, unchanged by rename, reorder or the removal of another
  Subnetwork;
* `model` — a `tvb.simulator.models.Model` instance, edited parameters included;
* `integrator` — an `IntegratorViewModel` instance, edited parameters included, its `dt` always the
  shared value.

and `HybridSimulatorAdapterModel` with `dt`.

This mirrors how the classic Cockpit persists its own selection — actual instances, not class
identifiers. Nothing is written to the database or H5 yet; persistence belongs with the operation in
Phase 6.

#### Identity and regrouping

Phase 2's board operations keep addressing Subnetworks by `subnetwork_index` — no client change — while
the dynamics draft and the saved dynamics are keyed by `id`. Consequences to implement and test:

* renaming a Subnetwork, reordering, or removing a *different* one preserves its dynamics;
* removing a Subnetwork drops its dynamics;
* `prepare_subnetworks` regenerating the default grouping (a Connectivity change, or a stored grouping
  that is no longer an exact partition) mints new ids, so dynamics keyed by ids that no longer exist are
  discarded rather than reattached.

### Draft and Save Configuration

The Phase 2 split is repeated for the dynamics, for the same reason: every edit must round-trip so the
server can validate it, but the wizard summary may only change on save.

| where | holds | changed by |
|---|---|---|
| `HybridSubnetworkViewModel.model` / `.integrator` | what the wizard step summarises | `save_subnetwork_dynamics` only |
| session draft (`KEY_DYNAMICS_DRAFT`), keyed by Subnetwork `id` | what the column shows | the sub-wizard steps |

The draft is deep-copied from the saved configuration when the step is entered, the way
`copy_subnetworks` already keeps the board and the summary apart — Model and Integrator instances are
mutated in place, so a shared instance would let an edit silently rewrite the summary.

Entering the step seeds any Subnetwork with no saved dynamics with the class defaults
(`ModelsEnum.GENERIC_2D_OSCILLATOR`, `IntegratorViewModelsEnum.HEUN`, the shared `dt`), so a Subnetwork
is never left without a Model or an Integrator.

That makes the original "block progression while any Subnetwork is missing a Model or an Integrator"
gate unreachable. It is replaced by: **Next is disabled while the draft differs from the saved
configuration**, with the button title saying so. Each answer carries `is_modified` per Subnetwork and
overall, shown as *Unsaved changes* / *Configuration saved*, which is what stops the column and the
summary from disagreeing.

The draft survives stepping away and back, and is dropped together with the grouping when the
Connectivity changes or the configuration is reset.

Switching to another Subnetwork keeps the draft — unsaved work on the Subnetwork being left is not
thrown away — since the draft holds every Subnetwork at once.

### Name sanitization belongs here

`NetworkSet.__init__` builds a namedtuple from its Subnetworks' names by joining them with spaces and
splitting the result back into field names, so each tvb_library `name` must be a **valid, unique Python
identifier**. UI display names are not (the Phase 2 default "Subnetwork A" already isn't), and
sanitizing can collide where display names do not ("Sub A" and "Sub-A" both yield "Sub_A").

Since this phase's checkpoint is to verify the configuration translates cleanly into `Subnetwork`
objects, the rule lives here: a `HybridSimulatorService` helper maps display names to identifiers
deterministically, disambiguating collisions, and Phase 6 reuses it rather than re-inventing it.

### Tests

* Model selection, Integrator selection and their edited parameters are stored per Subnetwork;
* defaults are correctly seeded for both when the step is entered;
* Noise parameters are stored for a stochastic Integrator, and Noise Equation parameters for a
  Multiplicative Noise — including that steps 5 and 6 are skipped for a deterministic Integrator and
  for Additive Noise respectively;
* `variables_of_interest` is stored per Subnetwork;
* invalid parameter values are rejected with a useful message;
* a Model parameter array whose length is neither `1` nor the Subnetwork's `nnodes` is rejected, naming
  the Subnetwork, the parameter and both acceptable lengths;
* changing one Subnetwork's Model, Integrator or parameters does not affect another Subnetwork;
* switching a Subnetwork's Model or Integrator **class** resets its parameters to that class's
  defaults;
* re-submitting the Integrator class step **without** changing the class preserves the edited
  Integrator parameters (regression for finding 5);
* the shared `dt` applies uniformly to every Subnetwork, including ones configured before the value was
  last changed, and stays read-only on each Integrator form;
* submitting the Integrator parameters step with **no** `dt` key in the POST data still stores the
  shared value (regression for the disabled-field trap);
* selecting a Subnetwork loads its own configuration, unaffected by what is being edited for another;
* editing the dynamics leaves the saved configuration alone; saving updates the wizard summary;
* **Next** is disabled while the draft differs from the saved configuration;
* a Subnetwork's dynamics survive renaming it, reordering, and removing a different Subnetwork; are
  dropped when that Subnetwork is removed; and are discarded when the grouping is regenerated;
* sanitized names are valid Python identifiers and stay unique when display names sanitize to the same
  string;
* configuration survives navigation between the Hybrid Simulator steps;
* the classic Simulator Cockpit's Model/Integrator forms and fragments are unchanged and its own suite
  still passes.

Client-side, driven out of tree as in Phase 2 (there is still no JavaScript test infrastructure in this
repository):

* the column's sub-wizard advances, steps back and accumulates read-only steps;
* switching Subnetworks reloads the sub-wizard at step 1 for the newly selected one;
* exactly one Subnetwork's form is on screen at a time (finding 6);
* no `ReferenceError` is raised while rendering any of the six steps.

### Checkpoint

Verify that the UI configuration translates cleanly into `tvb.simulator.hybrid.Subnetwork` objects:
`name` (sanitized to an identifier), `model`, `scheme` (the Integrator, built with the shared `dt`),
`nnodes` and `node_indices` from the Phase 2 grouping.

`projections`, `monitors`, `stimuli` and initial conditions stay at their empty/default values for this
checkpoint — they are addressed later (Phase 4 Projections, Phase 5 Monitors). **Initial conditions are
still not covered by any phase**; decide whether they join Phase 5's global configuration before
Phase 6 starts.

---

## Phase 3 – Implementation Summary

Implemented as specified above, with the deviations recorded at the end of this section.

### Where the configuration lives

`HybridSubnetworkViewModel` gained an `id` and a `dynamics`; `HybridSimulatorAdapterModel` gained `dt`.

```text
HybridSimulatorAdapterModel
    connectivity, dt
    subnetworks = [ HybridSubnetworkViewModel
                        id, name, node_indices
                        dynamics = HybridSubnetworkDynamics(model, integrator) ]
```

`dynamics` is its own object rather than two bare attributes on the Subnetwork, because that is what the
reused Cockpit fragments are filled from and into: `SimulatorModelFragment` and
`SimulatorIntegratorFragment` only require a `model` and an `integrator` attribute, so they operate on
`HybridSubnetworkDynamics` unchanged. `model` and `integrator` stay available on the Subnetwork as
read-only properties, so the Phase 2 code and tests reading them are unaffected.

Both are seeded in `__init__` rather than through a trait default: a trait default would be **one shared
instance** for every Subnetwork, and the parameter forms edit these in place, so one Subnetwork's edit
would silently change every other one.

### Two drafts, two save actions

| where | holds | changed by |
|---|---|---|
| `hybrid_simulator.subnetworks[i].node_indices` | what the Subnetworks step lists | `save_subnetworks` |
| `KEY_SUBNETWORKS_DRAFT` | what the grouping board shows | add / rename / remove / move |
| `hybrid_simulator.subnetworks[i].dynamics` | what the dynamics step lists | `save_subnetwork_dynamics` |
| `KEY_DYNAMICS_DRAFT`, keyed by Subnetwork `id` | what the dynamics column shows | the six sub-wizard steps |
| `hybrid_simulator.dt` | the shared step size | submitting the dynamics step, on the way into the chain |

`prepare_dynamics_draft` seeds an entry from the saved dynamics for every Subnetwork missing one and
drops entries keyed by an `id` that no longer exists. That single rule is what makes a rename preserve
the configuration, a removal discard it, and a regenerated grouping not reattach it to whichever
Subnetwork now sits on the same position.

### The step chain

Everything is one wizard stack in the configuration column: the six steps are appended, locked and
stepped back through exactly like the Connectivity and Subnetworks steps, and the third column is
handed back to the Results view for this part of the wizard.

Two things the chain needed:

* **the selector survives locking.** The dynamics step carries the Subnetwork selector and locks like
  any finished step, but switching Subnetwork is what rebuilds the steps under it. The selector is
  marked `data-hybrid-keep-enabled`, and `_lockHybridForm` leaves anything inside such a marker alone.
* **switching Subnetwork rebuilds downwards.** `hybridSelectSubnetwork` drops every step stacked under
  the dynamics step, appends the first step of the newly selected one, then re-reads the dynamics step
  so its selector and summary are not left stale.

Every `display_subform` stays `False`, so `select_field.html` never emits its inline script. That is the
point of reusing the chain rather than nesting: no hybrid `refresh_subform` endpoint and no page-local
`setEventsOnFormFields` are needed, and neither shared fragment had to be modified.

### The traps, and what closed them

* **Disabled `dt` is not submitted.** The Integrator parameters handler injects
  `data['dt'] = str(hybrid_simulator.dt)` before `fill_from_post`, as `set_integrator_params` does for a
  branch. Covered by a test that posts that step with no `dt` key at all.
* **Disabled fieldsets are not serialized.** `hybridSubmit` now enables them for the length of the call,
  as the classic `wizzard_submit` does.
* **`SimulatorIntegratorFragment.fill_trait` replaces the Integrator unconditionally.** The Integrator
  class step compares against `type(dynamics.integrator)` itself and assigns only on a real change, so
  re-submitting the same class keeps the edited parameters.
* **An unknown posted class reaches `fill_trait` as a plain string** and raises `AttributeError` there.
  Both class-selection steps call `form.validate()` first and re-render the step with its error instead.
  The classic Cockpit leaves this unguarded.
* **An endpoint must not call another exposed endpoint.** `select_subnetwork` first returned
  `configure_subnetwork_dynamics()`, which renders; the outer decorator would then have rendered that
  HTML a second time — invisible in tests, where `RENDER_HTML` is off, and broken in the app. Both now go
  through the undecorated `_subnetwork_dynamics_column`. The render check covers it.

### Returning to a Subnetwork shows what it holds

Selecting a Subnetwork from the boxes puts **its whole configuration** on screen — Model, Model
parameters, Integrator, Integrator parameters and, where they apply, Noise and its Equation — with every
step read-only except the last. All the fields are visible at once, and any of them can be reached with
*Previous*. Pressing **Next** from the dynamics step still opens the first step alone, so configuring a
Subnetwork for the first time is still a walk through it; only *returning* to one replays the chain.

`select_subnetwork` renders **all of those steps in one answer** through
`burst/hybrid_subnetwork_chain.html`, with every step but the last marked `is_read_only` — its fieldset
`disabled` and its buttons `visibility: hidden`, which is exactly the state the client's own
`_lockHybridForm` leaves a finished step in. The buttons are hidden rather than omitted so that
`_unlockHybridForm` can reveal them again when the user steps back. Which steps exist follows the
configured Integrator, as stepping through would: a stochastic one adds its Noise step, and a
Multiplicative Noise its Equation step on top of that.

The client therefore only appends what it is given — no request sequencing, no deciding what to lock.

*First attempt, and why it was replaced:* the client was given the remaining step urls in a
`data-hybrid-chain` attribute and walked them itself with GETs (each step answers with the step after
it, and skips its POST branch on a GET, so this was safe). It did not work in the browser and could not
be reproduced outside one — there is no JavaScript test infrastructure here, so the sequencing and the
locking were the one part of this feature that nothing could assert. Rendering the stack server-side
moved all of it under the render checks, which now assert the exact list of forms, that all but the last
are disabled, and that only the last has visible buttons. It also removed a real defect that attempt
introduced: the Jinja whitespace-stripping comment before `data-hybrid-chain` glued it onto the previous
attribute (`data-hybrid-context-title=""data-hybrid-chain="..."`).

*Not* done this way: rendering every step unlocked so any field could be edited in place. The reused
Cockpit forms share one flat POST namespace — `Generic2dOscillator.a` and a Linear equation's `a`
collide — so a single submit across the whole chain would need the fields prefixed, which means no longer
reusing those forms as they are. Read-only-with-Previous is what the classic Cockpit does with a loaded
configuration anyway.

### The Subnetwork dynamics step describes each Subnetwork once

The step first carried both a summary table and the Subnetwork selector boxes, which said the same thing
twice — and worse, not always the same thing: the table described the **saved** dynamics while the boxes
describe the **draft**, so the two disagreed exactly while something was being edited.

The table is gone. Each box now names its Subnetwork, its region count, its Model, its Integrator and,
for a stochastic one, its Noise, using the labels the selectors themselves offered — `TupleEnum` carries
those (`str(member)`), so a Subnetwork is described with the same words it was configured with rather
than with a class name (`Generic 2D Oscillator`, not `Generic2dOscillator`).

The step deliberately carries **no** saved/unsaved marker: the boxes show the draft and nothing says so.
A marker was tried and removed as clutter. Nothing is lost silently, because the two places that need a
saved configuration refuse to proceed without one and say why — `set_projections`, and the Subnetworks
step's own Next. `is_modified` is still computed on the rendering rules, which is what those gates read.

### Deviations from the specification above

1. **The shared `dt` is applied to the saved Integrators too**, not only to the draft. It is a
   simulation-wide setting applied on its own step, not a pending per-Subnetwork edit; leaving the saved
   Integrators behind reported an unsaved change the user could not save away.
2. **The Subnetworks step's `Next` is gated on the grouping being saved.** The spec only gated the
   dynamics step. The dynamics step configures the *saved* Subnetworks, so opening it over an unsaved
   grouping would configure Subnetworks the configuration does not hold.
3. **Model parameter array lengths are validated on save**, against `1` or that Subnetwork's `nnodes`,
   naming the Subnetwork, the parameter and both accepted lengths. This was flagged as an assumption in
   the spec and is implemented as described.

### Revised after review: the configuration column, not the third one

The first implementation put the Subnetwork selector and the six steps in the **third** column, as its
own wizard stack, which is what the specification above originally described. On review the Model and
Integrator configuration was moved into the **configuration column**, with the rest of the wizard; the
specification above has been rewritten to match. What that changed:

* the third column configures nothing for this step and goes back to showing the Results, the way the
  Connectivity step already leaves it;
* the second wizard stack is gone — `_hybridStackOf`, `_isMainHybridStack`, the `data-hybrid-stack`
  markers and the container arguments threaded through the stack helpers were all removed, since there
  is one stack again;
* `burst/hybrid_dynamics_fragment.html` and `burst/hybrid_subnetwork_dynamics.html` are gone with it.
  The steps render through `hybrid_simulator_fragment.html` like every other wizard step, which grew the
  selector and the Save Configuration branch;
* `configure_subnetwork_dynamics` is gone. `select_subnetwork` now answers with the first step of the
  newly selected Subnetwork rather than with a re-rendered column;
* the selector needed `data-hybrid-keep-enabled`, because it now sits on a step that gets locked;
* the styling moved from light-on-dark to the configuration column's dark-on-light, and the overrides
  that had been needed to make TVB form fields legible on the third column's dark ground were dropped —
  base.css styles them correctly here with nothing extra;
* **the `Apply` workaround disappeared.** It existed only because the shared `dt` sat on a step whose
  `Next` was disabled for want of a Phase 4 step. The chain now follows that step, so `Next` applies
  `dt` on its way into the first Subnetwork's Model, and the button is an ordinary `Next` again.

### Tests

Python: 88 hybrid tests pass — 39 service, 47 controller, 2 render checks. The controller tests cover the
draft/save split per Subnetwork, the shared `dt` reaching Integrators configured before it changed, the
disabled-`dt` regression, class-switch reset versus re-submit preservation, per-Subnetwork isolation,
identity across rename and removal, and the parameter-shape refusal. The classic Simulator Cockpit suite
(41 tests) passes untouched, as do the view model and simulator adapter suites (21 tests).

One pre-existing fragility was fixed in passing: `cherrypy.request.method` is shared and leaked between
tests, so `_configured_hybrid_simulator` now sets `GET` explicitly and restores what it found. Two tests
had been passing only because of the method a previously-run test happened to leave behind.

Client: still no JavaScript test infrastructure in this repository. The render check renders the whole
six-step chain — including a stochastic Integrator with Multiplicative Noise and its Equation — and
asserts each step's action url, that `dt` is present and disabled, and that the closing step offers the
save action. That is what stands in for a JS test here, and it is what caught the double-render bug.

---

## Phase 3 addition – Set up region Model

**Status: Done.**

The classic Cockpit offers a **Set up region Model** button next to the Model parameters, which opens a
page of its own. The Hybrid Simulator offers the same action on its Model parameters step, but fills the
**third column** with it instead of navigating away, and shows only the right-hand half of that page —
the region list — not its 3D view.

### What the classic page actually does

It is not a free-form per-region parameter editor. Each region is given a saved **Dynamic**, created on
the Phase plane page; `RegionsModelParametersController.index` bails out to a "no dynamics" page when the
user has none. On submit, every node's Dynamic is read and
`SerializationManager.write_model_parameters` groups them into one array per parameter, **contracting a
constant array back to a single value**. That is exactly the shape a Subnetwork's Model needs, and
exactly what the Phase 3 `1`-or-`nnodes` validation already expects.

### Decisions

| decision | choice | why |
|---|---|---|
| region scope | only the selected Subnetwork's own regions | its Model applies to the nodes it owns, so a parameter value is needed for each of those and no other |
| Model class | only Dynamics built on the Subnetwork's configured Model class are offered | a conflict cannot arise, and the wizard step stays the only place a Model class is decided. The classic page instead **overwrites** the Simulator's Model with whatever class the chosen Dynamics carry |
| region selection | the Phase 2 board's click / Ctrl / Shift idiom | see below |
| save | into the dynamics draft, committed by the existing **Save Configuration** | one save action per Subnetwork, and the wizard summary stays honest |

### Why the classic component could not be reused

`TVBUI.RegionAssociatorView` drives the 3D view through globals — `GVAR_interestAreaNodeIndexes`,
`CONN_pickedIndex`, `GFUNC_toggleNodeInInterestArea`, `GFUNC_updateLeftSideVisualization` — and binds a
`#GLcanvas` click handler. None of those exist on this page once the left column is dropped. The region
list therefore reuses the interaction already built for the Subnetwork board, in
`hybrid_region_model.js`.

### How it behaves

The panel is opened by its button rather than by a step declaring `data-hybrid-context-url`, so nothing
closes it while the user stays on the Model parameters step; moving on re-syncs the third column, which
is what hands it back to the Results view.

Three actions, mirroring the classic page's own split:

| action | does |
|---|---|
| **Apply to selection** | records which configuration sits on which region; the Model is not touched |
| **Submit** | writes those values onto the Subnetwork's Model, one array per parameter, and answers with the **Model parameters step re-rendered**, which is how the values appear in the middle column |
| **Select all** / **Clear selection** | one toggle over every region of the Subnetwork |

Submit refreshes that step through `hybridReplaceStep`, deliberately *not* through `_afterHybridRender`:
the latter re-syncs the third column, which would close the panel the user is still working in.

A Model parameter needs a value for **every** node of the Subnetwork, so submitting an incomplete
placement is refused and answers with the step unchanged — what is on screen keeps matching the
configuration. The panel says how many regions are still without one. The placement itself lives in
`HybridSimulatorContext.KEY_REGION_MODEL`, keyed by Subnetwork id, because the parameter arrays alone
cannot say which Dynamic produced them. A regrouping that takes regions away from a Subnetwork drops
them from its placement rather than leaving a stale one behind.

### With nothing to place

The panel offers only Dynamics built on the Subnetwork's Model class, so it is often empty at first. It
then names that Model (with the label the Model selector used) and links to the **Phase plane page**,
`/burst/dynamic`, where model configurations are defined — the same page the classic
`model_param_region_empty` template points at.

The link is built in the controller through `build_path`, not in the template: `deploy_context` is only
put in the template context of a full page, not of a fragment, so a template-side
`{{ deploy_context }}/burst/dynamic` would silently render as `/burst/dynamic` and break under a deployed
context path. It opens in a new tab, so the wizard and its in-session configuration are not left behind;
pressing *Set up region Model* again picks up whatever was saved meanwhile.

### Styling

The warning amber on this column was `#e8b84b`, which measures 4.08:1 against the column's dark ground —
under the 4.5:1 that text at these sizes needs. It is `#ffdd99` throughout now, at 5.75:1. The three
places using it are all on that column: the board's unsaved marker, its empty-Subnetwork count, and a
region with no configuration on it.

### Tests

10 controller tests, 9 service tests and a render check covering both panel states. They assert that
only the Subnetwork's own regions are listed, that only matching Dynamics are offered, that placing
leaves the Model alone, that Submit writes one value per node while contracting the parameters every
configuration agrees on and answers with the Model parameters step, that an incomplete placement is
refused, that the result reaches the saved configuration only through Save Configuration, and that
regions moved to another Subnetwork lose their placement. The classic `region_model_parameters_controller`
suite still passes untouched.

---

## Phase 4 – Generate Projections

**Status: Done** — see the implementation summary below.

Generate IntraProjections and InterProjections from the selected Connectivity and Subnetwork assignments.

For the first implementation:

* slice Connectivity `weights` and `tract_lengths` according to Subnetwork node indices;
* automatically create the necessary IntraProjections;
* automatically create the necessary InterProjections;
* use safe/default coupling-variable selections where possible.

The initial version should avoid requiring users to manually edit projection matrices.

### Projection configuration

Consider:

* `source_cvar`;
* `target_cvar`;
* `cfun`;
* `scale`;
* `cv`;
* `dt`.

Initially, expose only parameters that cannot be safely derived.

### Tests

Given known Subnetwork node indices:

* verify IntraProjection weights/lengths;
* verify InterProjection weights/lengths;
* verify source/target Subnetworks;
* verify coupling-variable selection;
* compare generated projections with the hybrid demo notebooks.

### Checkpoint

Inspect the generated `NetworkSet` before exposing projection editing.

---

## Phase 4 – Implementation Summary

This phase also discharges the **Phase 3 checkpoint**, which asked for the UI configuration to be
translated into real `tvb.simulator.hybrid.Subnetwork` objects: Phase 4 needs exactly those objects, so
they are built here rather than twice.

### Translating the configuration

`HybridSimulatorService.build_library_subnetworks` turns each `HybridSubnetworkViewModel` into a library
`Subnetwork`: the sanitized `name`, the configured `model`, the Integrator as `scheme`, `nnodes` and
`node_indices`.

Two things made this smaller than expected:

* the Integrator **view models subclass the library Integrators** (`HeunStochasticViewModel` is a
  `HeunStochastic`), so one can be handed to `scheme` directly — no conversion layer;
* Model and Integrator are **deep copied first**. `configure()` mutates what it is given, and the
  objects on the configuration are the ones the forms keep editing.

### The projections are the library's job, not this service's

`tvb.simulator.hybrid.projection_utils` already has `create_intra_projection` and
`create_inter_projection`, and both take a `connectivity` plus node indices and do the slicing
themselves — including indexing weights and lengths as **`(target, source)`**, the Connectivity's own
orientation. So `build_network_set` only decides *which* projections exist, and the library builds them.
Nothing about the slicing is reimplemented here, which is what `tvb_framework/AGENTS.md` asks for.

### Coupling variables

The two sides are **not** symmetric, and the library has separate resolvers that say so:

| | indexes | resolver |
|---|---|---|
| `source_cvar` | the history buffer, so a **state variable index** | `resolve_source_cvar` |
| `target_cvar` | the coupling array, so a **slot in the target model's `cvar` list** | `resolve_target_cvar` |

The safe default is therefore `model.cvar[0]` for the source and slot `0` for the target — each side's
own first coupling variable. Choosing them per projection is follow-up work.

### What gets generated

* one `IntraProjection` per Subnetwork, over its own weights and tract-lengths block;
* one `InterProjection` per ordered pair **whose weights block holds any non-zero weight**. A pair the
  Connectivity does not connect in that direction gets none — a projection there would only carry
  zeros — and the step lists those pairs rather than quietly leaving them out;
* a `NetworkSet` over all of it, configured.

### The wizard step

The closing step of the Subnetwork configuration used to be a dead end: its only action stored the
dynamics. It now also carries a **Next** onto the Projections step.

That needed one addition to the client: its own action url stores the dynamics, so Next has to post
somewhere else, and it may not post to the *answer's* url either — `hybridSubmit` treats an answer whose
form id equals the current one as a rejection of this step rather than as the next step. `hybridSubmitTo`
takes the url to post to, and `hybridSubmit` is now a one-line wrapper passing the form's own action.

The Projections step derives everything on every render rather than storing it. Nothing there is a user
choice yet, and keeping sparse matrices in the session would only let them fall out of step with the
grouping. Generating over an unsaved configuration is refused and hands the dynamics step back, the same
rule the Subnetworks step applies to an unsaved grouping.

`SET_PROJECTIONS_URL` is deliberately **not** added to `HYBRID_WIZARD_STEPS`: a stack rebuild walks that
list, and rebuilding would then regenerate the whole `NetworkSet` on every step-back. The existing
fallback lands the user on the dynamics step instead.

### Tests

9 service tests and 6 controller tests, plus a render check that only passes if the configuration really
does translate into a `NetworkSet`. They assert the Intra weights and lengths against a known
Connectivity, that an Inter projection is generated for the connected direction and **not** for the
unconnected one, the `(target, source)` shape after a regrouping, the cvar defaults on both sides, that
`NetworkSet.States` is named after the Subnetworks, and that generating is refused while the dynamics are
unsaved.

Not done: the plan also asks to **compare the generated projections with the hybrid demo notebooks**.
The tests compare against a hand-built Connectivity whose blocks are known constants, which pins the
slicing and orientation, but no demo has been run end to end against a GUI-built configuration. That
belongs with Phase 6, where a simulation can actually be launched and compared.

---

## Phase 5 – Global Hybrid Simulator configuration

**Status: Done** — see the implementation summary below.

Add the remaining simulation-level configuration — the Monitors and the simulation length — and the
translation of the whole configuration into a `tvb.simulator.hybrid.Simulator`.

Reuse existing Simulator Cockpit components where possible. These are configurations are common with the Simulator Cockpit.

### Scope

| question | decision |
|---|---|
| which Monitors | all nine the classic Cockpit offers for a region simulation |
| Monitor scope | **global only** — the `Simulator.monitors` list, not per-Subnetwork recorders |
| per-Monitor `variables_of_interest` | **not exposed** — the library discards it |
| initial conditions | **not in this phase** — Phase 6 decides |
| backend | **not exposed** — always `"python"` |
| Subnetworks with different VOI counts | **reported, not refused** |

Per-Subnetwork recorders (`Subnetwork.add_monitor`, Section 5 of the stimuli demo) were considered and
left out. They would dissolve the VOI reconciliation below — each recorder sees exactly one Model — but
they produce no whole-brain output, which is what Phase 6 has to persist and what the projection
monitors need. They belong with the follow-up features.

### What the library takes

`hybrid.Simulator(nets=..., monitors=[...], simulation_length=...)`. `backend` is a plain constructor
keyword, not a trait, and defaults to `"python"`; the numba backend accepts only a whitelist of Model
classes and Heun/Euler Integrators, so exposing it is left to a follow-up as Phase 3 anticipated.

Monitor view models **subclass the library Monitors** (`RawViewModel(MonitorViewModel, Raw)`), so a
stored view model can be handed to `monitors=` directly — the same reuse Phase 4 got from the Integrator
view models. They are deep copied first, for the reason Phase 4 already records: `configure()` mutates
what it is given, and the stored objects are the ones the forms keep editing.

### Output layout: merged or concatenated

`build_library_subnetworks` always sets `node_indices`, so `NetworkSet._is_merged_mode()` reduces to one
question — do all Subnetworks expose the same *number* of variables of interest.

| | when | output shape | node axis |
|---|---|---|---|
| merged | all VOI counts equal | `(t, n_vois, n_regions, modes)` | the original Connectivity ordering |
| concatenated | any count differs | `(t, Σ vois, Σ nnodes, modes)` | Subnetwork after Subnetwork |

A mismatch is the common case rather than the exception: JansenRit declares four variables of interest
by default, Generic2dOscillator one.

**The step reports which layout the configuration produces and refuses neither.** Phase 3 deliberately
made `variables_of_interest` editable per Subnetwork, and blocking here would take that back. The report
names the Subnetworks and their counts, so a user who wants connectome-ordered output knows exactly what
to change. How concatenated output is packaged into a TVB datatype is Phase 6's problem, and is recorded
as such below.

### Monitors

All nine: Raw, Temporally sub-sample, Spatial average, Global average, Temporal average, EEG, MEG,
Intracerebral / Stereo EEG, BOLD. `BOLD Region ROI` is excluded — the classic Cockpit offers it only for
surface simulations, and the Hybrid Simulator has no surface.

`variables_of_interest` is **not** rendered. `Simulator.validate_dts` assigns `monitor.voi = slice(None)`
to every monitor, so a selection made there is discarded before the first step is integrated; the field
would describe something the simulation does not do. Everything else on each monitor's form is kept.

`Form.fields` iterates `self.__dict__`, so the hybrid forms **subclass the classic ones and drop that one
attribute** rather than restating the EEG/MEG/iEEG/BOLD/SpatialAverage field definitions. Two methods
have to go with it: `MonitorForm.fill_from_post` dereferences `session_stored_simulator.model`, and
`fill_trait` writes the VOI indexes back onto the monitor. Both assume a classic `SimulatorAdapterModel`
with a single Model, which the Hybrid Simulator does not have.

The projection monitors' forms select `projection`, `sensors` and `region_mapping` as **datatype GIDs**,
so nothing about them has to work in this phase: the configuration is stored, not run. What it takes to
run them is listed under *Obligations this phase hands to Phase 6*.

### Where the configuration lives

`HybridSimulatorAdapterModel` gains:

* `simulation_length` — label, doc and default borrowed from the classic `Simulator` trait, the way `dt`
  already borrows from `Integrator.dt`;
* `monitors = List(of=MonitorViewModel, default=(TemporalAverageViewModel(),))`.

`__init__` must instantiate a fresh monitor, as `SimulatorAdapterModel.__init__` does. A trait default is
one shared instance across every session — the trap Phase 3 already documented for Model and Integrator.

Still nothing in the database: this lives in the session with the rest of the configuration until Phase 6.

### The wizard chain

Projections → Monitors → one parameters step per selected Monitor, in order → Simulation length.

* the Projections step's `next_button_enabled=False` is the switch this phase turns on;
* a BOLD monitor takes the extra hrf Equation step after its parameters, as the classic Cockpit does;
* the monitor ordering mirrors `MonitorsWizardHandler.get_current_and_next_monitor_form`, but the handler
  itself is not reused: it is bound to `SimulatorWizzardURLs` and to `SimulatorAdapterModel`, and the
  hybrid cockpit has its own of both;
* **no draft/save pair.** Phase 3 needed one because a Subnetwork's configuration could be abandoned by
  selecting a different Subnetwork mid-edit. The monitors are global and the chain is linear, so each
  step commits, exactly as the classic Cockpit does;
* the length step closes the phase. Its Next stays disabled until Phase 6 adds the launch.

**Open, to be decided while implementing:** `HYBRID_WIZARD_STEPS`. Phase 4 deliberately kept
`SET_PROJECTIONS_URL` out of that list, because a stack rebuild walks it and would regenerate the whole
`NetworkSet` on every step-back. But `hybridPreviousStep` slices `(0, undefined)` — the entire list — for
any url it cannot find, so appending the new steps while leaving Projections out makes a rebuild from the
length step render everything anyway. This needs a deliberate answer rather than an append.

### Service

* `set_monitors(hybrid_simulator, ui_names)` — UI names to view model instances, mirroring
  `MonitorsWizardHandler.set_monitors_list_on_simulator`;
* `output_layout(subnetworks)` — merged or concatenated, the resulting shape, and the Subnetworks whose
  VOI counts disagree;
* `validate_monitors(monitors, simulation_length, dt)` — a period longer than the simulation records
  nothing at all (BOLD defaults to 2000 ms), and a period below `dt` is not recordable;
* `build_hybrid_simulator(network_set, monitors, simulation_length)` — the configured `hybrid.Simulator`,
  monitors deep copied.

`build_hybrid_simulator` belongs here rather than in Phase 6, for the reason `build_network_set` belonged
in Phase 4: it is what makes *"configuration reaches the Hybrid Simulator correctly"* testable without
launching anything. Phase 6 then only has to run it and persist the result.

### Tests

* configuration reaches the Hybrid Simulator correctly — `simulation_length`, the monitor list and the
  `NetworkSet` all arrive on the built Simulator;
* monitors are configured correctly — each monitor's `dt` and `istep` after construction, and editing a
  stored monitor afterwards leaving the built one untouched (the deep copy);
* simulation length is respected — the sample count a `TemporalAverage` of a known period produces over
  a known length;
* invalid configuration produces useful validation messages — period above the length, period below `dt`;
* the layout report — equal VOI counts give merged and the connectome shape; one differing Subnetwork
  gives concatenated and is named;
* controller — the chain urls in order, two selected monitors producing two parameters steps, the BOLD
  equation step, the length round-trip, and the existing refusal to proceed over unsaved dynamics;
* render check — the whole chain, including a projection monitor with its GID fields and BOLD with its
  Equation step.

### Obligations this phase hands to Phase 6

Offering every classic monitor is cheap here and expensive there. Recorded now so the cost is not
discovered later:

1. **`voi = slice(None)` breaks the projection monitors.** `Projection.sample` allocates
   `numpy.zeros((gain.shape[0], len(self.voi)))`, and `len()` of a slice raises `TypeError`.
2. **`config_for_sim` is never called.** The hybrid Simulator calls only `_config_dt`, `_config_stock`
   and `record`, so a Projection monitor has no gain matrix and no `rmap`, and `SpatialAverage` has no
   spatial mask. Either a shim exposing `connectivity`, `surface`, `model`, `integrator` and
   `number_of_nodes` is handed to `config_for_sim`, or the framework computes and assigns those itself.
3. **Concatenated mode misaligns them silently.** A gain matrix is `(n_sensors, n_regions)` in connectome
   order, and `SpatialAverage`'s cortical/hemisphere masks are indexed the same way, so in concatenated
   mode they weight the wrong nodes rather than failing. Both are meaningful only in merged mode.
4. **GIDs have to become datatypes.** `projection`, `sensors` and `region_mapping` are stored as GIDs and
   must be loaded before the monitor can be configured.
5. Unrelated, noticed in passing: merged-mode `NetworkSet.observe` shapes its result with
   `subnets[0].model.number_of_modes`, while every subnetwork's observation has already been summed to a
   single mode — so a multi-mode first Subnetwork duplicates its output across the mode axis.

### Checkpoint

Build the Simulator from a saved configuration and inspect it before Phase 6 launches anything: `nets` is
the Phase 4 `NetworkSet`, `monitors` are the configured ones with `dt` and `istep` set from the shared
`dt`, and `simulation_length` is what the closing step stored.

---

## Phase 5 – Implementation Summary

Implemented as specified above, with the deviations recorded at the end of this section.

### Where the configuration lives

`HybridSimulatorAdapterModel` gained `monitors` (a `List(of=MonitorViewModel)` defaulting to a single
Temporal average) and `simulation_length`, whose label, doc and default are the classic `Simulator`
trait's, the way `dt` already borrows from `Integrator.dt`. Its `__init__` re-instantiates the default
Monitor: a trait default is one shared instance across every session, and the parameter forms edit the
Monitor in place. Still nothing in the database - this lives in the session until Phase 6.

### The Monitor forms

`tvb/adapters/forms/hybrid_monitor_forms.py` holds a `HybridMonitorForm` per classic Monitor form,
each inheriting its classic sibling so that not one field definition is restated.

`MonitorForm` adds exactly three methods on top of `Form`, and all three exist to carry
`variables_of_interest`. All three are taken back down to `Form`:

| method | what it assumed |
|---|---|
| `fill_from_post` | resolves the posted names against `session_stored_simulator.model` |
| `fill_trait` | writes the resolved indices onto the Monitor |
| `fill_from_trait` | reads them back into the field |

The field itself is deleted in `__init__`, which is what keeps it off the page: `Form.fields` yields
what is on the instance.

**`fill_trait` had to be overridden, not merely left alone.** With nothing resolved it writes
`numpy.array([])`, whose dtype is `float64`, and `Monitor.variables_of_interest` is an `int` typed
`NArray` that refuses it outright. The Monitor's own `variables_of_interest` is therefore left at
`None`, which `Monitor._config_vois` reads as 'all of them' - the same thing the Hybrid Simulator
imposes with `voi = slice(None)`.

**The base ordering matters.** BOLD and Spatial average override `fill_trait` / `fill_from_trait` to
carry their own fields, so the hybrid form lists the classic sibling **first** and `HybridMonitorForm`
second: `class HybridBoldMonitorForm(BoldMonitorForm, HybridMonitorForm)`. The MRO then runs BOLD's own
method, whose `super()` reaches this phase's override instead of `MonitorForm`'s. Listing the hybrid
base first would have skipped BOLD's own behaviour entirely.

Spatial average needed one more thing. Its classic `fill_from_trait` prunes the default-mask choices out
of `session_stored_simulator.connectivity`, so the hybrid form takes a `connectivity_gid` and does that
pruning against the Connectivity the Hybrid Simulator was given, dropping the surface-only choice
outright.

### The wizard chain

Projections → Monitors → one parameters step per Monitor → Simulation summary.

`_monitor_chain` builds the ordered `(url, monitor, is_equation)` list once, and every step is rendered
by position in it: the previous url is the step before, the next is the step after, and running off the
end is the summary. A Raw Monitor contributes no step - it records every integration step and documents
its sampling period as ignored, which is the rule the classic Cockpit applies through `first_monitor`. A
BOLD Monitor contributes a second one for its haemodynamic response Equation.

Each step's action url carries the Monitor class as a path segment
(`/burst/hybrid/set_monitor_params/EEGViewModel`), which is how the exposed method receives it - the
shape the classic Cockpit already uses.

### A step entered by a POST cannot move on by posting to itself

The Projections step is reached by posting to its own url, from the closing step of the Subnetwork
configuration. Giving it a POST branch that moved on therefore broke entering it at all, which its own
tests caught immediately. It instead names where its Next posts, through the `next_form_action_url` that
the closing step of the Subnetwork configuration already had.

That turned the template's second button into a general rule rather than one step's special case: an
ordinary step's Next now posts to `next_form_action_url` when one is set and to its own action
otherwise, and the extra button is rendered only for the step that genuinely needs two (Save
Configuration *and* Next).

### Where a sampling period is refused

On the step of the Monitor it belongs to, not on the summary. The summary would otherwise have to answer
with a step already on screen, and the client would append a second form carrying an id it already has.
Every non-Raw Monitor is posted through on the way to the summary, so nothing escapes the check.

### `HYBRID_WIZARD_STEPS`, the question Phase 5 left open

**Left unchanged.** Phase 4 kept `SET_PROJECTIONS_URL` off that list so a stack rebuild would not
regenerate the whole `NetworkSet` on every step back. The Monitor steps cannot go on it either: their
urls depend on what was selected, so there are no fixed ones to list. A url that is not on the list
rebuilds up to the Subnetwork dynamics, which is where every later step can be reached again, and that
is the behaviour the new steps now document rather than change.

### Tests

**151 hybrid tests** - 66 service, 79 controller, 6 render checks. The suites this work touches pass
unchanged: the classic Simulator Cockpit and the simulator adapters (44), and the view model, forms and
serialization suites (29).

```bash
python -m pytest tvb/tests/framework/core/services/hybrid_simulator_service_test.py \
                 tvb/tests/framework/interfaces/web/controllers/hybrid_simulator_controller_test.py \
                 tvb/tests/framework/interfaces/web/controllers/hybrid_render_check_test.py
```

The service tests assert that the configuration reaches the Simulator, that a Monitor's `dt` and `istep`
come from the shared `dt`, that editing a stored Monitor afterwards leaves the built one alone, both
refusals and Raw's exemption from them, and both output layouts. The controller tests assert the chain
urls and legends, that Raw gets no step, that BOLD gets its Equation step, that no step offers the
variables of interest, and that a period which records nothing hands its own step back.

**The Phase 5 checkpoint is a test**: `test_the_stored_configuration_builds_and_runs_a_hybrid_simulator`
walks the wizard, builds the `NetworkSet` and the `Simulator` out of what the session holds, runs it, and
asserts the sample count and that every Connectivity region is recorded at its own position. Its expected
sample count is *derived* from the shared `dt` rather than assumed: the default `dt` is `0.01220703125`,
which is not a round fraction of a 1 ms sampling period.

The render checks render the whole chain, a projection Monitor with its datatype fields and a BOLD
Monitor with its Equation step included, and assert that `variables_of_interest` appears on none of them.

### Deviations from the specification above

1. **The simulation length is not a step of its own.** It sits on the Monitors step, which now asks what
   to record and for how long at once. The classic Cockpit keeps it for last only because it shares that
   step with the Launch button; there is nothing to launch yet, and a sampling period means little away
   from the length it is sampling. It also gives the period refusals something to check against before
   any Monitor is configured.
2. **A Simulation summary closes the phase**, which the specification did not ask for. Something has to
   be the terminus - a step whose Next is disabled until Phase 6 - and a form is a poor terminus because
   its value would never be submitted. The summary is where the output-layout report belongs anyway, next
   to what each Monitor samples, and it is where Phase 6's simulation name and Launch button will go.

### Checked, and left to Phase 6

All nine Monitors **build** without complaint - `Simulator(...)` and `configure()` were run against each
one, and each came back with its `dt` and `istep` set. The obligations listed above are run-time ones:
they are what happens when a projection Monitor's `sample` is first called, not when it is configured.
Phase 5 therefore offers all nine honestly, and Phase 6 has the list of what to make work.

---

## Phase 6 – Launch one Hybrid simulation

**Status: Done** — see the implementation summary below.

Construct:

```text
Connectivity
    ↓
Subnetworks
    ↓
Projections
    ↓
NetworkSet
    ↓
Hybrid Simulator
```

Launch a single simulation using the `tvb_library` Hybrid Simulator API.

Persist the operation/results through the normal TVB framework mechanisms where possible.

### Scope

| question | decision |
|---|---|
| how a launch is recorded | a **BurstConfiguration**, with both histories filtered by the algorithm behind it |
| EEG / MEG / iEEG / Spatial average | **configured through a shim**, and refused unless the output is connectome ordered |
| initial conditions | **nothing exposed** — the library's random draw from each Model's `state_variable_range` |
| the variable axis labels | **by position** — `Variable 1`, `Variable 2`, … |
| branching, continuing, PSE, stimuli | out of scope |

### What this phase does not have to build

Three things were checked before specifying, and each removes work the plan would otherwise have carried:

* **The configuration already persists.** `h5.store_view_model` round-trips a whole
  `HybridSimulatorAdapterModel` — for a two-Subnetwork configuration it writes eleven files, one per
  nested view model (the Subnetworks, their dynamics, Models, Integrators and Monitors) — and
  `load_view_model` reads it back intact. No framework change is needed to store what the wizard holds.
* **Output packaging is one argument.** `Monitor.create_time_series(connectivity=...)` returns a
  `TimeSeriesRegion` and `create_time_series(connectivity=None)` a plain `TimeSeries`. Both have a
  registered Index and H5 class.
* **The Simulator is already built.** Phase 5's `build_network_set` and `build_hybrid_simulator` produce
  a configured `tvb.simulator.hybrid.Simulator`, and its checkpoint test already runs one.

### The adapter

`tvb/adapters/simulator/hybrid_simulator_adapter.py`, holding `HybridSimulatorAdapter(ABCAdapter)`.

Registered by adding `"hybrid_simulator_adapter"` to `ALL_SIMULATORS`, which is what gives it an
Algorithm row, plus `HYBRID_SIMULATOR_MODULE` / `HYBRID_SIMULATOR_CLASS` on `IntrospectionRegistry` and
in `tvb/config/__init__.py`, the way the classic one is named.

`ABCAdapter` requires five methods:

| method | what it does here |
|---|---|
| `get_form_class` | a small `HybridSimulatorAdapterForm` over the Connectivity, as the classic adapter's form is |
| `get_output` | `[TimeSeriesIndex]` — **no** `SimulationHistoryIndex`, since branching is not offered and `SimulationHistory.populate_from` reads a classic Simulator |
| `configure` | the Phase 5 service: `build_network_set`, then `build_hybrid_simulator` |
| `get_required_memory_size` / `get_required_disk_size` | estimated here, see below |
| `launch` | run, then write one TimeSeries per Monitor |

**The size estimates cannot be delegated.** The classic adapter asks the Simulator for
`memory_requirement()` and `storage_requirement()`; the hybrid Simulator has neither. They are estimated
from the recorded shape instead — samples × variables × nodes × 8 bytes per Monitor — and the memory one
has to be honest about something the classic path does not do: `hybrid.Simulator.run()` returns its
results as whole arrays rather than yielding them per step, so the adapter holds every Monitor's output
in memory and writes it after the run, where the classic adapter streams it slice by slice into H5.

### Output packaging

| layout | datatype | node axis |
|---|---|---|
| connectome ordered | `TimeSeriesRegion`, keyed to the Connectivity | one column per region, in region order |
| concatenated | plain `TimeSeries` | Subnetwork after Subnetwork |

Phase 5 already computes which one applies, and reports it on the Simulation summary; this phase reads
the same `output_layout` and passes the Connectivity or `None` accordingly. A `TimeSeriesRegion` over a
concatenated array would claim a region ordering the data does not have, which is the one thing worth
refusing to write.

**The variable axis is labelled by position** — `Variable 1`, `Variable 2`, … Connectome ordered output
requires the Subnetworks to agree on the *number* of variables they watch, never on their names:
JansenRit watches `y0, y1` where Generic2dOscillator watches `V, W`. Labelling by position claims
nothing that is untrue of any Subnetwork, and the per-Subnetwork names stay recoverable from the stored
configuration. `start_time` is zero, as nothing is being continued.

### Projection and Spatial average Monitors

Phase 5 offers all nine Monitors and recorded why four of them cannot yet run. The cause is single:
**the hybrid Simulator never calls `config_for_sim`**. It calls `_config_dt`, `_config_stock` and
`record` only, so `Projection._state`, `_period_in_steps` and the gain matrix, and `SpatialAverage`'s
`spatial_mean`, are never created — every one of them is assigned inside `config_for_sim`.

A shim object standing in for a classic Simulator closes all four at once. It has to answer for
`connectivity`, `surface` (`None`), `model`, `integrator` (anything carrying `dt`) and
`number_of_nodes`, and its Model must expose as many `variables_of_interest` as the merged output has
variables.

**It must be applied before the Simulator is constructed.** `Monitor._config_vois` sets
`voi = arange(len(model.variables_of_interest))` and `Projection.config_for_sim` then sizes
`_state = zeros((gain.shape[0], len(self.voi)))`. `Simulator.__init__` overwrites `voi` with
`slice(None)` afterwards, which is harmless *in that order*: `len(voi)` was read while it was still a
concrete array, and `slice(None)` then selects exactly those same rows at sample time. In the other
order `len()` is applied to a slice and raises.

**Refused unless the output is connectome ordered.** A gain matrix is `(n_sensors, n_regions)` and a
cortical or hemisphere mask is indexed the same way, so both are meaningful only when column *i* is
region *i*. In concatenated mode they would weight the wrong nodes and return a number rather than
fail, so the launch is refused with a message naming the Subnetworks whose variable counts differ and
the Monitor that needs them to agree. Phase 5 reports that layout; this is where it becomes a rule.

### The launch, and the burst

`HybridSimulatorController.launch_simulation` mirrors the classic one: store the `BurstConfiguration`,
then run `prepare_operation` and `launch_operation` on a thread, answering `{'id': ...}` or
`{'error': ...}`. The Simulation summary step gains the simulation name and the Launch button, which is
what its disabled `Next` is holding open today.

**Telling the two histories apart.** `dao.get_bursts_for_project` filters by project alone, so a
BurstConfiguration created here would otherwise appear in the classic Simulator's history, where opening
it would try to read a classic configuration. A burst's simulation operation names its algorithm, so one
DAO query joining `BurstConfiguration` to `Operation` can separate them — asked once rather than per
burst.

The rule is deliberately asymmetric: **the hybrid history shows the bursts whose operation used the
hybrid adapter, and the classic history shows everything else.** A burst is stored before its operation
exists — `store_burst` runs first so the client gets an id, and the thread fills `fk_simulation` in
afterwards — so there is a real moment where the algorithm is unknown. Sending those to the classic
history leaves every existing burst exactly where it is today and never hides one from both.

### The client

`hybridLaunchSimulation()`, mirroring `launchNewBurst`: post the summary step, then reload the hybrid
history through the `load_hybrid_history` endpoint that already exists, and point the results tree at
the new burst. `displayHybridResultsTree` passes the placeholder burst id `"0"` today, which is the one
thing on that panel that has to change.

### Tests

Use a small deterministic simulation.

Verify:

* simulation launches;
* operation completes;
* expected monitor output is produced;
* output node ordering remains consistent with the original Connectivity;
* failures are reported through the normal TVB operation mechanism.

Compare at least one GUI-created simulation against the equivalent Python hybrid demo.

How each is made concrete:

* **node ordering** is asserted rather than assumed: give two Subnetworks distinguishable Model
  parameters and non-contiguous node indices, then check that the columns carrying each Subnetwork's
  signature are exactly its own `node_indices`. Connectome ordering is the claim a `TimeSeriesRegion`
  makes, so it is the claim worth testing;
* **failures** are checked by launching a configuration that cannot run — a projection Monitor over
  Subnetworks that disagree on their variable counts is one the phase itself creates — and asserting the
  operation ends in `ERROR` carrying the message;
* **the demo comparison** builds the `simulate_hybrid_getting_started` configuration through the service
  and compares the result against the same network hand-built with the library API. The notebook's own
  weights are random, so both sides are built over the same Connectivity instead.

### Out of scope

* branching and continuing a simulation, and the `SimulationHistory` that would carry them;
* loading a past hybrid simulation back into the cockpit — the history template already has the link,
  and it has nothing behind it;
* PSE, stimuli and the numba backend, which are the follow-up features below.

---

## Phase 6 – Implementation Summary

Implemented as specified above, with the deviations recorded at the end of this section.

### The adapter

`tvb/adapters/simulator/hybrid_simulator_adapter.py` holds `HybridSimulatorAdapter`. It is registered by
adding `"hybrid_simulator_adapter"` to `ALL_SIMULATORS`, which is what gives it an Algorithm row, and
named by `HYBRID_SIMULATOR_MODULE` / `HYBRID_SIMULATOR_CLASS` in `tvb/config/__init__.py` and on
`IntrospectionRegistry`, the way the classic one is named.

`get_form_class` returns `HybridConnectivityFragment` rather than a new class: the cockpit's own
Connectivity step already is that form, with the same field and the same filters.

`configure` does the whole translation — the layout, the Monitors, the `NetworkSet`, the `Simulator` —
and wraps a `HybridSubnetworkException` into a `LaunchException`, so a configuration that cannot run is
reported the way every other adapter reports one.

The size estimates could not be delegated: the hybrid Simulator has no `memory_requirement`. They are
computed from what will actually be recorded, samples × variables × nodes × 8 bytes per Monitor, and the
memory one is genuinely the larger of the two because `Simulator.run` returns whole arrays instead of
yielding them — every Monitor's output is held until the run is over, where the classic adapter streams
its own into H5 as it goes.

### Output packaging

The layout Phase 5 reports is the one this phase writes:

| layout | passed to `create_time_series` | result |
|---|---|---|
| connectome ordered | the Connectivity | `TimeSeriesRegion`, one column per region |
| concatenated | `None` | plain `TimeSeries` |

Nothing else was needed, because **every Monitor that changes the node axis already says what it
produces**: EEG a `TimeSeriesEEG` over its Sensors, Global average a plain `TimeSeries`, Spatial average
one or the other depending on its mask. Withholding the Connectivity is therefore enough to stop a
Monitor claiming a region ordering the data does not have.

The variable axis is labelled by position — `Variable 1`, `Variable 2`, … Connectome ordered output
requires the Subnetworks to agree on the *number* of variables they watch, never on their names, so
that is the only labelling true of all of them.

### The Monitors that needed a classic Simulator

`ClassicSimulatorShim` carries the five things `config_for_sim` reads: the Connectivity, a `None`
surface, a Model exposing one variable of interest per variable the merged output holds, an Integrator
carrying `dt`, and the node count. That is the whole fix for EEG, MEG, iEEG and Spatial average.

It is applied **inside** `build_hybrid_simulator`, between the deep copy of the Monitors and the
construction of the Simulator, and the order is the point: `Projection.config_for_sim` sizes its
recording buffer from `len(self.voi)` while `voi` is still a concrete index array, and
`Simulator.__init__` replaces `voi` with `slice(None)` immediately after — which selects exactly those
same rows. In the other order `len()` is applied to a slice and raises.

`validate_monitors_for_layout` refuses these Monitors over concatenated output, naming the Monitor and
each Subnetwork's variable count.

### The launch, and the two histories

`HybridSimulatorController.launch_simulation` stores a `BurstConfiguration` and hands off on a thread to
`SimulatorService.async_launch_and_prepare_simulation` — **reused unchanged**. Nothing in it is
particular to the classic Simulator: it takes the Algorithm and the view model it is given.

`dao.get_bursts_for_project_by_algorithm` separates the two cockpits' histories in one query, by the
Algorithm of the Operation behind each burst. The rule is asymmetric on purpose: the Hybrid history
shows the bursts whose Operation used the Hybrid adapter, and the classic history shows **everything
else**. A burst is stored before its Operation exists — the id has to be answered to the browser while
the Operation is created on the launching thread — so there is a real moment where the Algorithm is
unknown, and sending those to the classic history leaves every burst that predates this phase exactly
where it already was. `BurstService.get_available_bursts` falls back to the unfiltered list when the
database holds no Hybrid Algorithm at all, which is what keeps an older database working.

### Deviations from the specification above

1. **The refusals are checked on the stored Monitors, before any of them is converted.** The
   specification put `config_for_sim` and the layout check together; in practice a Monitor that cannot
   run should not first have its Sensors, Projection matrix and Region mapping loaded out of the
   database. What is refused depends only on the Monitor's class and on the layout, both of which the
   stored configuration already carries, so the check happens there and the loading happens after it.
2. **The launch step carries only the simulation name.** The specification described the summary
   gaining "the simulation name field and the Launch button"; the simulation length is already on the
   Monitors step from Phase 5, so nothing else had to move.

### Found in passing, not fixed

`OperationService.initiate_prelaunch` ends with
`if operation.fk_operation_group and 'SimulatorAdapter' in operation.algorithm.classname`. That
substring also matches **`HybridSimulatorAdapter`**, so a Hybrid operation belonging to an operation
group would launch the classic metric operation after it. Nothing creates one today — operation groups
come from PSE, which the Hybrid Simulator does not offer — so it cannot fire, but it is a trap waiting
for the PSE follow-up and should be made an exact comparison before that lands.

### Tests

The Phase 6 list, and what each one became:

* **simulation launches / operation completes** — `test_happy_flow_launch` runs a real simulation
  through `TestFactory.launch_synchronously`, which asserts the operation finished, and checks the
  stored `TimeSeriesRegion`'s four dimensions;
* **expected monitor output** — the same test, plus the concatenated case asserting that a plain
  `TimeSeries` is written and **no** `TimeSeriesRegion` is;
* **node ordering** — `test_connectome_ordering_is_preserved` gives the two Subnetworks different Model
  parameters and the second one a scattered, non-contiguous set of nodes, then asserts that the columns
  carrying its dynamics are exactly its own `node_indices`. A concatenating output would place them
  contiguously and fail;
* **failures reported through the operation mechanism** — `test_a_refused_configuration_fails_the_operation`
  drives a refused configuration through `OperationService` and asserts the operation ends in `ERROR`
  carrying the reason;
* **compared against the demos** — `test_the_generated_simulation_matches_one_written_by_hand` builds
  the same two-Subnetwork network twice: once through the service, once by hand with `IntraProjection`,
  `InterProjection` and blocks sliced straight out of the Connectivity, the way the `simulate_hybrid_*`
  notebooks write them. Both are run from the same zero initial conditions and the outputs must be
  identical. **This is the comparison Phase 4 deferred**, and it could not be made until a simulation
  could actually be run.

Also covered: the launch stores a named burst and hands off; an unusable simulation name is refused; and
the two histories do not show each other's bursts, including the burst whose Operation does not exist
yet.

### Checkpoint

A Hybrid simulation can be configured, launched, and its results found in the project. The whole chain
from the specification is exercised end to end by the adapter tests, and the wizard's own path to it by
the controller ones.

---

# Follow-up features

**Status: Not started.**

These should be implemented only after the basic workflow is stable.

## Projection editing

Allow users to:

* enable/disable individual projections;
* change `source_cvar`;
* change `target_cvar`;
* change coupling function;
* change scaling;
* optionally provide custom weights/lengths.

Support configurations where some Subnetworks are intentionally not connected.

## Subnetwork parameter editing

Allow editing Model and Integrator parameters for each Subnetwork.

Investigate whether existing Simulator Cockpit forms and Setup Region Model components can be reused.

## Stimuli

Allow Stimuli to be configured for individual Subnetworks and their appropriate coupling/state variables.

Follow the hybrid stimulus demo as a reference.

## Parameter Space Exploration

Add Hybrid PSE only after single Hybrid simulations are stable.

Potential sweep targets include:

* Model parameters;
* Projection parameters;
* Subnetwork parameters;
* global simulation parameters.

Reuse the existing TVB PSE infrastructure where possible.

---

# Implementation principles

* Implement and review one phase at a time.
* Do not implement future phases while working on the current phase.
* Prefer reusing existing TVB framework components.
* Keep scientific Hybrid Simulator behaviour inside `tvb_library`.
* Keep UI/configuration/persistence logic inside `tvb_framework`.
* Add focused tests with every phase.
* Preserve the classic Simulator Cockpit behaviour.
* Do not commit or push changes; all changes must be reviewed manually first.
