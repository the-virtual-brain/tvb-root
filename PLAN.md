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

## Phase 0 – Understand the existing Simulator workflow

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

## Phase 4 – Generate Projections

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

## Phase 5 – Global Hybrid Simulator configuration

Add the remaining simulation-level configuration.

Initial scope:

* simulation length;
* Monitors;
* backend if appropriate;
* other required global Hybrid Simulator parameters.

Reuse existing Simulator Cockpit components where possible.

### Tests

* configuration reaches the Hybrid Simulator correctly;
* monitors are configured correctly;
* simulation length is respected;
* invalid configuration produces useful validation messages.

---

## Phase 6 – Launch one Hybrid simulation

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

### Tests

Use a small deterministic simulation.

Verify:

* simulation launches;
* operation completes;
* expected monitor output is produced;
* output node ordering remains consistent with the original Connectivity;
* failures are reported through the normal TVB operation mechanism.

Compare at least one GUI-created simulation against the equivalent Python hybrid demo.

---

# Follow-up features

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
