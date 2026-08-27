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

## Phase 2 – Create Subnetworks

Allow the user to divide Connectivity regions into Subnetworks.

Preferred first approach:

* reuse concepts/components from **Setup Region Model** where practical;
* allow creation and naming of Subnetworks;
* allow assigning selected Connectivity nodes to a Subnetwork;
* every Connectivity node must belong to exactly one Subnetwork.

Start with a simple interaction before considering drag-and-drop.

Example:

```text
Connectivity

Subnetwork A
  - Region 1
  - Region 2
  - Region 3

Subnetwork B
  - Region 4
  - Region 5
```

Optionally reuse the existing 3D Connectivity display to visualize Subnetwork membership with different colors.

### Tests

* create/rename/remove a Subnetwork;
* assign nodes;
* move nodes between Subnetworks;
* prevent invalid assignments;
* verify all nodes are assigned before continuing.

### Checkpoint

Evaluate the interaction with both small and large Connectivities.

Decide whether the existing region-selection UI is sufficient or whether a dedicated interaction is needed.

---

## Phase 3 – Configure each Subnetwork

For each Subnetwork allow selection of:

* Model;
* Integrator.

Start with default parameters supplied by the selected Model and Integrator.

Do not expose all model parameters initially.

All Integrators must currently use a compatible/common `dt` required by the Hybrid Simulator.

### Tests

* Model selection is stored per Subnetwork;
* Integrator selection is stored per Subnetwork;
* defaults are correctly created;
* changing one Subnetwork does not affect another;
* invalid combinations are rejected.

### Checkpoint

Verify that the UI configuration can be translated cleanly into `tvb.simulator.hybrid.Subnetwork` objects.

Only after this works consider exposing Model/Integrator parameter editing.

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
