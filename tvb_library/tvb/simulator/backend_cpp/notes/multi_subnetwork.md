I mean the Python objects can describe that topology, but the native C++ execution path cannot execute it yet.

The original demo has:

```text
cortex JansenRit  --InterProjection-->  thalamus ReducedSetFitzHughNagumo
```

That requires native C++ support for:

1. Multiple subnetworks in one simulation loop.
2. Separate state buffers per subnetwork.
3. Separate history/ring buffers per source subnetwork.
4. Inter-projection routing from source subnetwork to target subnetwork.
5. Mode mapping, because `JansenRit` has 1 mode but `ReducedSetFitzHughNagumo` has 3 modes.
6. `ReducedSetFitzHughNagumo` codegen/runtime support, including its mode-indexed expressions like `xi_{m}` and derived matrices like `Aik`, `Bik`, `Cik`.

Right now, `CppHybridBackend` native execution is narrower:

```text
one subnetwork only
single mode only
HeunDeterministic only
no inter-subnetwork projections
limited intra-projection support
```

The hard checks are in the backend/runtime. For example, `CompiledCppNetwork.run()` rejects more than one initial state / subnetwork, and `codegen.py` rejects `spec.inter_projections`. The runtime also assumes a single generated model with one `StateBuffer`.

So for the getting-started demo:

- `JansenRit` alone: C++ can run it.
- `ReducedSetFitzHughNagumo`: not currently handled by C++ codegen/runtime because it is multi-mode and uses mode-indexed dfun expressions.
- `JansenRit -> ReducedSetFitzHughNagumo`: not currently handled because it needs multi-subnetwork inter-projection execution.

The next real backend milestone is implementing inter-projection support: one state/history buffer per subnetwork, projection arrays with source/target subnet ids, mode maps, and coupling accumulation into the target subnetwork before each target integration step.