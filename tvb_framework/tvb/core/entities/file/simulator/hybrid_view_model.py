import uuid

import numpy

from tvb.basic.neotraits.api import Attr, Float, List, NArray
from tvb.core.entities.file.simulator.view_model import (
    EulerDeterministicViewModel,
    HeunDeterministicViewModel,
    IntegratorViewModel,
    MonitorViewModel,
    RawViewModel,
    TemporalAverageViewModel,
)
from tvb.core.neotraits.view_model import DataTypeGidAttr, Str, ViewModel
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models import Generic2dOscillator, Model


class HybridSubnetworkViewModel(ViewModel):
    stable_id = Str(label="Identifier")
    name = Str(label="Name")
    node_indices = NArray(dtype=numpy.int_)
    model = Attr(Model, default=Generic2dOscillator())
    integrator = Attr(IntegratorViewModel, default=HeunDeterministicViewModel())
    observable = Str(label="Observable")

    def __init__(self, **kwargs):
        kwargs.setdefault("stable_id", f"subnet_{uuid.uuid4().hex}")
        kwargs.setdefault("name", "Subnetwork")
        kwargs.setdefault("node_indices", numpy.array([], dtype=numpy.int_))
        kwargs.setdefault("model", Generic2dOscillator())
        kwargs.setdefault("integrator", HeunDeterministicViewModel())
        kwargs.setdefault("observable", kwargs["model"].variables_of_interest[0])
        super().__init__(**kwargs)


class HybridProjectionViewModel(ViewModel):
    kind = Str(label="Projection kind")
    source_id = Str(label="Source subnetwork")
    target_id = Str(label="Target subnetwork")
    weights = NArray()
    tract_lengths = NArray()


class HybridSimulatorAdapterModel(ViewModel):
    connectivity = DataTypeGidAttr(linked_datatype=Connectivity, label="Connectivity", required=False)
    subnetworks = List(of=HybridSubnetworkViewModel)
    projections = List(of=HybridProjectionViewModel)
    monitors = List(of=MonitorViewModel, default=(TemporalAverageViewModel(),))
    simulation_length = Float(default=1000.0, label="Simulation length (ms)")

    def __init__(self, **kwargs):
        kwargs.setdefault("subnetworks", [])
        kwargs.setdefault("projections", [])
        kwargs.setdefault("monitors", [TemporalAverageViewModel()])
        super().__init__(**kwargs)

    def validate(self, node_count=None):
        errors = []
        if self.connectivity is None:
            errors.append("Select a connectivity.")
        if len(self.subnetworks) < 2:
            errors.append("Create at least two subnetworks.")

        names = [subnet.name.strip() for subnet in self.subnetworks]
        stable_ids = [subnet.stable_id for subnet in self.subnetworks]
        if any(not name for name in names):
            errors.append("Every subnetwork must have a name.")
        if len(names) != len(set(names)):
            errors.append("Subnetwork names must be unique.")
        if len(stable_ids) != len(set(stable_ids)):
            errors.append("Subnetwork identifiers must be unique.")

        all_indices = [int(index) for subnet in self.subnetworks for index in subnet.node_indices]
        if len(all_indices) != len(set(all_indices)):
            errors.append("Each connectivity node must belong to exactly one subnetwork.")
        if node_count is not None:
            if sorted(all_indices) != list(range(node_count)):
                errors.append("Assign every connectivity node to exactly one subnetwork.")
        if any(len(subnet.node_indices) == 0 for subnet in self.subnetworks):
            errors.append("Subnetworks cannot be empty.")

        subnet_by_id = {subnet.stable_id: subnet for subnet in self.subnetworks}
        expected = {(subnet.stable_id, subnet.stable_id, "intra") for subnet in self.subnetworks}
        expected.update(
            (source.stable_id, target.stable_id, "inter")
            for source in self.subnetworks for target in self.subnetworks
            if source.stable_id != target.stable_id
        )
        actual = {(projection.source_id, projection.target_id, projection.kind)
                  for projection in self.projections}
        if actual != expected or len(self.projections) != len(expected):
            errors.append("Projection definitions do not match the configured subnetworks.")
        for projection in self.projections:
            source = subnet_by_id.get(projection.source_id)
            target = subnet_by_id.get(projection.target_id)
            if source is None or target is None:
                continue
            expected_shape = (len(target.node_indices), len(source.node_indices))
            label = f"Projection {source.name} -> {target.name}"
            if projection.weights.shape != expected_shape:
                errors.append(f"{label} has weights shape {projection.weights.shape}; expected {expected_shape}.")
            if projection.tract_lengths.shape != expected_shape:
                errors.append(
                    f"{label} has tract-length shape {projection.tract_lengths.shape}; expected {expected_shape}.")
            if not numpy.isfinite(projection.weights).all():
                errors.append(f"{label} weights must be finite.")
            if not numpy.isfinite(projection.tract_lengths).all() or numpy.any(projection.tract_lengths < 0):
                errors.append(f"{label} tract lengths must be finite and non-negative.")

        dts = []
        for subnet in self.subnetworks:
            if len(subnet.model.cvar) == 0:
                errors.append(f"Subnetwork {subnet.name} model has no coupling variable.")
            if subnet.observable not in subnet.model.variables_of_interest:
                errors.append(f"Subnetwork {subnet.name} has an invalid observable.")
            if not isinstance(subnet.integrator, (HeunDeterministicViewModel, EulerDeterministicViewModel)):
                errors.append(f"Subnetwork {subnet.name} uses an unsupported integrator.")
            dts.append(float(subnet.integrator.dt))
        if any(dt <= 0 for dt in dts):
            errors.append("Integrator time steps must be positive.")
        if dts and len(set(dts)) != 1:
            errors.append("All subnetworks must use the same integration time step.")

        if self.simulation_length <= 0:
            errors.append("Simulation length must be positive.")
        if not self.monitors:
            errors.append("Select at least one monitor.")
        for monitor in self.monitors:
            if not isinstance(monitor, (RawViewModel, TemporalAverageViewModel)):
                errors.append(f"Monitor {type(monitor).__name__} is not supported by the hybrid GUI.")
            if monitor.period <= 0 or monitor.period > self.simulation_length:
                errors.append("Monitor periods must be positive and no longer than the simulation.")
        return errors
