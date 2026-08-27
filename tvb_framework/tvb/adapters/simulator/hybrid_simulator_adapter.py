import numpy
from scipy.sparse import csr_matrix

from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.core.adapters.abcadapter import ABCAdapter, ABCAdapterForm
from tvb.core.adapters.exceptions import InvalidParameterException, LaunchException
from tvb.core.entities.file.simulator.hybrid_view_model import HybridSimulatorAdapterModel
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.datatypes.time_series import TimeSeriesRegion
from tvb.simulator.hybrid import NetworkSet, Simulator, Subnetwork
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.hybrid.projection_utils import create_inter_projection, create_intra_projection


class HybridSimulatorAdapterForm(ABCAdapterForm):
    def __init__(self):
        super().__init__()
        self.connectivity = TraitDataTypeSelectField(
            HybridSimulatorAdapterModel.connectivity, name="connectivity")

    def fill_from_trait(self, trait):
        self.connectivity.data = trait.connectivity.hex if trait.connectivity else None

    def fill_trait(self, trait):
        trait.connectivity = self.connectivity.value

    @staticmethod
    def get_view_model():
        return HybridSimulatorAdapterModel

    @staticmethod
    def get_required_datatype():
        return ConnectivityIndex

    @staticmethod
    def get_input_name():
        return "connectivity"

    @staticmethod
    def get_filters():
        return None


class HybridSimulatorAdapter(ABCAdapter):
    _ui_name = "Hybrid Simulation Core"

    def __init__(self):
        super().__init__()
        self.algorithm = None
        self.connectivity = None

    def get_form_class(self):
        return HybridSimulatorAdapterForm

    def get_output(self):
        return [TimeSeriesRegionIndex]

    def get_adapter_fragments(self, view_model):
        return {}

    def _to_has_traits(self, value):
        return self.view_model_to_has_traits(value) if hasattr(value, "linked_has_traits") else value

    def build_subnetworks(self, view_model):
        subnetworks = []
        for subnet_view in view_model.subnetworks:
            model = self._to_has_traits(subnet_view.model)
            model.variables_of_interest = (subnet_view.observable,)
            integrator = self._to_has_traits(subnet_view.integrator)
            subnetworks.append(Subnetwork(
                name=subnet_view.stable_id,
                model=model,
                scheme=integrator,
                nnodes=len(subnet_view.node_indices),
                node_indices=numpy.asarray(subnet_view.node_indices, dtype=numpy.int_),
            ))
        return subnetworks

    @staticmethod
    def build_projections(view_model, subnetworks):
        library_by_id = {subnet.name: subnet for subnet in subnetworks}
        inter_projections = []
        for projection_view in view_model.projections:
            source = library_by_id[projection_view.source_id]
            target = library_by_id[projection_view.target_id]
            source_cvar = source.model.state_variables[int(source.model.cvar[0])]
            target_cvar = target.model.state_variables[int(target.model.cvar[0])]
            kwargs = dict(
                source_cvar=source_cvar,
                target_cvar=target_cvar,
                weights=csr_matrix(projection_view.weights),
                lengths=csr_matrix(projection_view.tract_lengths),
                cv=3.0,
                dt=source.scheme.dt,
                coupling=Linear(),
            )
            if projection_view.kind == "intra":
                projection = create_intra_projection(subnet=source, **kwargs)
                source.projections = list(source.projections) + [projection]
            else:
                projection = create_inter_projection(source_subnet=source, target_subnet=target, **kwargs)
                inter_projections.append(projection)
        return inter_projections

    def build_simulator(self, view_model):
        subnetworks = self.build_subnetworks(view_model)
        projections = self.build_projections(view_model, subnetworks)
        monitors = [self._to_has_traits(monitor) for monitor in view_model.monitors]
        return Simulator(
            nets=NetworkSet(subnets=subnetworks, projections=projections),
            monitors=monitors,
            simulation_length=view_model.simulation_length,
            backend="python",
        )

    def configure(self, view_model):
        self.connectivity = self.load_traited_by_gid(view_model.connectivity)
        errors = view_model.validate(self.connectivity.number_of_regions)
        if errors:
            raise InvalidParameterException(" ".join(errors))
        try:
            self.algorithm = self.build_simulator(view_model)
            self.algorithm.configure()
        except (TypeError, ValueError, AssertionError) as exc:
            raise LaunchException(f"Invalid hybrid simulation configuration: {exc}") from exc

    def get_required_memory_size(self, view_model):
        node_count = sum(len(subnet.node_indices) for subnet in view_model.subnetworks)
        state_count = sum(subnet.model.nvar * len(subnet.node_indices) for subnet in view_model.subnetworks)
        steps = max(1, int(numpy.ceil(view_model.simulation_length / view_model.subnetworks[0].integrator.dt)))
        return int((state_count + node_count) * steps * 8)

    def get_required_disk_size(self, view_model):
        node_count = sum(len(subnet.node_indices) for subnet in view_model.subnetworks)
        samples = sum(numpy.ceil(view_model.simulation_length / monitor.period) for monitor in view_model.monitors)
        return int(samples * node_count * 8 / 1024)

    def get_execution_time_approximation(self, view_model):
        steps = view_model.simulation_length / view_model.subnetworks[0].integrator.dt
        nodes = sum(len(subnet.node_indices) for subnet in view_model.subnetworks)
        return max(1, int(6.57e-6 * steps * nodes))

    def launch(self, view_model):
        if self.algorithm is None:
            self.configure(view_model)
        results = self.algorithm.run()
        indexes = []
        observable_map = ", ".join(
            f"{subnet.name}: {subnet.observable}" for subnet in view_model.subnetworks)
        observable_map = observable_map[:180]
        for monitor, (times, data) in zip(self.algorithm.monitors, results):
            if data.ndim != 4 or data.shape[1] != 1 or data.shape[2] != self.connectivity.number_of_regions:
                raise InvalidParameterException(
                    f"Unexpected hybrid monitor result shape {data.shape}; expected (*, 1, "
                    f"{self.connectivity.number_of_regions}, *).")
            sample_period = float(times[1] - times[0]) if len(times) > 1 else float(monitor.period)
            time_series = TimeSeriesRegion(
                title=f"Hybrid {type(monitor).__name__} ({observable_map})",
                data=data,
                time=times,
                start_time=float(times[0]) if len(times) else 0.0,
                sample_period=sample_period,
                connectivity=self.connectivity,
                labels_dimensions={"State Variable": ["Hybrid observable"]},
            )
            indexes.append(self.store_complete(time_series))
        return indexes
