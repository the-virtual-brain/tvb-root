import importlib
import uuid
import os

from tvb.basic.logger.builder import get_logger
from tvb.simulator.integrators import IntegratorStochastic
from tvb.simulator.monitors import Bold
from tvb.simulator.noise import Multiplicative
from tvb.simulator.simulator import Simulator

from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.hdf5_storage_manager import HDF5StorageManager
from tvb.core.entities.file.simulator.configurations_h5 import CouplingH5, ModelH5, EquationH5, NoiseH5, IntegratorH5, \
    MonitorH5, SimulatorH5
from tvb.core.entities.model.model_operation import Operation
from tvb.core.entities.storage import dao
from tvb.core.services.operation_service import OperationService
from tvb.interfaces.neocom._h5loader import DirLoader
from tvb.interfaces.web.controllers import common


def get_configuration_path_and_type(dir_loader, configuration_gid):
    configuration_filename = dir_loader._locate(configuration_gid)
    configuration_path = os.path.join(dir_loader.base_dir, configuration_filename)
    storage_manager = HDF5StorageManager(dir_loader.base_dir, configuration_filename)
    configuration_type = storage_manager.get_metadata().get('type')
    package, cls_name = configuration_type.rsplit('.', 1)
    module = importlib.import_module(package)
    cls = getattr(module, cls_name)

    return configuration_path, cls


class SimulatorService(object):
    dir_loader = None

    def __init__(self):
        self.operation_service = OperationService()
        self.logger = get_logger(self.__class__.__name__)
        self.file_helper = FilesHelper()

    def serialize_simulator(self, simulator, simulator_id):
        # TODO: handle operation - simulator H5 relation
        project = common.get_current_project()
        user = common.get_logged_user()

        partial_operation = self._prepare_operation(project.id, user.id, simulator_id)
        storage_path = self.file_helper.get_project_folder(project, partial_operation)

        self.dir_loader = DirLoader(storage_path)

        coupling_gid = self._serialize_coupling(simulator.coupling)
        model_gid = self._serialize_model(simulator.model)
        integrator_gid = self._serialize_integrator(simulator.integrator)
        monitor_gid = self._serialize_monitor(simulator.monitors[0])

        simulator_gid = uuid.uuid4()
        simulator_path = self.dir_loader.path_for_has_traits(type(simulator), simulator_gid)

        with SimulatorH5(simulator_path) as simulator_h5:
            simulator_h5.gid.store(simulator_gid)
            simulator_h5.connectivity.store(uuid.uuid4())
            simulator_h5.conduction_speed.store(simulator.conduction_speed)
            simulator_h5.surface.store(uuid.uuid4())
            simulator_h5.stimulus.store(uuid.uuid4())
            simulator_h5.initial_conditions.store(simulator.initial_conditions)
            simulator_h5.simulation_length.store(simulator.simulation_length)

            simulator_h5.coupling.store(coupling_gid)
            simulator_h5.model.store(model_gid)
            simulator_h5.integrator.store(integrator_gid)
            simulator_h5.monitors.store([monitor_gid.hex])

        # partial_operation.parameters = json.dumps({'simulator_h5': simulator_path})
        # dao.store_entity(partial_operation)

        return simulator_gid

    def deserialize_simulator(self, simulator_gid, simulator_id):
        project = common.get_current_project()
        user = common.get_logged_user()

        partial_operation = self._prepare_operation(project.id, user.id, simulator_id)
        storage_path = self.file_helper.get_project_folder(project, partial_operation)

        self.dir_loader = DirLoader(storage_path)

        simulator_in_path = self.dir_loader.path_for_has_traits(Simulator, simulator_gid)
        simulator_in = Simulator()

        with SimulatorH5(simulator_in_path) as simulator_in_h5:
            simulator_in.conduction_speed = simulator_in_h5.conduction_speed.load()
            simulator_in.simulation_length = simulator_in_h5.simulation_length.load()
            simulator_in.initial_conditions = simulator_in_h5.initial_conditions.load()

            # ---------- Load GIDs of all references ---------
            coupling_in_gid = simulator_in_h5.coupling.load()
            model_in_gid = simulator_in_h5.model.load()
            monitors_gid_list = simulator_in_h5.monitors.load()
            monitor_in_gid = uuid.UUID(monitors_gid_list[0])
            integrator_in_gid = simulator_in_h5.integrator.load()

        coupling_in = self._deserialize_coupling(coupling_in_gid)
        model_in = self._deserialize_model(model_in_gid)
        integrator_in = self._deserialize_integrator(integrator_in_gid)
        monitor_in = self._deserialize_monitor(monitor_in_gid)

        simulator_in.coupling = coupling_in
        simulator_in.model = model_in
        simulator_in.integrator = integrator_in
        simulator_in.monitors = [monitor_in]

        return simulator_in

    def _prepare_operation(self, project_id, user_id, simulator_id):
        # sim_algo = FlowService().get_algorithm_by_identifier(simulator_id)

        operation = Operation(user_id, project_id, simulator_id, '')
        operation = dao.store_entity(operation)

        return operation

    def _serialize_coupling(self, coupling):
        coupling_gid = uuid.uuid4()
        coupling_path = self.dir_loader.path_for_has_traits(type(coupling), coupling_gid)

        with CouplingH5(coupling_path, coupling) as coupling_h5:
            coupling_h5.store(coupling)

        return coupling_gid

    def _serialize_model(self, model):
        model_gid = uuid.uuid4()
        model_path = self.dir_loader.path_for_has_traits(type(model), model_gid)

        with ModelH5(model_path, model) as model_h5:
            model_h5.store(model)

        return model_gid

    def _serialize_equation(self, equation):
        equation_gid = uuid.uuid4()
        equation_path = self.dir_loader.path_for_has_traits(type(equation), equation_gid)

        with EquationH5(equation_path, equation) as equation_h5:
            equation_h5.equation.store(equation.equation)
            equation_h5.parameters.store(equation.parameters)

        return equation_gid

    def _serialize_noise(self, noise, equation_gid=None):
        noise_gid = uuid.uuid4()
        noise_path = self.dir_loader.path_for_has_traits(type(noise), noise_gid)

        with NoiseH5(noise_path, noise) as noise_h5:
            noise_h5.gid.store(noise_gid)
            noise_h5.ntau.store(noise.ntau)
            noise_h5.noise_seed.store(noise.noise_seed)
            noise_h5.nsig.store(noise.nsig)

            if isinstance(noise, Multiplicative):
                noise_h5.b.store(equation_gid)

        return noise_gid

    def _serialize_integrator(self, integrator, noise_gid=None):
        integrator_gid = uuid.uuid4()
        integrator_path = self.dir_loader.path_for_has_traits(type(integrator), integrator_gid)

        with IntegratorH5(integrator_path, integrator) as integrator_h5:
            integrator_h5.gid.store(integrator_gid)
            integrator_h5.dt.store(integrator.dt)

            if issubclass(integrator, IntegratorStochastic):
                integrator_h5.noise.store(noise_gid)

        return integrator_gid

    def _serialize_monitor(self, monitor):
        monitor_gid = uuid.uuid4()
        monitor_path = self.dir_loader.path_for_has_traits(type(monitor), monitor_gid)

        monitor_h5 = MonitorH5(monitor_path, monitor)
        monitor_h5.store(monitor, store_references=False)
        monitor_h5.gid.store(monitor_gid)

        if isinstance(monitor, Bold) or issubclass(type(monitor), Bold):
            hrf_kernel = monitor.hrf_kernel
            hrf_kernel_gid = self._serialize_equation(hrf_kernel)
            monitor_h5.hrf_kernel.store(hrf_kernel_gid)
            monitor_h5.hrf_length.store(monitor.hrf_length)

        monitor_h5.close()

        return monitor_gid

    def _deserialize_coupling(self, coupling_in_gid):
        coupling_in_path, coupling_class = get_configuration_path_and_type(self.dir_loader, coupling_in_gid)
        coupling_in = coupling_class()

        with CouplingH5(coupling_in_path, coupling_in) as coupling_in_h5:
            coupling_in_h5.load_into(coupling_in)

        return coupling_in

    def _deserialize_model(self, model_in_gid):
        model_in_path, model_class = get_configuration_path_and_type(self.dir_loader, model_in_gid)
        model_in = model_class()

        with ModelH5(model_in_path, model_in) as model_in_h5:
            model_in_h5.load_into(model_in)

        return model_in

    def _deserialize_equation(self, equation_in_gid):
        equation_in_path, equation_class = get_configuration_path_and_type(self.dir_loader, equation_in_gid)
        equation_in = equation_class()

        with EquationH5(equation_in_path, equation_in) as equation_in_h5:
            equation_in.parameters = equation_in_h5.parameters.load()

        return equation_in

    def _deserialize_noise(self, noise_in_gid):
        noise_in_path, noise_class = get_configuration_path_and_type(self.dir_loader, noise_in_gid)
        noise_in = noise_class()

        with NoiseH5(noise_in_path, noise_in) as noise_in_h5:
            noise_in_h5.load_into(noise_in)

            if isinstance(noise_in, Multiplicative):
                equation_in_gid = noise_in_h5.b.load()
                equation_in = self._deserialize_equation(equation_in_gid)
                noise_in.b = equation_in

        return noise_in

    def _deserialize_integrator(self, integrator_in_gid):
        integrator_in_path, integrator_class = get_configuration_path_and_type(self.dir_loader, integrator_in_gid)
        integrator_in = integrator_class()

        with IntegratorH5(integrator_in_path, integrator_in) as integrator_in_h5:
            integrator_in_h5.load_into(integrator_in)
            if issubclass(integrator_in, IntegratorStochastic):
                noise_in_gid = integrator_in_h5.noise.load()
                noise_in = self._deserialize_noise(noise_in_gid)
                integrator_in.noise = noise_in

        return integrator_in

    def _deserialize_monitor(self, monitor_in_gid):
        monitor_in_path, monitor_class = get_configuration_path_and_type(self.dir_loader, monitor_in_gid)
        monitor_in = monitor_class()

        with MonitorH5(monitor_in_path, monitor_in) as monitor_in_h5:
            monitor_in_h5.load_into(monitor_in)

        if isinstance(monitor_in, Bold) or issubclass(type(monitor_in), Bold):
            hrf_kernel_in_gid = monitor_in_h5.hrf_kernel.load()
            hrf_kernel_in = self._deserialize_equation(hrf_kernel_in_gid)
            monitor_in.hrf_kernel = hrf_kernel_in
            # monitor_in.hrf_length = monitor_in_h5.hrf_length.load()

        return monitor_in
