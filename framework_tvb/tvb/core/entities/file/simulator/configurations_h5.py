from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import NArray, Attr
from tvb.simulator.integrators import IntegratorStochastic
from tvb.simulator.monitors import SpatialAverage, Projection, Bold
from tvb.simulator.noise import Multiplicative
from tvb.simulator.simulator import Simulator

from tvb.core.neotraits.h5 import H5File, Scalar, Reference, DataSet, Json


def get_full_class_name(class_entity):
    return class_entity.__module__ + '.' + class_entity.__name__


class EquationH5(H5File):

    def __init__(self, path, equation):
        super(EquationH5, self).__init__(path, None)
        equation_class = type(equation)

        self.generic_attributes.type = get_full_class_name(equation_class)
        self.equation = Scalar(equation_class.equation, self)
        self.parameters = Json(equation_class.parameters, self)


class NoiseH5(H5File):

    def __init__(self, path, noise):
        super(NoiseH5, self).__init__(path, None)
        noise_class = type(noise)

        self.generic_attributes.type = get_full_class_name(noise_class)
        self.ntau = Scalar(noise_class.ntau, self)
        self.noise_seed = Scalar(noise_class.noise_seed, self)
        self.nsig = DataSet(noise_class.nsig, self)

        if issubclass(noise_class, Multiplicative):
            self.b = Reference(Multiplicative.b, self)


class IntegratorH5(H5File):

    def __init__(self, path, integrator):
        super(IntegratorH5, self).__init__(path, None)
        integrator_class = type(integrator)

        self.generic_attributes.type = get_full_class_name(integrator_class)
        self.dt = Scalar(integrator_class.dt, self)

        if issubclass(integrator_class, IntegratorStochastic):
            self.noise = Reference(integrator_class.noise, self)


class CouplingH5(H5File):
    logger = get_logger()

    def __init__(self, path, coupling):
        super(CouplingH5, self).__init__(path, None)
        coupling_class = type(coupling)
        self.generic_attributes.type = get_full_class_name(coupling_class)

        for decl_attr_name in coupling_class.declarative_attrs:
            decl_attr = getattr(coupling_class, decl_attr_name)
            if type(decl_attr) is NArray:
                setattr(self, decl_attr_name, DataSet(decl_attr, self))
            else:
                if type(decl_attr) is Attr:
                    setattr(self, decl_attr_name, Scalar(decl_attr, self))
                else:
                    self.logger.warning('Parameter %s of type %s from coupling %s should be handled in H5',
                                        decl_attr_name, type(decl_attr).__name__, coupling_class.__name__)


class ModelH5(H5File):
    logger = get_logger()

    def __init__(self, path, model):
        super(ModelH5, self).__init__(path, None)
        model_class = type(model)
        self.generic_attributes.type = get_full_class_name(model_class)

        self.state_variable_range = Json(model_class.state_variable_range, self)
        self.variables_of_interest = Json(model_class.variables_of_interest, self)

        for decl_attr_name in model_class.declarative_attrs:
            decl_attr = getattr(model_class, decl_attr_name)
            if type(decl_attr) is NArray:
                setattr(self, decl_attr_name, DataSet(decl_attr, self))
            else:
                self.logger.warning(
                    'Parameter %s of type %s from model %s should be handled in H5',
                    decl_attr_name, type(decl_attr).__name__, model_class.__name__)


class MonitorH5(H5File):

    def __init__(self, path, monitor):
        super(MonitorH5, self).__init__(path, None)
        monitor_class = type(monitor)
        self.generic_attributes.type = get_full_class_name(monitor_class)

        self.period = Scalar(monitor_class.period, self)
        self.variables_of_interest = DataSet(monitor_class.variables_of_interest, self)

        if isinstance(monitor, SpatialAverage):
            self.spatial_mask = DataSet(monitor.spatial_mask, self)
            self.default_mask = Scalar(monitor.default_mask, self)

        if issubclass(monitor_class, Projection):
            self.region_mapping = Reference(monitor_class.region_mapping, self)
            self.obnoise = Reference(monitor_class.obsnoise, self)
            self.projection = Reference(monitor_class.projection, self)
            self.sensors = Reference(monitor_class.sensors, self)

            if hasattr(monitor_class, 'reference'):
                self.reference = Scalar(monitor_class.reference, self)

            if hasattr(monitor_class, 'sigma'):
                self.sigma = Scalar(monitor_class.sigma, self)

        if isinstance(monitor, Bold) or issubclass(monitor_class, Bold):
            self.hrf_kernel = Reference(monitor_class.hrf_kernel, self)
            self.hrf_length = Scalar(monitor_class.hrf_length, self)


class SimulatorH5(H5File):

    def __init__(self, path):
        super(SimulatorH5, self).__init__(path, None)
        self.connectivity = Reference(Simulator.connectivity, self)
        self.conduction_speed = Scalar(Simulator.conduction_speed, self)
        self.coupling = Reference(Simulator.coupling, self)
        self.surface = Reference(Simulator.surface, self)
        self.stimulus = Reference(Simulator.stimulus, self)
        self.model = Reference(Simulator.model, self)
        self.integrator = Reference(Simulator.integrator, self)
        self.initial_conditions = DataSet(Simulator.initial_conditions, self)
        self.monitors = Json(Simulator.monitors, self)
        self.simulation_length = Scalar(Simulator.simulation_length, self)
