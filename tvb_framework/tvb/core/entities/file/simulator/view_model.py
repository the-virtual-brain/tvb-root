# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#

from tvb.basic.neotraits.api import Attr, List, EnumAttr, TupleEnum
from tvb.core.entities.file.simulator.simulation_history_h5 import SimulationHistory
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.datatypes.patterns import SpatioTemporalPattern
from tvb.datatypes.projections import ProjectionSurfaceEEG, ProjectionSurfaceMEG, ProjectionSurfaceSEEG
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import SensorsEEG, SensorsMEG, SensorsInternal
from tvb.datatypes.surfaces import CorticalSurface
from tvb.simulator.integrators import HeunDeterministic, Integrator, IntegratorStochastic, HeunStochastic, \
    EulerDeterministic, EulerStochastic, RungeKutta4thOrderDeterministic, Identity, VODE, VODEStochastic, Dopri5, \
    Dopri5Stochastic, Dop853, Dop853Stochastic
from tvb.simulator.monitors import Monitor, EEG, MEG, iEEG, Raw, SubSample, SpatialAverage, GlobalAverage, \
    TemporalAverage, Projection, Bold, BoldRegionROI
from tvb.simulator.noise import Noise, Additive, Multiplicative
from tvb.simulator.simulator import Simulator


class NoiseViewModel(ViewModel, Noise):
    @property
    def linked_has_traits(self):
        return Noise


class AdditiveNoiseViewModel(NoiseViewModel, Additive):
    @property
    def linked_has_traits(self):
        return Additive


class MultiplicativeNoiseViewModel(NoiseViewModel, Multiplicative):
    @property
    def linked_has_traits(self):
        return Multiplicative

    def __init__(self):
        super(MultiplicativeNoiseViewModel, self).__init__()


class IntegratorViewModel(ViewModel, Integrator):
    @property
    def linked_has_traits(self):
        return Integrator


class IntegratorStochasticViewModel(IntegratorViewModel, IntegratorStochastic):
    noise = Attr(
        field_type=NoiseViewModel,
        label=IntegratorStochastic.noise.label,
        default=AdditiveNoiseViewModel(),
        required=IntegratorStochastic.noise.required,
        doc=IntegratorStochastic.noise.doc
    )

    @property
    def linked_has_traits(self):
        return IntegratorStochastic

    def __init__(self):
        super(IntegratorStochasticViewModel, self).__init__()
        self.noise = type(self.noise)()


class HeunDeterministicViewModel(IntegratorViewModel, HeunDeterministic):
    @property
    def linked_has_traits(self):
        return HeunDeterministic


class HeunStochasticViewModel(IntegratorStochasticViewModel, HeunStochastic):
    @property
    def linked_has_traits(self):
        return HeunStochastic


class EulerDeterministicViewModel(IntegratorViewModel, EulerDeterministic):
    @property
    def linked_has_traits(self):
        return EulerDeterministic


class EulerStochasticViewModel(IntegratorStochasticViewModel, EulerStochastic):
    @property
    def linked_has_traits(self):
        return EulerStochastic


class RungeKutta4thOrderDeterministicViewModel(IntegratorViewModel, RungeKutta4thOrderDeterministic):
    @property
    def linked_has_traits(self):
        return RungeKutta4thOrderDeterministic


class IdentityViewModel(IntegratorViewModel, Identity):
    @property
    def linked_has_traits(self):
        return Identity


class VODEViewModel(IntegratorViewModel, VODE):
    @property
    def linked_has_traits(self):
        return VODE


class VODEStochasticViewModel(IntegratorStochasticViewModel, VODEStochastic):
    @property
    def linked_has_traits(self):
        return VODEStochastic


class Dopri5ViewModel(IntegratorViewModel, Dopri5):
    @property
    def linked_has_traits(self):
        return Dopri5


class Dopri5StochasticViewModel(IntegratorStochasticViewModel, Dopri5Stochastic):
    @property
    def linked_has_traits(self):
        return Dopri5Stochastic


class Dop853ViewModel(IntegratorViewModel, Dop853):
    @property
    def linked_has_traits(self):
        return Dop853


class Dop853StochasticViewModel(IntegratorStochasticViewModel, Dop853Stochastic):
    @property
    def linked_has_traits(self):
        return Dop853Stochastic


class IntegratorViewModelsEnum(TupleEnum):
    HEUN = (HeunDeterministicViewModel, "Heun")
    STOCHASTIC_HEUN = (HeunStochasticViewModel, "Stochastic Heun")
    EULER = (EulerDeterministicViewModel, "Euler")
    EULER_MARUYAMA = (EulerStochasticViewModel, "Euler-Maruyama")
    RUNGE_KUTTA = (RungeKutta4thOrderDeterministicViewModel, "Runge-Kutta 4Th Order")
    DIFFERENCE_EQUATION = (IdentityViewModel, "Difference equation")
    VARIABLE_ORDER_ADAMS = (VODEViewModel, "Variable-Order Adams (BDF)")
    STOCHASTIC_VARIABLE_ODER_ADAMS = (VODEStochasticViewModel, "Stochastic variable-order Adams (BDF)")
    DOPRI_5 = (Dopri5ViewModel, "Dormand-Prince, order (4, 5)")
    DOPRI_5_STOCHASTIC = (Dopri5StochasticViewModel, "Stochastic Dormand-Prince, order (4, 5)")
    DOP_853 = (Dop853ViewModel, "Dormand-Prince, order 8 (5, 3)")
    DOP_853_STOCHASTIC = (Dop853StochasticViewModel, "Stochastic Dormand-Prince, order 8 (5, 3)")


class MonitorViewModel(ViewModel, Monitor):
    @property
    def linked_has_traits(self):
        return Monitor


class RawViewModel(MonitorViewModel, Raw):
    @property
    def linked_has_traits(self):
        return Raw

    def __str__(self):
        clsname = self.__class__.__name__
        return '%s(period=%f)' % (clsname, self.period)


class SubSampleViewModel(MonitorViewModel, SubSample):
    @property
    def linked_has_traits(self):
        return SubSample


class SpatialAverageViewModel(MonitorViewModel, SpatialAverage):
    @property
    def linked_has_traits(self):
        return SpatialAverage


class GlobalAverageViewModel(MonitorViewModel, GlobalAverage):
    @property
    def linked_has_traits(self):
        return GlobalAverage


class TemporalAverageViewModel(MonitorViewModel, TemporalAverage):
    @property
    def linked_has_traits(self):
        return TemporalAverage


class ProjectionViewModel(MonitorViewModel, Projection):
    region_mapping = DataTypeGidAttr(
        linked_datatype=RegionMapping,
        required=True,
        label=Projection.region_mapping.label,
        doc=Projection.region_mapping.doc
    )

    @property
    def linked_has_traits(self):
        return Projection


class EEGViewModel(ProjectionViewModel, EEG):
    projection = DataTypeGidAttr(
        linked_datatype=ProjectionSurfaceEEG,
        label=EEG.projection.label,
        doc=EEG.projection.doc
    )

    sensors = DataTypeGidAttr(
        linked_datatype=SensorsEEG,
        label=EEG.sensors.label,
        doc=EEG.sensors.doc
    )

    @property
    def linked_has_traits(self):
        return EEG


class MEGViewModel(ProjectionViewModel, MEG):
    projection = DataTypeGidAttr(
        linked_datatype=ProjectionSurfaceMEG,
        label=MEG.projection.label,
        doc=MEG.projection.doc
    )

    sensors = DataTypeGidAttr(
        linked_datatype=SensorsMEG,
        label=MEG.sensors.label,
        doc=MEG.sensors.doc
    )

    @property
    def linked_has_traits(self):
        return MEG


class iEEGViewModel(ProjectionViewModel, iEEG):
    projection = DataTypeGidAttr(
        linked_datatype=ProjectionSurfaceSEEG,
        label=iEEG.projection.label,
        doc=iEEG.projection.doc
    )

    sensors = DataTypeGidAttr(
        linked_datatype=SensorsInternal,
        label=iEEG.sensors.label,
        doc=iEEG.sensors.doc
    )

    @property
    def linked_has_traits(self):
        return iEEG


class BoldViewModel(MonitorViewModel, Bold):
    @property
    def linked_has_traits(self):
        return Bold

    def __init__(self):
        super(BoldViewModel, self).__init__()


class BoldRegionROIViewModel(BoldViewModel, BoldRegionROI):
    @property
    def linked_has_traits(self):
        return BoldRegionROI


class CortexViewModel(ViewModel, Cortex):

    @property
    def linked_has_traits(self):
        return Cortex

    surface_gid = DataTypeGidAttr(
        linked_datatype=CorticalSurface,
        label=Simulator.surface.label,
        default=Simulator.surface.default,
        required=Simulator.surface.required,
        doc=Simulator.surface.doc
    )

    local_connectivity = DataTypeGidAttr(
        linked_datatype=LocalConnectivity,
        required=Cortex.local_connectivity.required,
        label=Cortex.local_connectivity.label,
        doc=Cortex.local_connectivity.doc
    )

    region_mapping_data = DataTypeGidAttr(
        linked_datatype=RegionMapping,
        label=Cortex.region_mapping_data.label,
        doc=Cortex.region_mapping_data.doc
    )


class SimulatorAdapterModel(ViewModel, Simulator):

    @property
    def linked_has_traits(self):
        return Simulator

    connectivity = DataTypeGidAttr(
        linked_datatype=Connectivity,
        required=Simulator.connectivity.required,
        label=Simulator.connectivity.label,
        doc=Simulator.connectivity.doc
    )

    surface = Attr(
        field_type=CortexViewModel,
        label=Simulator.surface.label,
        default=Simulator.surface.default,
        required=Simulator.surface.required,
        doc=Simulator.surface.doc
    )

    stimulus = DataTypeGidAttr(
        linked_datatype=SpatioTemporalPattern,
        label=Simulator.stimulus.label,
        default=Simulator.stimulus.default,
        required=Simulator.stimulus.required,
        doc=Simulator.stimulus.doc
    )

    history_gid = DataTypeGidAttr(
        linked_datatype=SimulationHistory,
        required=False
    )

    integrator = EnumAttr(
        field_type=IntegratorViewModelsEnum,
        label=Simulator.integrator.label,
        default=IntegratorViewModelsEnum.HEUN.instance,
        required=Simulator.integrator.required,
        doc=Simulator.integrator.doc
    )

    monitors = List(
        of=MonitorViewModel,
        label=Simulator.monitors.label,
        default=(TemporalAverageViewModel(),),
        doc=Simulator.monitors.doc
    )

    def __init__(self):
        super(SimulatorAdapterModel, self).__init__()
        self.coupling = type(self.coupling)()
        self.model = type(self.model)()
        self.integrator = type(self.integrator)()
        self.monitors = (type(self.monitors[0])(),)

    @property
    def first_monitor(self):
        if isinstance(self.monitors[0], RawViewModel):
            if len(self.monitors) > 1:
                return self.monitors[1]
            else:
                return None
        return self.monitors[0]

    def determine_indexes_for_chosen_vars_of_interest(self):
        all_variables = self.model.__class__.variables_of_interest.element_choices
        chosen_variables = self.model.variables_of_interest
        indexes = self.get_variables_of_interest_indexes(all_variables, chosen_variables)
        return indexes

    @staticmethod
    def get_variables_of_interest_indexes(all_variables, chosen_variables):
        variables_of_interest_indexes = {}

        if not isinstance(chosen_variables, (list, tuple)):
            chosen_variables = [chosen_variables]

        for variable in chosen_variables:
            variables_of_interest_indexes[variable] = all_variables.index(variable)
        return variables_of_interest_indexes
