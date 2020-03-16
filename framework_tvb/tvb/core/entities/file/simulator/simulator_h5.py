# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#
import uuid
from tvb.basic.neotraits.api import Attr
from tvb.simulator.simulator import Simulator
from tvb.core.entities.file.simulator.configurations_h5 import SimulatorConfigurationH5
from tvb.core.neotraits.h5 import Reference, Scalar, Json, DataSet


class SimulatorH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(SimulatorH5, self).__init__(path)
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
        self.simulation_state = Reference(Attr(field_type=uuid.UUID), self, name='simulation_state')

    def store(self, datatype, scalars_only=False, store_references=False):
        # type: (Simulator, bool, bool) -> None
        self.gid.store(datatype.gid)
        self.connectivity.store(datatype.connectivity)
        self.conduction_speed.store(datatype.conduction_speed)
        self.initial_conditions.store(datatype.initial_conditions)
        self.simulation_length.store(datatype.simulation_length)

        integrator_gid = self.store_config_as_reference(datatype.integrator)
        self.integrator.store(integrator_gid)

        coupling_gid = self.store_config_as_reference(datatype.coupling)
        self.coupling.store(coupling_gid)

        model_gid = self.store_config_as_reference(datatype.model)
        self.model.store(model_gid)

        # TODO: handle multiple monitors
        monitor_gid = self.store_config_as_reference(datatype.monitors[0])
        self.monitors.store([monitor_gid.hex])

        if datatype.surface:
            cortex_gid = self.store_config_as_reference(datatype.surface)
            self.surface.store(cortex_gid)

        if datatype.stimulus:
            self.stimulus.store(datatype.stimulus)

        self.type.store(self.get_full_class_name(type(datatype)))

    def load_into(self, datatype):
        # type: (Simulator) -> None
        datatype.conduction_speed = self.conduction_speed.load()
        datatype.initial_conditions = self.initial_conditions.load()
        datatype.simulation_length = self.simulation_length.load()
        datatype.integrator = self.load_from_reference(self.integrator.load())
        datatype.coupling = self.load_from_reference(self.coupling.load())
        datatype.model = self.load_from_reference(self.model.load())
        # TODO: handle multiple monitors
        datatype.monitors = [self.load_from_reference(self.monitors.load()[0])]
        if self.surface.load():
            datatype.surface = self.load_from_reference(self.surface.load())
