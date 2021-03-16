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

"""
Find a Connectivity in current project (by Subject) and later run a simulation on it.

__main__ will contain the code.
"""
from time import sleep
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.simulator.coupling import Scaling
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.services.algorithm_service import AlgorithmService
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.command.lab import *


# Before starting this, we need to have TVB web interface launched at least once
# (to have a default project, user and connectivity)
def run_simulation():
    log = get_logger(__name__)

    # This ID of a project needs to exists in DB, and it can be taken from the WebInterface:
    project = dao.get_project_by_id(1)

    # Find a structural Connectivity and load it in memory
    connectivity_index = dao.get_generic_entity(ConnectivityIndex, DataTypeMetaData.DEFAULT_SUBJECT, "subject")[0]
    connectivity = h5.load_from_index(connectivity_index)

    # Load the SimulatorAdapter algorithm from DB
    cached_simulator_algorithm = AlgorithmService().get_algorithm_by_module_and_class(
        IntrospectionRegistry.SIMULATOR_MODULE,
        IntrospectionRegistry.SIMULATOR_CLASS)

    # Instantiate a SimulatorService and launch the configured simulation
    simulator_model = SimulatorAdapterModel()
    simulator_model.connectivity = connectivity.gid
    simulator_model.simulation_length = 100
    simulator_model.coupling = Scaling()

    simulator_service = SimulatorService()
    burst = BurstConfiguration(project.id, name="Simulation")
    dao.store_entity(burst)
    launched_operation = simulator_service.async_launch_and_prepare_simulation(burst,
                                                                               project.administrator, project,
                                                                               cached_simulator_algorithm,
                                                                               simulator_model)

    # wait for the operation to finish
    while not launched_operation.has_finished:
        sleep(5)
        launched_operation = dao.get_operation_by_id(launched_operation.id)

    if launched_operation.status == STATUS_FINISHED:
        ts = dao.get_generic_entity(TimeSeriesRegionIndex, launched_operation.id, "fk_from_operation")[0]
        log.info("TimeSeries result is: %s " % ts)
    else:
        log.warning("Operation ended with problems [%s]: [%s]" % (launched_operation.status,
                                                                  launched_operation.additional_info))


if __name__ == "__main__":
    run_simulation()
