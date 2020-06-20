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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
A convenience module for the command interface

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
from datetime import datetime
from time import sleep
from tvb.adapters.uploaders.zip_connectivity_importer import ZIPConnectivityImporter, ZIPConnectivityImporterModel
from tvb.basic.profile import TvbProfile
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.storage import dao
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.model.model_operation import STATUS_FINISHED
from tvb.core.neocom import h5
from tvb.core.services.algorithm_service import AlgorithmService
from tvb.core.services.operation_service import OperationService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.simulator_service import SimulatorService
from tvb.core.services.user_service import UserService
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.simulator.simulator import Simulator

TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)
LOG = get_logger(__name__)


def list_projects():
    fmt = "%24s %5s"
    print(fmt % ('name', 'id'))
    for p in dao.get_all_projects():
        print(fmt % (p.name, p.id))


def list_datatypes(project_id):
    fmt = "%24s %16s %5s %32s %12s"
    print(fmt % ('type', 'tag', 'id', 'gid', 'date'))
    for dt in dao.get_datatypes_in_project(project_id):
        print(fmt % (dt.type, dt.user_tag_1, dt.id, dt.gid, dt.create_date))


def datatype_details(dt_id):
    dt = dao.get_datatype_by_id(dt_id)
    print(ProjectService().get_datatype_details(dt.gid))


def load_dt(dt_id):
    dt = dao.get_datatype_by_id(dt_id)
    dt_idx = dao.get_generic_entity(dt.module + '.' + dt.type, dt_id)[0]
    dt_ht = h5.load_from_index(dt_idx)
    return dt_ht


def new_project(name):
    usr = UserService.get_administrators()[0]
    proj = ProjectService().store_project(usr, True, None, name=name, description=name, users=[usr])
    return proj


def import_conn_zip(project_id, zip_path):
    project = dao.get_project_by_id(project_id)

    importer = ABCAdapter.build_adapter_from_class(ZIPConnectivityImporter)
    view_model = ZIPConnectivityImporterModel()
    view_model.uploaded = zip_path

    OperationService().fire_operation(importer, project.administrator, project_id, view_model=view_model)


def fire_simulation(project_id, simulator):
    project = dao.get_project_by_id(project_id)
    assert isinstance(simulator, Simulator)
    # Load the SimulatorAdapter algorithm from DB
    cached_simulator_algorithm = AlgorithmService().get_algorithm_by_module_and_class(IntrospectionRegistry.SIMULATOR_MODULE,
                                                                                      IntrospectionRegistry.SIMULATOR_CLASS)

    # Instantiate a SimulatorService and launch the configured simulation
    simulator_service = SimulatorService()
    burst = BurstConfiguration(project.id)
    burst.name = "Sim " + str(datetime.now())
    burst.start_time = datetime.now()
    dao.store_entity(burst)

    launched_operation = simulator_service.async_launch_and_prepare_simulation(burst, project.administrator, project,
                                                                               cached_simulator_algorithm, simulator,
                                                                               None)
    LOG.info("Operation launched ....")
    return launched_operation


def wait_to_finish(operation):
    # Wait for the operation to finish
    while not operation.has_finished:
        sleep(5)
        operation = dao.get_operation_by_id(operation.id)

    if operation.status == STATUS_FINISHED:
        LOG.info("Operation finished successfully")
    else:
        LOG.warning("Operation ended with problems [%s]: [%s]" % (operation.status, operation.additional_info))
