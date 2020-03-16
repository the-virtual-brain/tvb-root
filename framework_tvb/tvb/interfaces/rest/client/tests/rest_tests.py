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
import os
import tempfile
from uuid import UUID
import tvb_data.sensors as demo_data
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapterModel
from tvb.adapters.uploaders.sensors_importer import SensorsImporterModel
from tvb.adapters.visualizers.sensors import SensorsViewerModel
from tvb.core.neocom import h5
from tvb.interfaces.rest.client.operation.operation_api import OperationApi
from tvb.interfaces.rest.client.simulator.simulation_api import SimulationApi
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestRestService(TransactionalTestCase):
    # TODO: integrate with Jenkins build
    # Following tests are intended as examples and work when the server is started

    def transactional_setup_method(self):
        self.base_url = "http://127.0.0.1:9090/api"
        self.temp_folder = tempfile.gettempdir()
        self.test_user = TestFactory.create_user('Rest_User')
        self.test_project = TestFactory.create_project(self.test_user, 'Rest_Project')

    def test_launch_operation(self):
        operation_api = OperationApi(self.base_url)

        project_gid = "651a7f9c-3159-11ea-ada0-3464a92a26e5"
        algorithm_module = "tvb.adapters.uploaders.sensors_importer"
        algorithm_classname = "SensorsImporter"
        view_model = SensorsViewerModel()

        meg_file_path = os.path.join(os.path.dirname(demo_data.__file__), 'meg_151.txt.bz2')
        meg_sensors_index = TestFactory.import_sensors(self.test_user, self.test_project, meg_file_path,
                                                       SensorsImporterModel.OPTIONS['MEG Sensors'])
        meg_sensors = h5.load_from_index(meg_sensors_index)
        view_model.sensors = meg_sensors.gid

        operation_api.launch_operation(project_gid, algorithm_classname, view_model,
                                       self.temp_folder)

    def test_fire_simulation(self):
        simulation_api = SimulationApi(self.base_url)
        session_stored_simulator = SimulatorAdapterModel()

        TestFactory.import_zip_connectivity(self.test_user, self.test_project)
        connectivity = TestFactory.get_entity(self.test_project, ConnectivityIndex)
        session_stored_simulator.connectivity = UUID(connectivity.gid)

        simulation_api.fire_simulation(self.test_project.gid, session_stored_simulator, None, self.temp_folder)
