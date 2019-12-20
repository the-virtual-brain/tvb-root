import os
import tempfile
from uuid import UUID
import tvb_data
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapterModel
from tvb.interfaces.rest.client.simulator.simulation_api import SimulationApi
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestRestService(TransactionalTestCase):

    def transactional_setup_method(self):
        self.base_url = "http://127.0.0.1:9090/api"
        self.temp_folder = tempfile.gettempdir()
        self.test_user = TestFactory.create_user('Rest_User')
        self.test_project = TestFactory.create_project(self.test_user, 'Rest_Project')

    def test_fire_simulation(self):
        simulation_api = SimulationApi(self.base_url)

        session_stored_simulator = SimulatorAdapterModel()

        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_76.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")
        connectivity = TestFactory.get_entity(self.test_project, ConnectivityIndex)
        session_stored_simulator.connectivity = UUID(connectivity.gid)

        simulation_api.fire_simulation(self.test_project.gid, session_stored_simulator, None, self.temp_folder)

