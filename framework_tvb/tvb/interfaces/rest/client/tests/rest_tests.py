import os
import tempfile
from uuid import UUID
import tvb_data
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapterModel
from tvb.adapters.uploaders.sensors_importer import SensorsImporterModel
from tvb.adapters.visualizers.sensors import SensorsViewerModel
from tvb.core.neocom import h5
from tvb.interfaces.rest.client.operation.operation_api import OperationApi
from tvb.interfaces.rest.client.simulator.simulation_api import SimulationApi
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory
import tvb_data.sensors as demo_data


class TestRestService(TransactionalTestCase):
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

        operation_api.launch_operation(project_gid, algorithm_module, algorithm_classname, view_model,
                                       self.temp_folder)

    def test_fire_simulation(self):
        simulation_api = SimulationApi(self.base_url)

        session_stored_simulator = SimulatorAdapterModel()

        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_76.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")
        connectivity = TestFactory.get_entity(self.test_project, ConnectivityIndex)
        session_stored_simulator.connectivity = UUID(connectivity.gid)

        simulation_api.fire_simulation(self.test_project.gid, session_stored_simulator, None, self.temp_folder)
