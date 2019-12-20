import requests
import tempfile
import os
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.model.simulator.simulator import SimulatorIndex
from tvb.core.services.simulator_serializer import SimulatorSerializer
from tvb.interfaces.rest.client.main_api import MainApi


class SimulationApi(MainApi):

    def fire_simulation(self, project_gid, session_stored_simulator, burst_config, temp_folder):
        simulator_index = SimulatorIndex()
        temp_name = tempfile.mkdtemp(dir=TvbProfile.current.TVB_TEMP_FOLDER)
        destination_folder = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, temp_name)
        simulation_state_gid = None

        SimulatorSerializer().serialize_simulator(session_stored_simulator, simulator_index.gid,
                                                  simulation_state_gid, destination_folder)
        zip_folder_path = temp_folder + '/SimulationData.zip'
        FilesHelper().zip_folder(zip_folder_path, destination_folder)

        # TODO: HANDLE BURST_CONFIG SENDING
        file_obj = open(zip_folder_path, 'rb')
        response = requests.post(self.server_url + '/simulation/' + project_gid, files={"archive": ("SimulationData.zip", file_obj)})