from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.interfaces.web.controllers import common


class SimulatorContext(object):
    def __init__(self):
        self.session_stored_simulator = None
        self.is_simulator_copy = None
        self.is_simulator_load = None
        self.last_loaded_form_url = None
        self.burst_config = None
        self.is_branch = None
        self.project = None

    def init_session_stored_simulator(self):
        simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)

        if simulator is None:
            self.session_stored_simulator = SimulatorAdapterModel()
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)

    def get_current_project(self):
        self.project = common.get_current_project()

    @staticmethod
    def get_logged_user():
        return common.get_logged_user()

    def get_session_params(self):
        self.session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        self.is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        self.is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False

    def add_last_loaded_form_url_to_session(self, last_loaded_form_url):
        self.last_loaded_form_url = last_loaded_form_url
        common.add2session(common.KEY_LAST_LOADED_FORM_URL, self.last_loaded_form_url)

    def add_burst_config_to_session(self, burst_config=None):
        if burst_config is None:
            self.burst_config = BurstConfiguration(self.project.id)
            common.add2session(common.KEY_BURST_CONFIG, self.burst_config)
        else:
            self.burst_config = burst_config
            common.add2session(common.KEY_BURST_CONFIG, burst_config)

    @staticmethod
    def remove_burst_config_from_session():
        common.remove_from_session(common.KEY_BURST_CONFIG)

    def add_simulator_load_to_session(self, is_simulator_load):
        self.is_simulator_load = is_simulator_load
        common.add2session(common.KEY_IS_SIMULATOR_LOAD, is_simulator_load)

    @staticmethod
    def init_session_at_burst_loading(simulator):
        common.add2session(common.KEY_SIMULATOR_CONFIG, simulator)
        common.add2session(common.KEY_IS_SIMULATOR_LOAD, True)
        common.add2session(common.KEY_IS_SIMULATOR_COPY, False)

    def add_branch_and_copy_to_session(self, branch, copy):
        self.is_branch = branch
        self.is_simulator_copy = copy

        common.add2session(common.KEY_IS_SIMULATOR_BRANCH, branch)
        common.add2session(common.KEY_IS_SIMULATOR_COPY, copy)

    def init_session_at_copy_preparation(self, simulator, burst_config_copy):
        self.session_stored_simulator = simulator
        self.is_simulator_load = False
        self.burst_config = burst_config_copy

        common.add2session(common.KEY_SIMULATOR_CONFIG, simulator)
        common.add2session(common.KEY_IS_SIMULATOR_LOAD, False)
        common.add2session(common.KEY_BURST_CONFIG, burst_config_copy)

    def init_session_at_sim_reset(self):
        self.session_stored_simulator = None
        self.is_simulator_copy = False
        self.is_simulator_load = False
        self.is_branch = False

        common.add2session(common.KEY_SIMULATOR_CONFIG, None)
        common.add2session(common.KEY_IS_SIMULATOR_COPY, False)
        common.add2session(common.KEY_IS_SIMULATOR_LOAD, False)
        common.add2session(common.KEY_IS_SIMULATOR_BRANCH, False)

    def init_session_at_sim_config_from_zip(self, burst_config, simulator):
        self.burst_config = burst_config
        self.session_stored_simulator = simulator
        self.is_simulator_copy = True
        self.is_simulator_load = False

        common.add2session(common.KEY_BURST_CONFIG, burst_config)
        common.add2session(common.KEY_SIMULATOR_CONFIG, simulator)
        common.add2session(common.KEY_IS_SIMULATOR_COPY, True)
        common.add2session(common.KEY_IS_SIMULATOR_LOAD, False)

    @staticmethod
    def set_warning_message(message):
        common.set_warning_message(message)
