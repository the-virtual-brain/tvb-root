# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
from os import path
from mock import patch
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.adapters.uploaders.sensors_importer import SensorsImporterForm
from tvb.basic.profile import TvbProfile
import tvb_data
import tvb_data.surfaceData
import tvb_data.regionMapping
import tvb_data.sensors
import cherrypy
from datetime import datetime
from cherrypy.lib.sessions import RamSession
from cherrypy.test import helper
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.simulator.simulator_h5 import SimulatorH5
from tvb.core.entities.model.simulator.burst_configuration import BurstConfiguration2
from tvb.core.entities.model.simulator.simulator import SimulatorIndex
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.services.simulator_service import SimulatorService
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.equations import FirstOrderVolterra, GeneralizedSigmoid
from tvb.adapters.simulator.equation_forms import get_form_for_equation
from tvb.datatypes.surfaces import CORTICAL
from tvb.interfaces.web.controllers.common import KEY_USER, KEY_PROJECT, KEY_IS_SIMULATOR_LOAD, KEY_IS_SIMULATOR_COPY, \
    KEY_LAST_LOADED_FORM_URL
from tvb.interfaces.web.controllers.simulator_controller import SimulatorController, common
from tvb.simulator.coupling import Sigmoidal
from tvb.simulator.integrators import HeunDeterministic, IntegratorStochastic, Dopri5Stochastic, EulerStochastic
from tvb.simulator.models import Generic2dOscillator
from tvb.simulator.monitors import TemporalAverage, MEG, Bold
from tvb.simulator.noise import Multiplicative
from tvb.simulator.simulator import Simulator
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest


class TestSimulationController(BaseTransactionalControllerTest, helper.CPWebCase):

    def transactional_setup_method(self):
        self.simulator_controller = SimulatorController()
        self.test_user = TestFactory.create_user('SimulationController_User')
        self.test_project = TestFactory.create_project(self.test_user, "SimulationController_Project")
        TvbProfile.current.web.RENDER_HTML = False
        self.session_stored_simulator = Simulator()

        self.sess_mock = RamSession()
        self.sess_mock[KEY_USER] = self.test_user
        self.sess_mock[KEY_PROJECT] = self.test_project

        cherrypy.request.method = "POST"

    def test_set_connectivity(self):
        zip_path = path.join(path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")
        connectivity = TestFactory.get_entity(self.test_project, ConnectivityIndex)

        self.sess_mock['_connectivity'] = connectivity.gid
        self.sess_mock['_conduction_speed'] = "3.0"
        self.sess_mock['_coupling'] = "Sigmoidal"

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_connectivity(**self.sess_mock._data)

        assert self.session_stored_simulator.connectivity.gid.hex == connectivity.gid, "Connectivity was not set correctly."
        assert self.session_stored_simulator.conduction_speed == 3.0, "Conduction speed was not set correctly."
        assert isinstance(self.session_stored_simulator.coupling, Sigmoidal), "Coupling was not set correctly."

    def test_set_coupling_params(self):
        self.sess_mock['_a'] = '[0.00390625]'
        self.sess_mock['_b'] = '[0.0]'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_coupling_params(**self.sess_mock._data)

        assert self.session_stored_simulator.coupling.a[0] == [0.00390625], "a value was not set correctly."
        assert self.session_stored_simulator.coupling.b[0] == [0.0], "b value was not set correctly."

    def test_set_surface(self):
        zip_path = path.join(path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        TestFactory.import_surface_zip(self.test_user, self.test_project, zip_path, CORTICAL, True)
        surface = TestFactory.get_entity(self.test_project, SurfaceIndex)

        self.sess_mock['_surface'] = surface.gid

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_surface(**self.sess_mock._data)

        assert self.session_stored_simulator.surface is not None, "Surface was not set."

    def test_set_surface_none(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_surface(**self.sess_mock._data)

        assert self.session_stored_simulator.surface is None, "Surface should not be set."

    def test_set_cortex_without_local_connectivity(self):
        zip_path = path.join(path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_76.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")
        connectivity = TestFactory.get_entity(self.test_project, ConnectivityIndex)

        zip_path = path.join(path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        TestFactory.import_surface_zip(self.test_user, self.test_project, zip_path, CORTICAL, True)
        surface = TestFactory.get_entity(self.test_project, SurfaceIndex)

        text_file = path.join(path.dirname(tvb_data.regionMapping.__file__), 'regionMapping_16k_76.txt')
        region_mapping = TestFactory.import_region_mapping(text_file, surface.gid, connectivity.gid, self.test_user, self.test_project)

        self.session_stored_simulator.surface = Cortex()

        self.sess_mock['_region_mapping'] = region_mapping.gid
        self.sess_mock['_local_connectivity'] = 'explicit-None-value'
        self.sess_mock['_coupling_strength'] = '[1.0]'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_cortex(**self.sess_mock._data)

        assert self.session_stored_simulator.surface.region_mapping_data.gid.hex == region_mapping.gid,\
            'Region mapping was not set correctly'
        assert self.session_stored_simulator.surface.local_connectivity is None,\
            'Default value should have been set to local connectivity.'
        assert self.session_stored_simulator.surface.coupling_strength == [1.0],\
            "coupling_strength was not set correctly."

    def test_set_stimulus_none(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_stimulus(**self.sess_mock._data)

        assert self.session_stored_simulator.stimulus is None, "Stimulus should not be set."

    def test_set_model(self):
        self.sess_mock['_model'] = 'Generic 2d Oscillator'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_model(**self.sess_mock._data)

        assert isinstance(self.session_stored_simulator.model, Generic2dOscillator), "Model class is incorrect."

    def test_set_model_params(self):
        self.sess_mock['_tau'] = '[1.0]'
        self.sess_mock['_I'] = '[0.0]'
        self.sess_mock['_a'] = '[-2.0]'
        self.sess_mock['_b'] = '[-10.0]'
        self.sess_mock['_c'] = '[0.0]'
        self.sess_mock['_d'] = '[0.02]'
        self.sess_mock['_e'] = '[3.0]'
        self.sess_mock['_f'] = '[1.0]'
        self.sess_mock['_g'] = '[0.0]'
        self.sess_mock['_alpha'] = '[1.0]'
        self.sess_mock['_beta'] = '[1.0]'
        self.sess_mock['_gamma'] = '[1.0]'
        self.sess_mock['_variables_of_interest'] = 'V'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_model_params(**self.sess_mock._data)

        assert self.session_stored_simulator.model.tau == [1.0], "Tau has incorrect value."
        assert self.session_stored_simulator.model.I == [0.0], "I has incorrect value."
        assert self.session_stored_simulator.model.a == [-2.0], "a has incorrect value."
        assert self.session_stored_simulator.model.b == [-10.0], "b has incorrect value."
        assert self.session_stored_simulator.model.c == [0.0], "c has incorrect value."
        assert self.session_stored_simulator.model.d == [0.02], "d has incorrect value."
        assert self.session_stored_simulator.model.e == [3.0], "e has incorrect value."
        assert self.session_stored_simulator.model.f == [1.0], "f has incorrect value."
        assert self.session_stored_simulator.model.g == [0.0], "g has incorrect value."
        assert self.session_stored_simulator.model.alpha == [1.0], "alpha has incorrect value."
        assert self.session_stored_simulator.model.beta == [1.0], "beta has incorrect value."
        assert self.session_stored_simulator.model.gamma == [1.0], "gamma has incorrect value."
        assert self.session_stored_simulator.model.variables_of_interest == ['V'], \
            "variables_of_interest has incorrect value."

    def test_set_integrator(self):
        self.sess_mock['_integrator'] = 'Heun'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_integrator(**self.sess_mock._data)

        assert isinstance(self.session_stored_simulator.integrator, HeunDeterministic), "Integrator was not set correctly."

    def test_set_integrator_params(self):
        self.sess_mock['_dt'] = '0.01220703125'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_integrator_params(**self.sess_mock._data)

        assert self.session_stored_simulator.integrator.dt == 0.01220703125, 'dt value was not set correctly.'

    def test_set_integrator_params_stochastic(self):
        self.sess_mock['_dt'] = '0.01220703125'
        self.sess_mock['_noise'] = 'Multiplicative'

        self.session_stored_simulator.integrator = Dopri5Stochastic()

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_integrator_params(**self.sess_mock._data)

        assert isinstance(self.session_stored_simulator.integrator, IntegratorStochastic), \
            "Coupling should be Stochastic Dormand-Prince."
        assert self.session_stored_simulator.integrator.dt == 0.01220703125, 'dt value was not set correctly.'
        assert isinstance(self.session_stored_simulator.integrator.noise, Multiplicative), 'Noise class is incorrect.'

    def test_set_noise_params(self):
        self.sess_mock['_ntau'] = '0.0'
        self.sess_mock['_noise_seed'] = '42'
        self.sess_mock['_nsig'] = '[1.0]'

        self.session_stored_simulator.integrator = EulerStochastic()

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_noise_params(**self.sess_mock._data)

        assert self.session_stored_simulator.integrator.noise.ntau == 0.0, "ntau value was not set correctly."
        assert self.session_stored_simulator.integrator.noise.noise_seed == 42, \
            "noise_seed value was not set correctly."
        assert self.session_stored_simulator.integrator.noise.nsig == [1.0], "nsig value was not set correctly."

    def test_set_noise_equation_params(self):
        self.sess_mock['_low'] = '0.1'
        self.sess_mock['_high'] = '1.0'
        self.sess_mock['_midpoint'] = '1.0'
        self.sess_mock['_sigma'] = '0.3'

        self.session_stored_simulator.integrator = Dopri5Stochastic()
        self.session_stored_simulator.integrator.noise = Multiplicative()
        self.session_stored_simulator.integrator.noise.b = GeneralizedSigmoid()

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_noise_equation_params(**self.sess_mock._data)

        assert self.session_stored_simulator.integrator.noise.b.parameters['low'] == 0.1,\
            "low value was not set correctly"
        assert self.session_stored_simulator.integrator.noise.b.parameters['high'] == 1.0,\
            "high value was not set correctly"
        assert self.session_stored_simulator.integrator.noise.b.parameters['midpoint'] == 1.0,\
            "midpoint value was not set correctly"
        assert self.session_stored_simulator.integrator.noise.b.parameters['sigma'] == 0.3,\
            "sigma value was not set correctly"

    def test_set_monitors(self):
        self.sess_mock['_monitor'] = 'Temporal average'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_monitors(**self.sess_mock._data)

        assert isinstance(self.session_stored_simulator.monitors[0], TemporalAverage), 'Monitor class is incorrect.'

    def test_set_monitor_params_empty(self):

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_monitor_params(**self.sess_mock._data)

    def test_set_monitor_params(self):
        zip_path = path.join(path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_76.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")
        connectivity = TestFactory.get_entity(self.test_project, ConnectivityIndex)

        zip_path = path.join(path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        TestFactory.import_surface_zip(self.test_user, self.test_project, zip_path, CORTICAL, True)
        surface = TestFactory.get_entity(self.test_project, SurfaceIndex)

        text_file = path.join(path.dirname(tvb_data.regionMapping.__file__), 'regionMapping_16k_76.txt')
        region_mapping = TestFactory.import_region_mapping(text_file, surface.gid, connectivity.gid, self.test_user, self.test_project)

        meg_file = path.join(path.dirname(tvb_data.sensors.__file__), 'meg_151.txt.bz2')
        eeg_sensors = TestFactory.import_sensors(self.test_user, self.test_project, meg_file,
                                                       SensorsImporterForm.options['MEG Sensors'])

        self.sess_mock['_period'] = '0.9765625'
        self.sess_mock['_variables_of_interest'] = ''
        self.sess_mock['_region_mapping'] = region_mapping.gid
        self.sess_mock['_projection'] = eeg_sensors.gid
        self.sess_mock['_sigma'] = 1.0
        self.sess_mock['_sensors'] = eeg_sensors.gid

        self.session_stored_simulator.monitors = [MEG()]

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_monitor_params(**self.sess_mock._data)

        assert self.session_stored_simulator.monitors[0].period == 0.9765625, "Period was not set correctly."
        assert self.session_stored_simulator.monitors[0].variables_of_interest is None,\
            "Variables of interest should have not been added."

    def test_set_monitor_params_bold(self):
        self.sess_mock['_period'] = '2000.0'
        self.sess_mock['_variables_of_interest'] = ''
        self.sess_mock['_equation'] = 'HRF kernel: Volterra Kernel'

        self.session_stored_simulator.monitors = [Bold()]

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_monitor_params(**self.sess_mock._data)

        assert self.session_stored_simulator.monitors[0].period == 2000.0, "Period was not set correctly."
        assert self.session_stored_simulator.monitors[0].variables_of_interest is None,\
            "Variables of interest should have not been added."

    def test_set_monitor_equation(self):
        self.sess_mock['_tau_s'] = '0.8'
        self.sess_mock['_tau_f'] = '0.4'
        self.sess_mock['_k_1'] = '5.6'
        self.sess_mock['_V_0'] = '0.02'

        self.session_stored_simulator.monitors = [Bold()]
        self.session_stored_simulator.monitors[0].equation = FirstOrderVolterra()

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_monitor_equation(**self.sess_mock._data)

        form = get_form_for_equation(type(self.session_stored_simulator.monitors[0].equation))()

        assert form.tau_s.data == 0.8, "tau_s value was not set correctly."
        assert form.tau_f.data == 0.4, "tau_f value was not set correctly."
        assert form.k_1.data == 5.6, "k_1 value was not set correctly."
        assert form.V_0.data == 0.02, "V_0 value was not set correctly."

    def test_set_simulation_length(self):
        burst_config = BurstConfiguration2(self.test_project.id)

        self.sess_mock['_simulation_length'] = '1000.0'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            common.add2session(common.KEY_BURST_CONFIG, burst_config)
            self.simulator_controller.set_simulation_length(**self.sess_mock._data)

        assert self.session_stored_simulator.simulation_length == 1000.0, "simulation_length was not set correctly."

    def test_set_simulation_length_with_burst_config_name(self):
        burst_config = BurstConfiguration2(self.test_project.id)
        burst_config.name = "Test Burst Config"
        self.sess_mock['_simulation_length'] = '1000.0'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            common.add2session(common.KEY_BURST_CONFIG, burst_config)
            self.simulator_controller.set_simulation_length(**self.sess_mock._data)

        assert self.session_stored_simulator.simulation_length == 1000.0, "simulation_length was not set correctly."

    def test_load_burst_history(self):
        burst_config1 = BurstConfiguration2(self.test_project.id)
        burst_config2 = BurstConfiguration2(self.test_project.id)
        burst_config3 = BurstConfiguration2(self.test_project.id)

        dao.store_entity(burst_config1)
        dao.store_entity(burst_config2)
        dao.store_entity(burst_config3)

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_BURST_CONFIG, burst_config1)
            burst_parameters = self.simulator_controller.load_burst_history()

        assert len(burst_parameters['burst_list']) == 3, "The burst configurations where not stored."

    def test_reset_simulator_configuration(self):
        zip_path = path.join(path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")
        connectivity = TestFactory.get_entity(self.test_project, ConnectivityIndex)

        self.sess_mock['_connectivity'] = connectivity.gid
        self.sess_mock['_conduction_speed'] = "3.0"
        self.sess_mock['_coupling'] = "Sigmoidal"

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            rendering_rules = self.simulator_controller.set_connectivity(**self.sess_mock._data)

        assert rendering_rules['renderer'].is_first_fragment is False,\
            "Page should have advanced past the first fragment."

        with patch('cherrypy.session', self.sess_mock, create=True):
            rendering_rules = self.simulator_controller.reset_simulator_configuration()

        assert rendering_rules['renderer'].is_first_fragment is True,\
            "Page should be set to the first fragment."

    def test_get_history_status(self):
        burst_config = BurstConfiguration2(self.test_project.id)
        burst_config.start_time = datetime.now()
        dao.store_entity(burst_config)
        burst = dao.get_bursts_for_project(self.test_project.id)
        self.sess_mock['burst_ids'] = '["' + str(burst[0].id) + '"]'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_BURST_CONFIG, self.session_stored_simulator)
            common.add2session(common.KEY_BURST_CONFIG, burst_config)
            result = self.simulator_controller.get_history_status(**self.sess_mock._data).split(',')

        assert int(result[0][2:]) == burst[0].id, "Incorrect burst was used."
        assert result[1] == ' "running"', "Status should be set to running."
        assert result[2] == ' false', "Burst shouldn't be group."
        assert result[3] == ' ""', "Message should be empty, which means that there shouldn't be any errors."
        assert int(result[4][2:-4]) >= 0, "Running time should be greater than or equal to 0."

    def test_rename_burst(self):
        burst_config = BurstConfiguration2(self.test_project.id)
        burst_config.name = 'Test Burst Configuration'
        new_name = "Test Burst Configuration 2"
        dao.store_entity(burst_config)
        burst = dao.get_bursts_for_project(self.test_project.id)
        self.sess_mock['burst_id'] = str(burst[0].id)
        self.sess_mock['burst_name'] = new_name

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_BURST_CONFIG, self.session_stored_simulator)
            common.add2session(common.KEY_BURST_CONFIG, burst_config)
            result = self.simulator_controller.rename_burst(str(burst[0].id), new_name)

        assert result == '{"success": "Simulation successfully renamed!"}',\
            "Some error happened at renaming, probably because of invalid new name."
        assert dao.get_bursts_for_project(self.test_project.id)[0].name == new_name, "Name wasn't actually changed."

    def test_export(self):
        op = TestFactory.create_operation()
        simulator_index = SimulatorIndex()
        simulator_index.fill_from_has_traits(self.session_stored_simulator)

        burst_config = BurstConfiguration2(self.test_project.id, simulator_index.id)
        burst_config = dao.store_entity(burst_config)

        simulator_index.fk_from_operation = op.id
        simulator_index = dao.store_entity(simulator_index)
        simulator_index.fk_parent_burst = burst_config.id
        simulator_index = dao.store_entity(simulator_index)

        simulator_h5 = h5.path_for_stored_index(simulator_index)
        with SimulatorH5(simulator_h5) as h5_file:
            h5_file.store(self.session_stored_simulator)

        burst = dao.get_bursts_for_project(self.test_project.id)
        self.sess_mock['burst_id'] = str(burst[0].id)

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_BURST_CONFIG, self.session_stored_simulator)
            common.add2session(common.KEY_BURST_CONFIG, burst_config)
            result = self.simulator_controller.export(str(burst[0].id))

        assert path.exists(result.input.name), "Simulation was not exported!"

    def test_copy_simulator_configuration(self):
        zip_path = path.join(path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")
        connectivity = TestFactory.get_entity(self.test_project, ConnectivityIndex)

        simulator_index = SimulatorIndex()
        simulator_index.fill_from_has_traits(self.session_stored_simulator)

        burst_config = BurstConfiguration2(self.test_project.id, simulator_index.id)
        burst_config = dao.store_entity(burst_config)

        simulator_index.fk_from_operation = burst_config.id
        simulator_index = dao.store_entity(simulator_index)
        simulator_index.fk_parent_burst = burst_config.id
        simulator_index = dao.store_entity(simulator_index)

        burst = dao.get_bursts_for_project(self.test_project.id)

        self.sess_mock['burst_id'] = str(burst[0].id)
        self.sess_mock['_connectivity'] = connectivity.gid
        self.sess_mock['_conduction_speed'] = "3.0"
        self.sess_mock['_coupling'] = "Sigmoidal"

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_connectivity(**self.sess_mock._data)
            self.simulator_controller.set_stimulus(**self.sess_mock._data)

        storage_path = FilesHelper().get_project_folder(self.test_project, str(simulator_index.fk_from_operation))
        simulator_service = SimulatorService()
        simulator_service.serialize_simulator(self.session_stored_simulator, simulator_index.gid, None, storage_path)

        with patch('cherrypy.session', self.sess_mock, create=True):
            self.simulator_controller.copy_simulator_configuration(str(burst[0].id))
            is_simulator_load = common.get_from_session(KEY_IS_SIMULATOR_LOAD)
            is_simulator_copy = common.get_from_session(KEY_IS_SIMULATOR_COPY)

        database_simulator = dao.get_generic_entity(SimulatorIndex, burst_config.id, 'fk_parent_burst')[0]

        assert simulator_index.gid == database_simulator.gid, "Simulator was not added correctly!"
        assert not is_simulator_load, "Simulator Load Flag should be True!"
        assert is_simulator_copy, "Simulator Copy Flag should be False!"

    def test_load_burst_only(self):
        zip_path = path.join(path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")
        connectivity = TestFactory.get_entity(self.test_project, ConnectivityIndex)

        simulator_index = SimulatorIndex()
        simulator_index.fill_from_has_traits(self.session_stored_simulator)

        burst_config = BurstConfiguration2(self.test_project.id, simulator_index.id)
        burst_config = dao.store_entity(burst_config)

        simulator_index.fk_from_operation = burst_config.id
        simulator_index = dao.store_entity(simulator_index)
        simulator_index.fk_parent_burst = burst_config.id
        simulator_index = dao.store_entity(simulator_index)

        burst = dao.get_bursts_for_project(self.test_project.id)

        self.sess_mock['burst_id'] = str(burst[0].id)
        self.sess_mock['_connectivity'] = connectivity.gid
        self.sess_mock['_conduction_speed'] = "3.0"
        self.sess_mock['_coupling'] = "Sigmoidal"

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_connectivity(**self.sess_mock._data)
            self.simulator_controller.set_stimulus(**self.sess_mock._data)

        storage_path = FilesHelper().get_project_folder(self.test_project, str(simulator_index.fk_from_operation))
        simulator_service = SimulatorService()
        simulator_service.serialize_simulator(self.session_stored_simulator, simulator_index.gid, None, storage_path)

        with patch('cherrypy.session', self.sess_mock, create=True):
            self.simulator_controller.load_burst_read_only(str(burst[0].id))
            is_simulator_load = common.get_from_session(KEY_IS_SIMULATOR_LOAD)
            is_simulator_copy = common.get_from_session(KEY_IS_SIMULATOR_COPY)
            last_loaded_form_url = common.get_from_session(KEY_LAST_LOADED_FORM_URL)

        database_simulator = dao.get_generic_entity(SimulatorIndex, burst_config.id, 'fk_parent_burst')[0]

        assert simulator_index.gid == database_simulator.gid, "Simulator was not added correctly!"
        assert is_simulator_load, "Simulator Load Flag should be True!"
        assert not is_simulator_copy, "Simulator Copy Flag should be False!"
        assert last_loaded_form_url == '/burst/setup_pse', "Incorrect last form URL!"
