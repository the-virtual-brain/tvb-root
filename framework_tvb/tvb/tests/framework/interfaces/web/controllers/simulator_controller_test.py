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
from cherrypy.lib.sessions import RamSession
from cherrypy.test import helper
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.datatypes.surfaces import CORTICAL
from tvb.interfaces.web.controllers.common import KEY_USER, KEY_PROJECT
from tvb.interfaces.web.controllers.simulator_controller import SimulatorController, common
from tvb.simulator.coupling import Sigmoidal
from tvb.simulator.integrators import HeunDeterministic, IntegratorStochastic, Dopri5Stochastic
from tvb.simulator.models import Generic2dOscillator
from tvb.simulator.monitors import TemporalAverage, MEG
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

    def test_set_stimulus_none(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_surface(**self.sess_mock._data)

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
        assert self.session_stored_simulator.monitors[0].variables_of_interest is None, "Variables of interest should have not been added."
        # assert self.session_stored_simulator.monitors[0].region_mapping is not None, "Region Mapping was not added."
        # assert self.session_stored_simulator.monitors[0].projection is not None, "Projection was not added."
        assert self.session_stored_simulator.monitors[0].sigma == 1.0, "Sigma was not set correctly."
        # assert self.session_stored_simulator.monitors[0].sensors is not None, "Sensors where not added."






