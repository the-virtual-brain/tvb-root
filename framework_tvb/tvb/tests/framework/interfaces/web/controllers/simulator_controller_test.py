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

from os import path
from uuid import UUID
from mock import patch
import numpy
import tvb_data.surfaceData
import tvb_data.regionMapping
import tvb_data.sensors
import tvb_data.projectionMatrix
from datetime import datetime
from cherrypy.lib.sessions import RamSession
from cherrypy.test import helper
from tvb.adapters.creators.stimulus_creator import RegionStimulusCreator
from tvb.adapters.datatypes.db.patterns import StimuliRegionIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapterModel, CortexViewModel
from tvb.adapters.uploaders.sensors_importer import SensorsImporterModel
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.simulator.simulator_h5 import SimulatorH5
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.services.flow_service import FlowService
from tvb.core.services.simulator_serializer import SimulatorSerializer
from tvb.datatypes.equations import FirstOrderVolterra, GeneralizedSigmoid, TemporalApplicableEquation
from tvb.datatypes.surfaces import CORTICAL
from tvb.interfaces.web.controllers.common import *
from tvb.interfaces.web.controllers.simulator_controller import SimulatorController, common
from tvb.simulator.coupling import Sigmoidal
from tvb.simulator.integrators import HeunDeterministic, IntegratorStochastic, Dopri5Stochastic, EulerStochastic
from tvb.simulator.models import ModelsEnum
from tvb.simulator.monitors import TemporalAverage, MEG, Bold, SubSample, EEG, iEEG
from tvb.simulator.noise import Multiplicative
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest


class TestSimulationController(BaseTransactionalControllerTest, helper.CPWebCase):

    def transactional_setup_method(self):
        self.simulator_controller = SimulatorController()
        self.test_user = TestFactory.create_user('SimulationController_User')
        self.test_project = TestFactory.create_project(self.test_user, "SimulationController_Project")
        connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project)

        self.session_stored_simulator = SimulatorAdapterModel()
        self.session_stored_simulator.connectivity = UUID(connectivity.gid)

        self.sess_mock = RamSession()
        self.sess_mock[KEY_USER] = self.test_user
        self.sess_mock[KEY_PROJECT] = self.test_project

        cherrypy.request.method = "POST"

    def test_set_connectivity(self):
        zip_path = path.join(path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")

        self.sess_mock['connectivity'] = connectivity.gid
        self.sess_mock['conduction_speed'] = "3.0"
        self.sess_mock['coupling'] = "Sigmoidal"

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_connectivity(**self.sess_mock._data)

        assert self.session_stored_simulator.connectivity.hex == connectivity.gid, "Connectivity was not set correctly."
        assert self.session_stored_simulator.conduction_speed == 3.0, "Conduction speed was not set correctly."
        assert isinstance(self.session_stored_simulator.coupling, Sigmoidal), "Coupling was not set correctly."

    def test_set_coupling_params(self):
        self.sess_mock['a'] = '[0.00390625]'
        self.sess_mock['b'] = '[0.0]'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_coupling_params(**self.sess_mock._data)

        assert self.session_stored_simulator.coupling.a[0] == [0.00390625], "a value was not set correctly."
        assert self.session_stored_simulator.coupling.b[0] == [0.0], "b value was not set correctly."

    def test_set_surface(self):
        zip_path = path.join(path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        TestFactory.import_surface_zip(self.test_user, self.test_project, zip_path, CORTICAL, True)
        surface = TestFactory.get_entity(self.test_project, SurfaceIndex)

        self.sess_mock['surface'] = surface.gid

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
        connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")

        zip_path = path.join(path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        surface = TestFactory.import_surface_zip(self.test_user, self.test_project, zip_path, CORTICAL, True)

        text_file = path.join(path.dirname(tvb_data.regionMapping.__file__), 'regionMapping_16k_76.txt')
        region_mapping = TestFactory.import_region_mapping(self.test_user, self.test_project, text_file, surface.gid,
                                                           connectivity.gid)

        self.session_stored_simulator.surface = CortexViewModel()

        self.sess_mock['region_mapping'] = region_mapping.gid
        self.sess_mock['local_connectivity'] = 'explicit-None-value'
        self.sess_mock['coupling_strength'] = '[1.0]'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_cortex(**self.sess_mock._data)

        assert self.session_stored_simulator.surface.region_mapping_data.hex == region_mapping.gid, \
            'Region mapping was not set correctly'
        assert self.session_stored_simulator.surface.local_connectivity is None, \
            'Default value should have been set to local connectivity.'
        assert self.session_stored_simulator.surface.coupling_strength == [1.0], \
            "coupling_strength was not set correctly."

    def test_set_stimulus_none(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_stimulus(**self.sess_mock._data)

        assert self.session_stored_simulator.stimulus is None, "Stimulus should not be set."

    def test_set_stimulus(self):
        zip_path = path.join(path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        connectivity_index = TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path)
        weight_array = numpy.zeros(connectivity_index.number_of_regions)

        region_stimulus_creator = RegionStimulusCreator()
        view_model = region_stimulus_creator.get_view_model_class()()
        view_model.connectivity = UUID(connectivity_index.gid)
        view_model.weight = weight_array
        view_model.temporal = TemporalApplicableEquation()
        view_model.temporal.parameters['a'] = 1.0
        view_model.temporal.parameters['b'] = 2.0

        FlowService().fire_operation(region_stimulus_creator, self.test_user, self.test_project.id,
                                     view_model=view_model)
        region_stimulus_index = TestFactory.get_entity(self.test_project, StimuliRegionIndex)

        self.sess_mock['region_stimuli'] = UUID(region_stimulus_index.gid)

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_stimulus(**self.sess_mock._data)

        assert self.session_stored_simulator.stimulus.hex == region_stimulus_index.gid, \
            "Stimuli was not set correctly."

    def test_set_model(self):
        self.sess_mock['model'] = 'Generic 2d Oscillator'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_model(**self.sess_mock._data)

        assert isinstance(self.session_stored_simulator.model,
                          ModelsEnum.GENERIC_2D_OSCILLATOR.get_class()), "Model class is incorrect."

    def test_set_model_params(self):
        self.sess_mock['tau'] = '[1.0]'
        self.sess_mock['I'] = '[0.0]'
        self.sess_mock['a'] = '[-2.0]'
        self.sess_mock['b'] = '[-10.0]'
        self.sess_mock['c'] = '[0.0]'
        self.sess_mock['d'] = '[0.02]'
        self.sess_mock['e'] = '[3.0]'
        self.sess_mock['f'] = '[1.0]'
        self.sess_mock['g'] = '[0.0]'
        self.sess_mock['alpha'] = '[1.0]'
        self.sess_mock['beta'] = '[1.0]'
        self.sess_mock['gamma'] = '[1.0]'
        self.sess_mock['variables_of_interest'] = 'V'

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
        self.sess_mock['integrator'] = 'Heun'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_integrator(**self.sess_mock._data)

        assert isinstance(self.session_stored_simulator.integrator,
                          HeunDeterministic), "Integrator was not set correctly."

    def test_set_integrator_params(self):
        self.sess_mock['dt'] = '0.01220703125'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_integrator_params(**self.sess_mock._data)

        assert self.session_stored_simulator.integrator.dt == 0.01220703125, 'dt value was not set correctly.'

    def test_set_integrator_params_stochastic(self):
        self.sess_mock['dt'] = '0.01220703125'
        self.sess_mock['noise'] = 'Multiplicative'

        self.session_stored_simulator.integrator = Dopri5Stochastic()

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_integrator_params(**self.sess_mock._data)

        assert isinstance(self.session_stored_simulator.integrator, IntegratorStochastic), \
            "Coupling should be Stochastic Dormand-Prince."
        assert self.session_stored_simulator.integrator.dt == 0.01220703125, 'dt value was not set correctly.'
        assert isinstance(self.session_stored_simulator.integrator.noise, Multiplicative), 'Noise class is incorrect.'

    def test_set_noise_params(self):
        self.sess_mock['ntau'] = '0.0'
        self.sess_mock['noise_seed'] = '42'
        self.sess_mock['nsig'] = '[1.0]'

        self.session_stored_simulator.integrator = EulerStochastic()

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_noise_params(**self.sess_mock._data)

        assert self.session_stored_simulator.integrator.noise.ntau == 0.0, "ntau value was not set correctly."
        assert self.session_stored_simulator.integrator.noise.noise_seed == 42, \
            "noise_seed value was not set correctly."
        assert self.session_stored_simulator.integrator.noise.nsig == [1.0], "nsig value was not set correctly."

    def test_set_noise_equation_params(self):
        self.sess_mock['low'] = '0.1'
        self.sess_mock['high'] = '1.0'
        self.sess_mock['midpoint'] = '1.0'
        self.sess_mock['sigma'] = '0.3'

        self.session_stored_simulator.integrator = Dopri5Stochastic()
        self.session_stored_simulator.integrator.noise = Multiplicative()
        self.session_stored_simulator.integrator.noise.b = GeneralizedSigmoid()

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_noise_equation_params(**self.sess_mock._data)

        assert self.session_stored_simulator.integrator.noise.b.parameters['low'] == 0.1, \
            "low value was not set correctly"
        assert self.session_stored_simulator.integrator.noise.b.parameters['high'] == 1.0, \
            "high value was not set correctly"
        assert self.session_stored_simulator.integrator.noise.b.parameters['midpoint'] == 1.0, \
            "midpoint value was not set correctly"
        assert self.session_stored_simulator.integrator.noise.b.parameters['sigma'] == 0.3, \
            "sigma value was not set correctly"

    def test_set_monitors(self):
        self.sess_mock['monitor'] = 'Temporal average'

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_monitors(**self.sess_mock._data)

        assert isinstance(self.session_stored_simulator.monitors[0], TemporalAverage), 'Monitor class is incorrect.'

    def test_set_monitor_params(self):
        self.session_stored_simulator.model.variables_of_interest = ('V', 'W', 'V - W')
        variable_of_interest_indexes = {'W': 1, 'V - W': 2}
        self.sess_mock['variables_of_interest'] = list(variable_of_interest_indexes.keys())
        self.sess_mock['period'] = '0.8'
        self.session_stored_simulator.monitors = [SubSample()]

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            common.add2session(common.KEY_BURST_CONFIG, BurstConfiguration(self.test_project.id))
            self.simulator_controller.set_monitor_params(**self.sess_mock._data)

        assert self.session_stored_simulator.monitors[0].period == 0.8, "Period was not set correctly."
        assert list(self.session_stored_simulator.monitors[0].variables_of_interest) == \
            list(variable_of_interest_indexes.values()), "Variables of interest were not set correctly."

    def set_region_mapping(self):
        zip_path = path.join(path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_76.zip')
        connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")

        zip_path = path.join(path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        surface = TestFactory.import_surface_zip(self.test_user, self.test_project, zip_path, CORTICAL, True)

        text_file = path.join(path.dirname(tvb_data.regionMapping.__file__), 'regionMapping_16k_76.txt')
        region_mapping = TestFactory.import_region_mapping(self.test_user, self.test_project, text_file, surface.gid,
                                                           connectivity.gid)
        return region_mapping

    def test_set_eeg_monitor_params(self):
        region_mapping = self.set_region_mapping()

        eeg_sensors_file = path.join(path.dirname(tvb_data.sensors.__file__), 'eeg_unitvector_62.txt')
        eeg_sensors = TestFactory.import_sensors(self.test_user, self.test_project, eeg_sensors_file,
                                                 SensorsImporterModel.OPTIONS['EEG Sensors'])

        surface_file = path.join(path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        surface = TestFactory.import_surface_zip(self.test_user, self.test_project, surface_file, CORTICAL, True)

        eeg_projection_file = path.join(path.dirname(tvb_data.projectionMatrix.__file__),
                                        'projection_eeg_62_surface_16k.mat')
        eeg_projections = TestFactory.import_projection_matrix(self.test_user, self.test_project, eeg_projection_file,
                                                               eeg_sensors.gid, surface.gid)

        self.session_stored_simulator.model.variables_of_interest = ('V', 'W', 'V - W')
        variable_of_interest_indexes = {'W': 1, 'V - W': 2}
        self.sess_mock['variables_of_interest'] = list(variable_of_interest_indexes.keys())
        self.sess_mock['period'] = '0.75'
        self.sess_mock['region_mapping'] = region_mapping.gid
        self.sess_mock['projection'] = eeg_projections.gid
        self.sess_mock['sigma'] = "1.0"
        self.sess_mock['sensors'] = eeg_sensors.gid

        self.session_stored_simulator.monitors = [EEG()]

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            common.add2session(common.KEY_BURST_CONFIG, BurstConfiguration(self.test_project.id))
            self.simulator_controller.set_monitor_params(**self.sess_mock._data)

        assert self.session_stored_simulator.monitors[0].period == 0.75, "Period was not set correctly."
        assert list(self.session_stored_simulator.monitors[0].variables_of_interest) == \
            list(variable_of_interest_indexes.values()), "Variables of interest were not set correctly."
        assert self.session_stored_simulator.monitors[0].region_mapping.gid.hex == region_mapping.gid, \
            "Region Mapping wasn't set and stored correctly."
        assert self.session_stored_simulator.monitors[0].sensors.gid.hex == eeg_sensors.gid, \
            "Region Mapping wasn't set and stored correctly."
        assert self.session_stored_simulator.monitors[0].projection.gid is not None, \
            "Projection wasn't stored correctly."

    def test_set_meg_monitor_params(self):
        region_mapping = self.set_region_mapping()

        meg_sensors_file = path.join(path.dirname(tvb_data.sensors.__file__), 'meg_brainstorm_276.txt')
        meg_sensors = TestFactory.import_sensors(self.test_user, self.test_project, meg_sensors_file,
                                                 SensorsImporterModel.OPTIONS['MEG Sensors'])

        surface_file = path.join(path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        surface = TestFactory.import_surface_zip(self.test_user, self.test_project, surface_file, CORTICAL, True)

        meg_projection_file = path.join(path.dirname(tvb_data.projectionMatrix.__file__),
                                        'projection_meg_276_surface_16k.npy')
        meg_projections = TestFactory.import_projection_matrix(self.test_user, self.test_project, meg_projection_file,
                                                               meg_sensors.gid, surface.gid)

        self.session_stored_simulator.model.variables_of_interest = ('V', 'W', 'V - W')
        variable_of_interest_indexes = {'W': 1, 'V - W': 2}
        self.sess_mock['variables_of_interest'] = list(variable_of_interest_indexes.keys())
        self.sess_mock['period'] = '0.75'
        self.sess_mock['region_mapping'] = region_mapping.gid
        self.sess_mock['projection'] = meg_projections.gid
        self.sess_mock['sigma'] = 1.0
        self.sess_mock['sensors'] = meg_sensors.gid

        self.session_stored_simulator.monitors = [MEG()]

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            common.add2session(common.KEY_BURST_CONFIG, BurstConfiguration(self.test_project.id))
            self.simulator_controller.set_monitor_params(**self.sess_mock._data)

        assert self.session_stored_simulator.monitors[0].period == 0.75, "Period was not set correctly."
        assert list(self.session_stored_simulator.monitors[0].variables_of_interest) == \
            list(variable_of_interest_indexes.values()), "Variables of interest were not set correctly."
        assert self.session_stored_simulator.monitors[0].region_mapping.gid.hex == region_mapping.gid, \
            "Region Mapping wasn't set and stored correctly."
        assert self.session_stored_simulator.monitors[0].sensors.gid.hex == meg_sensors.gid, \
            "Region Mapping wasn't set and stored correctly."
        assert self.session_stored_simulator.monitors[0].projection.gid is not None, \
            "Projection wasn't stored correctly."

    def test_set_seeg_monitor_params(self):
        region_mapping = self.set_region_mapping()

        seeg_sensors_file = path.join(path.dirname(tvb_data.sensors.__file__), 'seeg_588.txt')
        seeg_sensors = TestFactory.import_sensors(self.test_user, self.test_project, seeg_sensors_file,
                                                  SensorsImporterModel.OPTIONS['Internal Sensors'])

        surface_file = path.join(path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        surface = TestFactory.import_surface_zip(self.test_user, self.test_project, surface_file, CORTICAL, True)

        seeg_projection_file = path.join(path.dirname(tvb_data.projectionMatrix.__file__),
                                         'projection_seeg_588_surface_16k.npy')
        seeg_projections = TestFactory.import_projection_matrix(self.test_user, self.test_project, seeg_projection_file,
                                                                seeg_sensors.gid, surface.gid)

        self.session_stored_simulator.model.variables_of_interest = ('V', 'W', 'V - W')
        variable_of_interest_indexes = {'W': 1, 'V - W': 2}
        self.sess_mock['variables_of_interest'] = list(variable_of_interest_indexes.keys())
        self.sess_mock['period'] = '0.75'
        self.sess_mock['region_mapping'] = region_mapping.gid
        self.sess_mock['projection'] = seeg_projections.gid
        self.sess_mock['sigma'] = "1.0"
        self.sess_mock['sensors'] = seeg_sensors.gid

        self.session_stored_simulator.monitors = [iEEG()]

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            common.add2session(common.KEY_BURST_CONFIG, BurstConfiguration(self.test_project.id))
            self.simulator_controller.set_monitor_params(**self.sess_mock._data)

        assert self.session_stored_simulator.monitors[0].period == 0.75, "Period was not set correctly."
        assert list(self.session_stored_simulator.monitors[0].variables_of_interest) == \
            list(variable_of_interest_indexes.values()), "Variables of interest were not set correctly."
        assert self.session_stored_simulator.monitors[0].region_mapping.gid.hex == region_mapping.gid, \
            "Region Mapping wasn't set and stored correctly."
        assert self.session_stored_simulator.monitors[0].sensors.gid.hex == seeg_sensors.gid, \
            "Region Mapping wasn't set and stored correctly."
        assert self.session_stored_simulator.monitors[0].projection.gid is not None, \
            "Projection wasn't stored correctly."

    def test_set_bold_monitor_params(self):
        self.session_stored_simulator.model.variables_of_interest = ('V', 'W', 'V - W')
        variable_of_interest_indexes = {'W': 1, 'V - W': 2}

        self.sess_mock['variables_of_interest'] = list(variable_of_interest_indexes.keys())
        self.sess_mock['period'] = '2000.0'
        self.sess_mock['hrf_kernel'] = 'HRF kernel: Volterra Kernel'

        self.session_stored_simulator.monitors = [Bold()]

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_monitor_params(**self.sess_mock._data)

        assert self.session_stored_simulator.monitors[0].period == 2000.0, "Period was not set correctly."
        assert list(self.session_stored_simulator.monitors[0].variables_of_interest) == \
            list(variable_of_interest_indexes.values()), "Variables of interest were not set correctly."

    def test_set_monitor_equation(self):
        self.sess_mock['tau_s'] = '0.8'
        self.sess_mock['tau_f'] = '0.4'
        self.sess_mock['k_1'] = '5.6'
        self.sess_mock['V_0'] = '0.02'

        self.session_stored_simulator.monitors = [Bold()]
        self.session_stored_simulator.monitors[0].equation = FirstOrderVolterra()

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            common.add2session(common.KEY_BURST_CONFIG, BurstConfiguration(self.test_project.id))
            self.simulator_controller.set_monitor_equation(**self.sess_mock._data)

        assert self.session_stored_simulator.monitors[0].equation.parameters[
                   'tau_s'] == 0.8, "tau_s value was not set correctly."
        assert self.session_stored_simulator.monitors[0].equation.parameters[
                   'tau_f'] == 0.4, "tau_f value was not set correctly."
        assert self.session_stored_simulator.monitors[0].equation.parameters[
                   'k_1'] == 5.6, "k_1 value was not set correctly."
        assert self.session_stored_simulator.monitors[0].equation.parameters[
                   'V_0'] == 0.02, "V_0 value was not set correctly."

    def test_load_burst_history(self):
        burst_config1 = BurstConfiguration(self.test_project.id)
        burst_config2 = BurstConfiguration(self.test_project.id)
        burst_config3 = BurstConfiguration(self.test_project.id)

        dao.store_entity(burst_config1)
        dao.store_entity(burst_config2)
        dao.store_entity(burst_config3)

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_BURST_CONFIG, burst_config1)
            burst_parameters = self.simulator_controller.load_burst_history()

        assert len(burst_parameters['burst_list']) == 3, "The burst configurations where not stored."

    def test_reset_simulator_configuration(self):
        zip_path = path.join(path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")

        self.sess_mock['connectivity'] = connectivity.gid
        self.sess_mock['conduction_speed'] = "3.0"
        self.sess_mock['coupling'] = "Sigmoidal"

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            rendering_rules = self.simulator_controller.set_connectivity(**self.sess_mock._data)

        assert rendering_rules['renderer'].is_first_fragment is False, \
            "Page should have advanced past the first fragment."

        with patch('cherrypy.session', self.sess_mock, create=True):
            rendering_rules = self.simulator_controller.reset_simulator_configuration()

        assert rendering_rules['renderer'].is_first_fragment is True, \
            "Page should be set to the first fragment."

    def test_get_history_status(self):
        burst_config = BurstConfiguration(self.test_project.id)
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
        new_name = "Test Burst Configuration 2"
        operation = TestFactory.create_operation()
        burst_config = TestFactory.store_burst(self.test_project.id, operation)
        burst = dao.get_bursts_for_project(self.test_project.id)
        self.sess_mock['burst_id'] = str(burst[0].id)
        self.sess_mock['burst_name'] = new_name

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_BURST_CONFIG, self.session_stored_simulator)
            common.add2session(common.KEY_BURST_CONFIG, burst_config)
            result = self.simulator_controller.rename_burst(burst[0].id, new_name)

        assert result == '{"success": "Simulation successfully renamed!"}', \
            "Some error happened at renaming, probably because of invalid new name."
        assert dao.get_bursts_for_project(self.test_project.id)[0].name == new_name, "Name wasn't actually changed."

    def test_export(self):
        op = TestFactory.create_operation(test_user=self.test_user, test_project=self.test_project)
        burst_config = BurstConfiguration(self.test_project.id)
        burst_config.fk_simulation = op.id
        burst_config.simulator_gid = self.session_stored_simulator.gid.hex
        burst_config = dao.store_entity(burst_config)

        storage_path = FilesHelper().get_project_folder(self.test_project, str(op.id))
        h5_path = h5.path_for(storage_path, SimulatorH5, self.session_stored_simulator.gid)
        with SimulatorH5(h5_path) as h5_file:
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
        connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")

        op = TestFactory.create_operation(test_user=self.test_user, test_project=self.test_project)
        burst_config = BurstConfiguration(self.test_project.id)
        burst_config.fk_simulation = op.id
        burst_config.simulator_gid = self.session_stored_simulator.gid.hex
        burst_config = dao.store_entity(burst_config)

        self.sess_mock['burst_id'] = str(burst_config.id)
        self.sess_mock['connectivity'] = connectivity.gid
        self.sess_mock['conduction_speed'] = "3.0"
        self.sess_mock['coupling'] = "Sigmoidal"

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_connectivity(**self.sess_mock._data)
            self.simulator_controller.set_stimulus(**self.sess_mock._data)

        storage_path = FilesHelper().get_project_folder(self.test_project, str(op.id))
        SimulatorSerializer().serialize_simulator(self.session_stored_simulator, None, storage_path)

        with patch('cherrypy.session', self.sess_mock, create=True):
            self.simulator_controller.copy_simulator_configuration(str(burst_config.id))
            is_simulator_load = common.get_from_session(KEY_IS_SIMULATOR_LOAD)
            is_simulator_copy = common.get_from_session(KEY_IS_SIMULATOR_COPY)

        assert not is_simulator_load, "Simulator Load Flag should be True!"
        assert is_simulator_copy, "Simulator Copy Flag should be False!"

    def test_load_burst_only(self):
        zip_path = path.join(path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")

        op = TestFactory.create_operation(test_user=self.test_user, test_project=self.test_project)
        burst_config = BurstConfiguration(self.test_project.id)
        burst_config.fk_simulation = op.id
        burst_config.simulator_gid = self.session_stored_simulator.gid.hex
        burst_config = dao.store_entity(burst_config)

        self.sess_mock['burst_id'] = str(burst_config.id)
        self.sess_mock['connectivity'] = connectivity.gid
        self.sess_mock['conduction_speed'] = "3.0"
        self.sess_mock['coupling'] = "Sigmoidal"

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.set_connectivity(**self.sess_mock._data)
            self.simulator_controller.set_stimulus(**self.sess_mock._data)

        storage_path = FilesHelper().get_project_folder(self.test_project, str(op.id))
        SimulatorSerializer().serialize_simulator(self.session_stored_simulator, None, storage_path)

        with patch('cherrypy.session', self.sess_mock, create=True):
            self.simulator_controller.load_burst_read_only(str(burst_config.id))
            is_simulator_load = common.get_from_session(KEY_IS_SIMULATOR_LOAD)
            is_simulator_copy = common.get_from_session(KEY_IS_SIMULATOR_COPY)
            last_loaded_form_url = common.get_from_session(KEY_LAST_LOADED_FORM_URL)

        assert is_simulator_load, "Simulator Load Flag should be True!"
        assert not is_simulator_copy, "Simulator Copy Flag should be False!"
        assert last_loaded_form_url == '/burst/setup_pse', "Incorrect last form URL!"

    def test_launch_simulation_with_default_parameters(self):
        self.sess_mock['input_simulation_name_id'] = 'HappySimulation'
        self.sess_mock['simulation_length'] = '10'
        launch_mode = 'new'

        burst_config = BurstConfiguration(self.test_project.id)

        with patch('cherrypy.session', self.sess_mock, create=True):
            common.add2session(common.KEY_BURST_CONFIG, burst_config)
            common.add2session(common.KEY_SIMULATOR_CONFIG, self.session_stored_simulator)
            self.simulator_controller.launch_simulation(launch_mode, **self.sess_mock._data)
