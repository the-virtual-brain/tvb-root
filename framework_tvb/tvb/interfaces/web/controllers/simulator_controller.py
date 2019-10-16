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
import copy
import threading
from tvb.simulator.integrators import IntegratorStochastic
from tvb.simulator.monitors import Bold
from tvb.simulator.noise import Additive
from tvb.adapters.simulator.equation_forms import get_form_for_equation
from tvb.adapters.simulator.model_forms import get_form_for_model
from tvb.adapters.simulator.noise_forms import get_form_for_noise
from tvb.adapters.simulator.range_parameter import SimulatorRangeParameters
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapterForm
from tvb.adapters.simulator.simulator_fragments import *
from tvb.adapters.simulator.monitor_forms import get_form_for_monitor
from tvb.adapters.simulator.integrator_forms import get_form_for_integrator
from tvb.adapters.simulator.coupling_forms import get_form_for_coupling
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.adapters.datatypes.db.simulation_state import SimulationStateIndex
from tvb.core.entities.model.model_operation import OperationGroup
from tvb.core.entities.model.simulator.burst_configuration import BurstConfiguration2
from tvb.core.entities.model.simulator.simulator import SimulatorIndex
from tvb.core.entities.storage import dao
from tvb.core.services.burst_service2 import BurstService2
from tvb.core.services.exceptions import BurstServiceException
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.core.services.simulator_service import SimulatorService
from tvb.core.neocom import h5
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.decorators import *


class SimulatorController(BurstBaseController):
    ACTION_KEY = 'action'
    PREVIOUS_ACTION_KEY = 'previous_action'
    FORM_KEY = 'form'
    IS_MODEL_FRAGMENT_KEY = 'is_model_fragment'
    IS_SURFACE_SIMULATION_KEY = 'is_surface_simulation'
    IS_FIRST_FRAGMENT_KEY = 'is_first_fragment'
    IS_LAST_FRAGMENT_KEY = 'is_last_fragment'
    IS_COPY = 'sim_copy'
    IS_LOAD = 'sim_load'

    dict_to_render = {
        ACTION_KEY: None,
        PREVIOUS_ACTION_KEY: None,
        FORM_KEY: None,
        IS_MODEL_FRAGMENT_KEY: False,
        IS_SURFACE_SIMULATION_KEY: False,
        IS_FIRST_FRAGMENT_KEY: False,
        IS_LAST_FRAGMENT_KEY: False,
        IS_COPY: False,
        IS_LOAD: False
    }

    def __init__(self):
        BurstBaseController.__init__(self)
        self.range_parameters = SimulatorRangeParameters()
        self.burst_service2 = BurstService2()
        self.simulator_service = SimulatorService()
        self.files_helper = FilesHelper()
        self.cached_simulator_algorithm = self.flow_service.get_algorithm_by_module_and_class(
            IntrospectionRegistry.SIMULATOR_MODULE, IntrospectionRegistry.SIMULATOR_CLASS)

    @expose_page
    @settings
    @context_selected
    def index(self):
        """Get on burst main page"""
        template_specification = dict(mainContent="burst/main_burst", title="Simulation Cockpit",
                                      baseUrl=TvbProfile.current.web.BASE_URL,
                                      includedResources='project/included_resources')
        project = common.get_current_project()

        burst_config = BurstConfiguration2(project.id)
        common.add2session(common.KEY_BURST_CONFIG, burst_config)
        template_specification['burstConfig'] = burst_config
        template_specification['burst_list'] = self.burst_service2.get_available_bursts(common.get_current_project().id)

        portlets_list = []  # self.burst_service.get_available_portlets()
        template_specification['portletList'] = portlets_list
        template_specification['selectedPortlets'] = json.dumps(portlets_list)

        form = self.prepare_first_fragment()

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.IS_FIRST_FRAGMENT_KEY] = True
        dict_to_render[self.FORM_KEY] = form
        dict_to_render[self.ACTION_KEY] = "/burst/set_connectivity"
        template_specification.update(**dict_to_render)

        return self.fill_default_attributes(template_specification)

    def prepare_first_fragment(self):
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        if is_simulator_copy is None:
            is_simulator_copy = False

        adapter_instance = ABCAdapter.build_adapter(self.cached_simulator_algorithm)
        form = adapter_instance.get_form()('', common.get_current_project().id, is_simulator_copy)

        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        if session_stored_simulator is None:
            session_stored_simulator = Simulator()  # self.burst_service.new_burst_configuration(common.get_current_project().id)
            common.add2session(common.KEY_SIMULATOR_CONFIG, session_stored_simulator)

        form.fill_from_trait(session_stored_simulator)
        return form

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_connectivity(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD)
        form = SimulatorAdapterForm()

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            form.fill_from_post(data)

            connectivity_index_gid = form.connectivity.value
            conduction_speed = form.conduction_speed.value
            coupling = form.coupling.value

            connectivity_index = ABCAdapter.load_entity_by_gid(connectivity_index_gid)
            connectivity = h5.load_from_index(connectivity_index)

            # TODO: handle this cases in a better manner
            session_stored_simulator.connectivity = connectivity
            session_stored_simulator.conduction_speed = conduction_speed
            session_stored_simulator.coupling = coupling()

        next_form = get_form_for_coupling(type(session_stored_simulator.coupling))()
        self.range_parameters.coupling_parameters = next_form.get_range_parameters()
        next_form.fill_from_trait(session_stored_simulator.coupling)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = next_form
        dict_to_render[self.ACTION_KEY] = '/burst/set_coupling_params'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_connectivity'
        dict_to_render[self.IS_COPY] = is_simulator_copy
        dict_to_render[self.IS_LOAD] = is_simulator_load
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_coupling_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD)

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            form = get_form_for_coupling(type(session_stored_simulator.coupling))()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.coupling)

        surface_fragment = SimulatorSurfaceFragment('', common.get_current_project().id)
        surface_fragment.fill_from_trait(session_stored_simulator)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = surface_fragment
        dict_to_render[self.ACTION_KEY] = '/burst/set_surface'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_coupling_params'
        dict_to_render[self.IS_COPY] = is_simulator_copy
        dict_to_render[self.IS_LOAD] = is_simulator_load
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_surface(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD)
        dict_to_render = copy.deepcopy(self.dict_to_render)
        surface_index = None

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            form = SimulatorSurfaceFragment()
            form.fill_from_post(data)

            surface_index_gid = form.surface.value
            # surface_index_gid = data['_surface']
            if surface_index_gid is None:
                session_stored_simulator.surface = None
            else:
                surface_index = ABCAdapter.load_entity_by_gid(surface_index_gid)
                session_stored_simulator.surface = Cortex()

        if session_stored_simulator.surface is None:
            stimuli_fragment = SimulatorStimulusFragment('', common.get_current_project().id, False)
            stimuli_fragment.fill_from_trait(session_stored_simulator)

            dict_to_render[self.FORM_KEY] = stimuli_fragment
            dict_to_render[self.ACTION_KEY] = '/burst/set_stimulus'
            dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_surface'
            dict_to_render[self.IS_COPY] = is_simulator_copy
            dict_to_render[self.IS_LOAD] = is_simulator_load
            return dict_to_render

        # TODO: work-around this situation: surf_index filter
        rm_fragment = SimulatorRMFragment('', common.get_current_project().id, surface_index)
        rm_fragment.fill_from_trait(session_stored_simulator)
        dict_to_render[self.FORM_KEY] = rm_fragment
        dict_to_render[self.ACTION_KEY] = '/burst/set_cortex'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_surface'
        dict_to_render[self.IS_COPY] = is_simulator_copy
        dict_to_render[self.IS_LOAD] = is_simulator_load
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_cortex(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD)

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            rm_fragment = SimulatorRMFragment()
            rm_fragment.fill_from_post(data)

            session_stored_simulator.surface.coupling_strength = rm_fragment.coupling_strength.data

            lc_gid = rm_fragment.lc.value
            if lc_gid == 'None':
                lc_index = ABCAdapter.load_entity_by_gid(lc_gid)
                lc = h5.load_from_index(lc_index)
                session_stored_simulator.surface.local_connectivity = lc

            rm_gid = rm_fragment.rm.value
            rm_index = ABCAdapter.load_entity_by_gid(rm_gid)
            rm = h5.load_from_index(rm_index)
            session_stored_simulator.surface.region_mapping_data = rm

        stimuli_fragment = SimulatorStimulusFragment('', common.get_current_project().id, True)
        stimuli_fragment.fill_from_trait(session_stored_simulator)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = stimuli_fragment
        dict_to_render[self.ACTION_KEY] = '/burst/set_stimulus'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_cortex'
        dict_to_render[self.IS_COPY] = is_simulator_copy
        dict_to_render[self.IS_LOAD] = is_simulator_load
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_stimulus(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD)

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            stimuli_fragment = SimulatorStimulusFragment('', common.get_current_project().id,
                                                         session_stored_simulator.is_surface_simulation)
            stimuli_fragment.fill_from_post(data)
            stimulus_gid = stimuli_fragment.stimulus.value
            if stimulus_gid != None:
                stimulus_index = ABCAdapter.load_entity_by_gid(stimulus_gid)
                stimulus = h5.load_from_index(stimulus_index)
                session_stored_simulator.stimulus = stimulus

        model_fragment = SimulatorModelFragment('', common.get_current_project().id)
        model_fragment.fill_from_trait(session_stored_simulator)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = model_fragment
        dict_to_render[self.ACTION_KEY] = '/burst/set_model'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_stimulus'
        dict_to_render[self.IS_MODEL_FRAGMENT_KEY] = True
        dict_to_render[self.IS_COPY] = is_simulator_copy
        dict_to_render[self.IS_LOAD] = is_simulator_load
        dict_to_render[self.IS_SURFACE_SIMULATION_KEY] = session_stored_simulator.is_surface_simulation
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_model(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD)

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            form = SimulatorModelFragment()
            form.fill_from_post(data)
            session_stored_simulator.model = form.model.value()

        form = get_form_for_model(type(session_stored_simulator.model))()
        self.range_parameters.model_parameters = form.get_range_parameters()
        form.fill_from_trait(session_stored_simulator.model)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = form
        dict_to_render[self.ACTION_KEY] = '/burst/set_model_params'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_model'
        dict_to_render[self.IS_COPY] = is_simulator_copy
        dict_to_render[self.IS_LOAD] = is_simulator_load
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_model_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD)

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            form = get_form_for_model(type(session_stored_simulator.model))()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.model)

        integrator_fragment = SimulatorIntegratorFragment('', common.get_current_project().id)
        integrator_fragment.fill_from_trait(session_stored_simulator)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = integrator_fragment
        dict_to_render[self.ACTION_KEY] = '/burst/set_integrator'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_model_params'
        dict_to_render[self.IS_COPY] = is_simulator_copy
        dict_to_render[self.IS_LOAD] = is_simulator_load
        return dict_to_render

    # TODO: add state_variables selection step
    # @cherrypy.expose
    # @using_jinja_template("wizzard_form")
    # @handle_error(redirect=False)
    # @check_user
    # def set_model_variables_to_monitor(self, data):
    #     session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
    #     form = get_form_for_model(type(session_stored_simulator.model.variables))()
    #     form.fill_from_post(data)
    #
    #     form.fill_trait(session_stored_simulator.model)
    #
    #     integrator_fragment = SimulatorIntegratorFragment('', common.get_current_project().id)
    #
    #     return {'form': integrator_fragment, 'action': '/burst/set_integrator',
    #             'previous_action': '/burst/set_model_variables_to_monitor'}

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_integrator(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD)

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            fragment = SimulatorIntegratorFragment()
            fragment.fill_from_post(data)
            session_stored_simulator.integrator = fragment.integrator.value()

        form = get_form_for_integrator(type(session_stored_simulator.integrator))()
        form.fill_from_trait(session_stored_simulator.integrator)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = form
        dict_to_render[self.ACTION_KEY] = '/burst/set_integrator_params'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_integrator'
        dict_to_render[self.IS_COPY] = is_simulator_copy
        dict_to_render[self.IS_LOAD] = is_simulator_load
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_integrator_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD)

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            form = get_form_for_integrator(type(session_stored_simulator.integrator))()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.integrator)

        if isinstance(session_stored_simulator.integrator, IntegratorStochastic):
            integrator_noise_fragment = get_form_for_noise(type(session_stored_simulator.integrator.noise))()
            self.range_parameters.integrator_noise_parameters = integrator_noise_fragment.get_range_parameters()
            integrator_noise_fragment.fill_from_trait(session_stored_simulator.integrator.noise)

            dict_to_render = copy.deepcopy(self.dict_to_render)
            dict_to_render[self.FORM_KEY] = integrator_noise_fragment
            dict_to_render[self.ACTION_KEY] = '/burst/set_noise_params'
            dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_integrator_params'
            dict_to_render[self.IS_COPY] = is_simulator_copy
            dict_to_render[self.IS_LOAD] = is_simulator_load
            return dict_to_render

        monitor_fragment = SimulatorMonitorFragment('', common.get_current_project().id)
        monitor_fragment.fill_from_trait(session_stored_simulator)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = monitor_fragment
        dict_to_render[self.ACTION_KEY] = '/burst/set_monitors'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_integrator_params'
        dict_to_render[self.IS_COPY] = is_simulator_copy
        dict_to_render[self.IS_LOAD] = is_simulator_load
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_noise_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD)

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            form = get_form_for_noise(type(session_stored_simulator.integrator.noise))()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.integrator.noise)

        if isinstance(session_stored_simulator.integrator.noise, Additive):
            monitor_fragment = SimulatorMonitorFragment('', common.get_current_project().id)
            monitor_fragment.fill_from_trait(session_stored_simulator)

            dict_to_render = copy.deepcopy(self.dict_to_render)
            dict_to_render[self.FORM_KEY] = monitor_fragment
            dict_to_render[self.ACTION_KEY] = '/burst/set_monitors'
            dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_noise_params'
            dict_to_render[self.IS_COPY] = is_simulator_copy
            dict_to_render[self.IS_LOAD] = is_simulator_load
            return dict_to_render

        equation_form = get_form_for_equation(type(session_stored_simulator.integrator.noise.b))()
        equation_form.equation.data = session_stored_simulator.integrator.noise.b.__class__.__name__

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = equation_form
        dict_to_render[self.ACTION_KEY] = '/burst/set_noise_equation_params'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_noise_params'
        dict_to_render[self.IS_COPY] = is_simulator_copy
        dict_to_render[self.IS_LOAD] = is_simulator_load
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_noise_equation_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD)

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            form = get_form_for_equation(type(session_stored_simulator.integrator.noise.b))()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.integrator.noise.b)

        monitor_fragment = SimulatorMonitorFragment('', common.get_current_project().id)
        monitor_fragment.fill_from_trait(session_stored_simulator)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = monitor_fragment
        dict_to_render[self.ACTION_KEY] = '/burst/set_monitors'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_noise_equation_params'
        dict_to_render[self.IS_COPY] = is_simulator_copy
        dict_to_render[self.IS_LOAD] = is_simulator_load
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_monitors(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD)

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            # TODO: handle multiple monitors
            fragment = SimulatorMonitorFragment()
            fragment.fill_from_post(data)

            session_stored_simulator.monitors = [fragment.monitor.value()]

        monitor = session_stored_simulator.monitors[0]
        form = get_form_for_monitor(type(monitor))('', common.get_current_project().id)
        form.fill_from_trait(monitor)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = form
        dict_to_render[self.ACTION_KEY] = '/burst/set_monitor_params'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_monitors'
        dict_to_render[self.IS_COPY] = is_simulator_copy
        dict_to_render[self.IS_LOAD] = is_simulator_load
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_monitor_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        monitor = session_stored_simulator.monitors[0]
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD)

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            form = get_form_for_monitor(type(monitor))()
            form.fill_from_post(data)
            form.fill_trait(monitor)

        if isinstance(monitor, Bold):
            next_form = get_form_for_equation(type(monitor.hrf_kernel))()
            next_form.fill_from_trait(session_stored_simulator.monitors[0].hrf_kernel)

            dict_to_render = copy.deepcopy(self.dict_to_render)
            dict_to_render[self.FORM_KEY] = next_form
            dict_to_render[self.ACTION_KEY] = '/burst/set_monitor_equation'
            dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_monitor_params'
            dict_to_render[self.IS_COPY] = is_simulator_copy
            dict_to_render[self.IS_LOAD] = is_simulator_load
            return dict_to_render
        session_stored_simulator.monitors = [monitor]

        next_form = SimulatorLengthFragment()
        next_form.fill_from_trait(session_stored_simulator)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = next_form
        dict_to_render[self.ACTION_KEY] = '/burst/set_simulation_length'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_monitor_params'
        dict_to_render[self.IS_COPY] = is_simulator_copy
        dict_to_render[self.IS_LOAD] = is_simulator_load
        if is_simulator_load:
            dict_to_render[self.ACTION_KEY] = ''
            dict_to_render[self.IS_LAST_FRAGMENT_KEY] = True
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_monitor_equation(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD)

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            form = get_form_for_monitor(type(session_stored_simulator.monitors[0]))()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.monitor.hrf_kernel)

        next_form = SimulatorLengthFragment()

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = next_form
        dict_to_render[self.ACTION_KEY] = '/burst/set_simulation_length'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_monitor_equation'
        dict_to_render[self.IS_COPY] = is_simulator_copy
        dict_to_render[self.IS_LOAD] = is_simulator_load
        if is_simulator_load:
            dict_to_render[self.ACTION_KEY] = ''
            dict_to_render[self.IS_LAST_FRAGMENT_KEY] = True
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_simulation_length(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD)
        session_burst_config = common.get_from_session(common.KEY_BURST_CONFIG)

        dict_to_render = copy.deepcopy(self.dict_to_render)

        if is_simulator_load:
            common.add2session(common.KEY_IS_SIMULATOR_LOAD, False)

        next_form = SimulatorFinalFragment()
        if session_burst_config.name:
            burst_name = session_burst_config.name
            copy_prefix = 'Copy of '
            if is_simulator_copy and burst_name.find(copy_prefix) < 0:
                burst_name = copy_prefix + burst_name
            next_form.simulation_name.data = burst_name

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            fragment = SimulatorLengthFragment()
            fragment.fill_from_post(data)
            session_stored_simulator.simulation_length = fragment.length.value

        dict_to_render[self.FORM_KEY] = next_form
        dict_to_render[self.ACTION_KEY] = '/burst/setup_pse'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_simulation_length'
        dict_to_render[self.IS_COPY] = is_simulator_copy
        dict_to_render[self.IS_LAST_FRAGMENT_KEY] = True
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def setup_pse(self, **data):
        next_form = SimulatorPSEConfigurationFragment(self.range_parameters.get_all_range_parameters())

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = next_form
        dict_to_render[self.ACTION_KEY] = '/burst/set_pse_params'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_simulation_length'
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_pse_params(self, **data):
        form = SimulatorPSEConfigurationFragment(self.range_parameters.get_all_range_parameters())
        form.fill_from_post(data)

        param1 = form.pse_param1.value
        param2 = None
        if not form.pse_param2.value == form.pse_param2.missing_value:
            param2 = form.pse_param2.value

        project_id = common.get_current_project().id
        next_form = SimulatorPSEParamRangeFragment(param1, param2, project_id=project_id)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = next_form
        dict_to_render[self.ACTION_KEY] = '/burst/launch_pse'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_pse_params'
        return dict_to_render

    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def launch_pse(self, **data):
        # TODO: Split into: set range values and Launch, show message with finished config and nr of simulations
        all_range_parameters = self.range_parameters.get_all_range_parameters()
        range_param1, range_param2 = SimulatorPSEParamRangeFragment.fill_from_post(all_range_parameters, **data)
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)

        project = common.get_current_project()
        user = common.get_logged_user()

        burst_config = common.get_from_session(common.KEY_BURST_CONFIG)
        burst_config.start_time = datetime.now()
        # if burst_name != 'none_undefined':
        #     burst_config.name = burst_name

        # TODO: branch simulation name is different
        if burst_config.name is None:
            new_id = dao.get_max_burst_id() + 1
            burst_config.name = 'simulation_' + str(new_id)

        operation_group = OperationGroup(project.id, ranges=[range_param1.to_json(), range_param2.to_json()])
        operation_group = dao.store_entity(operation_group)

        metric_operation_group = OperationGroup(project.id, ranges=[range_param1.to_json(), range_param2.to_json()])
        metric_operation_group = dao.store_entity(metric_operation_group)

        burst_config.operation_group = operation_group
        burst_config.operation_group_id = operation_group.id
        burst_config.metric_operation_group = metric_operation_group
        burst_config.metric_operation_group_id = metric_operation_group.id
        dao.store_entity(burst_config)

        try:
            thread = threading.Thread(target=self.simulator_service.async_launch_and_prepare_pse,
                                      kwargs={'burst_config': burst_config,
                                              'user': user,
                                              'project': project,
                                              'simulator_algo': self.cached_simulator_algorithm,
                                              'range_param1': range_param1,
                                              'range_param2': range_param2,
                                              'session_stored_simulator': session_stored_simulator})
            thread.start()
        except BurstServiceException as e:
            self.logger.exception("Could not launch burst!")
            return {'error': e.message}

    @expose_json
    def launch_simulation(self, launch_mode, **data):
        current_form = SimulatorFinalFragment()
        try:
            current_form.fill_from_post(data)
        except Exception as exc:
            self.logger.exception(exc)
            return {'error': str(exc)}

        burst_name = current_form.simulation_name.value
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY)

        project = common.get_current_project()
        user = common.get_logged_user()

        session_burst_config = common.get_from_session(common.KEY_BURST_CONFIG)
        if burst_name != 'none_undefined':
            session_burst_config.name = burst_name

        burst_config_to_store = session_burst_config
        simulation_state_index_gid = None
        if launch_mode == self.simulator_service.LAUNCH_NEW:
            if session_burst_config.name is None:
                new_id = dao.get_max_burst_id() + 1
                session_burst_config.name = 'simulation_' + str(new_id)
            if is_simulator_copy:
                burst_config_to_store = session_burst_config.clone()
        else:
            burst_config_to_store = session_burst_config.clone()
            count = dao.count_bursts_with_name(session_burst_config.name, session_burst_config.project_id)
            session_burst_config.name = session_burst_config.name + "_" + launch_mode + str(count)
            simulation_state_index = dao.get_generic_entity(
                SimulationStateIndex.__module__ + "." + SimulationStateIndex.__name__,
                session_burst_config.id, "fk_parent_burst")
            if simulation_state_index is None or len(simulation_state_index) < 1:
                exc = BurstServiceException("Simulation State not found for %s, thus we are unable to branch from "
                                            "it!" % session_burst_config.name)
                self.logger.error(exc)
                raise exc
            simulation_state_index_gid = simulation_state_index[0].gid

        burst_config_to_store.start_time = datetime.now()
        dao.store_entity(burst_config_to_store)

        try:
            thread = threading.Thread(target=self.simulator_service.async_launch_and_prepare_simulation,
                                      kwargs={'burst_config': burst_config_to_store,
                                              'user': user,
                                              'project': project,
                                              'simulator_algo': self.cached_simulator_algorithm,
                                              'session_stored_simulator': session_stored_simulator,
                                              'simulation_state_gid': simulation_state_index_gid})
            thread.start()
            return {'id': burst_config_to_store.id}
        except BurstServiceException as e:
            self.logger.exception('Could not launch burst!')
            return {'error': e.message}

    @expose_fragment('burst/burst_history')
    def load_burst_history(self):
        """
        Load the available burst that are stored in the database at this time.
        This is one alternative to 'chrome-back problem'.
        """
        session_burst = common.get_from_session(common.KEY_BURST_CONFIG)
        bursts = self.burst_service2.get_available_bursts(common.get_current_project().id)
        self.burst_service2.populate_burst_disk_usage(bursts)
        return {'burst_list': bursts,
                'selectedBurst': session_burst.id}

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def load_burst_read_only(self, burst_config_id):
        try:
            burst_config = dao.get_burst_by_id(burst_config_id)
            common.add2session(common.KEY_BURST_CONFIG, burst_config)

            simulator_index = dao.get_generic_entity(SimulatorIndex, burst_config.id, 'fk_parent_burst')[0]
            simulator_gid = simulator_index.gid

            project = common.get_current_project()
            storage_path = self.files_helper.get_project_folder(project, str(simulator_index.fk_from_operation))

            simulator, _, _ = self.simulator_service.deserialize_simulator(simulator_gid, storage_path)

            session_stored_simulator = simulator
            common.add2session(common.KEY_SIMULATOR_CONFIG, session_stored_simulator)
            common.add2session(common.KEY_IS_SIMULATOR_LOAD, True)
            common.add2session(common.KEY_IS_SIMULATOR_COPY, False)

            form = self.prepare_first_fragment()
            dict_to_render = copy.deepcopy(self.dict_to_render)
            dict_to_render[self.IS_FIRST_FRAGMENT_KEY] = True
            dict_to_render[self.FORM_KEY] = form
            dict_to_render[self.ACTION_KEY] = "/burst/set_connectivity"
            dict_to_render[self.IS_LOAD] = True
            return dict_to_render
        except Exception:
            ### Most probably Burst was removed. Delete it from session, so that client
            ### has a good chance to get a good response on refresh
            self.logger.exception("Error loading burst")
            common.remove_from_session(common.KEY_BURST_CONFIG)
            raise

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def copy_simulator_configuration(self, burst_config_id):
        burst_config = dao.get_burst_by_id(burst_config_id)
        common.add2session(common.KEY_BURST_CONFIG, burst_config)

        simulator_index = dao.get_generic_entity(SimulatorIndex, burst_config.id, 'fk_parent_burst')[0]
        simulator_gid = simulator_index.gid

        project = common.get_current_project()
        storage_path = self.files_helper.get_project_folder(project, str(simulator_index.fk_from_operation))

        simulator, _, _ = self.simulator_service.deserialize_simulator(simulator_gid, storage_path)

        session_stored_simulator = simulator
        common.add2session(common.KEY_SIMULATOR_CONFIG, session_stored_simulator)
        common.add2session(common.KEY_IS_SIMULATOR_COPY, True)
        common.add2session(common.KEY_IS_SIMULATOR_LOAD, False)

        form = self.prepare_first_fragment()
        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.IS_FIRST_FRAGMENT_KEY] = True
        dict_to_render[self.FORM_KEY] = form
        dict_to_render[self.ACTION_KEY] = "/burst/set_connectivity"
        dict_to_render[self.IS_COPY] = True
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def reset_simulator_configuration(self):
        common.add2session(common.KEY_SIMULATOR_CONFIG, None)
        common.add2session(common.KEY_IS_SIMULATOR_COPY, False)
        common.add2session(common.KEY_IS_SIMULATOR_LOAD, False)

        project = common.get_current_project()
        common.add2session(common.KEY_BURST_CONFIG, BurstConfiguration2(project.id))

        form = self.prepare_first_fragment()
        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.IS_FIRST_FRAGMENT_KEY] = True
        dict_to_render[self.FORM_KEY] = form
        dict_to_render[self.ACTION_KEY] = "/burst/set_connectivity"
        return dict_to_render

    @expose_json
    def rename_burst(self, burst_id, burst_name):
        """
        Rename the burst given by burst_id, setting it's new name to
        burst_name.
        """
        validation_result = SimulatorFinalFragment.is_burst_name_ok(burst_name)
        if validation_result is True:
            self.burst_service2.rename_burst(burst_id, burst_name)
            return {'success': "Simulation successfully renamed!"}
        else:
            self.logger.exception(validation_result)
            return {'error': validation_result}

    @expose_json
    def get_history_status(self, **data):
        """
        For each burst id received, get the status and return it.
        """
        return self.burst_service2.update_history_status(json.loads(data['burst_ids']))
