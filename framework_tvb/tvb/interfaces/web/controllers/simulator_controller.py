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
import json
import cherrypy
import datetime
from tvb.datatypes.cortex import Cortex
from tvb.basic.profile import TvbProfile
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.datatypes.region_mapping import RegionMapping
from tvb.simulator.monitors import Bold
from tvb.simulator.noise import Additive
from tvb.simulator.simulator import Simulator
from tvb.adapters.simulator.equation_forms import get_form_for_equation
from tvb.adapters.simulator.model_forms import get_form_for_model
from tvb.adapters.simulator.noise_forms import get_form_for_noise
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapterForm
from tvb.adapters.simulator.simulator_fragments import SimulatorSurfaceFragment, SimulatorStimulusFragment, \
    SimulatorModelFragment, SimulatorIntegratorFragment, SimulatorMonitorFragment, SimulatorLengthFragment, \
    SimulatorRMFragment
from tvb.adapters.simulator.monitor_forms import get_form_for_monitor
from tvb.adapters.simulator.integrator_forms import get_form_for_integrator
from tvb.adapters.simulator.coupling_forms import get_form_for_coupling
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.file.datatypes.connectivity_h5 import ConnectivityH5
from tvb.core.entities.file.datatypes.local_connectivity_h5 import LocalConnectivityH5
from tvb.core.entities.file.datatypes.region_mapping_h5 import RegionMappingH5
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.model.simulator.burst_configuration import BurstConfiguration2
from tvb.core.entities.model.simulator.simulator import SimulatorIndex
from tvb.core.entities.storage import dao
from tvb.core.services.burst_service import BurstService
from tvb.core.services.introspector_registry import IntrospectionRegistry
from tvb.core.services.operation_service import OperationService
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.neocom._h5loader import DirLoader
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.decorators import expose_page, settings, context_selected, handle_error, check_user, \
    using_jinja_template, expose_json, expose_fragment


class SimulatorController(BurstBaseController):
    ACTION_KEY = 'action'
    PREVIOUS_ACTION_KEY = 'previous_action'
    FORM_KEY = 'form'
    IS_MODEL_FRAGMENT_KEY = 'is_model_fragment'
    IS_SURFACE_SIMULATION_KEY = 'is_surface_simulation'
    IS_FIRST_FRAGMENT_KEY = 'is_first_fragment'
    IS_LAST_FRAGMENT_KEY = 'is_last_fragment'

    dict_to_render = {
        ACTION_KEY: None,
        PREVIOUS_ACTION_KEY: None,
        FORM_KEY: None,
        IS_MODEL_FRAGMENT_KEY: False,
        IS_SURFACE_SIMULATION_KEY: False,
        IS_FIRST_FRAGMENT_KEY: False,
        IS_LAST_FRAGMENT_KEY: False
    }

    def __init__(self):
        BurstBaseController.__init__(self)
        self.burst_service = BurstService()
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

        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        if session_stored_simulator is None:
            if session_stored_simulator is None:
                session_stored_simulator = Simulator()  # self.burst_service.new_burst_configuration(common.get_current_project().id)
                common.add2session(common.KEY_SIMULATOR_CONFIG, session_stored_simulator)

        adapter_instance = ABCAdapter.build_adapter(self.cached_simulator_algorithm)
        form = adapter_instance.get_form()('', common.get_current_project().id)
        # TODO: does not handle DTselectFields. They do not have traited_attr
        form.fill_from_trait(session_stored_simulator)

        project = common.get_current_project()

        burst_config = BurstConfiguration2(project.id)
        common.add2session(common.KEY_BURST_CONFIG, burst_config)
        template_specification['burstConfig'] = burst_config
        template_specification['burst_list'] = self.burst_service.get_available_bursts(common.get_current_project().id)

        portlets_list = []  # self.burst_service.get_available_portlets()
        template_specification['portletList'] = portlets_list
        template_specification['selectedPortlets'] = json.dumps(portlets_list)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.IS_FIRST_FRAGMENT_KEY] = True
        dict_to_render[self.FORM_KEY] = form
        dict_to_render[self.ACTION_KEY] = "/burst/set_connectivity"
        template_specification.update(**dict_to_render)

        return self.fill_default_attributes(template_specification)

    def get_first_fragment(self):
        pass

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_connectivity(self, **data):
        form = SimulatorAdapterForm()
        form.fill_from_post(data)

        connectivity_index_gid = form.connectivity.value
        conduction_speed = form.conduction_speed.value
        coupling = form.coupling.value

        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        connectivity_index = ABCAdapter.load_entity_by_gid(connectivity_index_gid)
        storage_path = self.files_helper.get_project_folder(connectivity_index.parent_operation.project,
                                                            str(connectivity_index.fk_from_operation))
        loader = DirLoader(storage_path)
        conn_path = loader.path_for(ConnectivityH5, connectivity_index_gid)
        connectivity = Connectivity()
        with ConnectivityH5(conn_path) as conn_h5:
            conn_h5.load_into(connectivity)
        # TODO: handle this cases in a better manner
        session_stored_simulator.connectivity = connectivity
        session_stored_simulator.conduction_speed = conduction_speed
        session_stored_simulator.coupling = coupling()

        form = get_form_for_coupling(type(session_stored_simulator.coupling))()

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = form
        dict_to_render[self.ACTION_KEY] = '/burst/set_coupling_params'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_connectivity'
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_coupling_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        form = get_form_for_coupling(type(session_stored_simulator.coupling))()
        form.fill_from_post(data)

        form.fill_trait(session_stored_simulator.coupling)

        surface_fragment = SimulatorSurfaceFragment('', common.get_current_project().id)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = surface_fragment
        dict_to_render[self.ACTION_KEY] = '/burst/set_surface'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_coupling_params'
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_surface(self, **data):
        surface_index_gid = data['_surface']
        dict_to_render = copy.deepcopy(self.dict_to_render)

        if 'None' in surface_index_gid:
            stimuli_fragment = SimulatorStimulusFragment('', common.get_current_project().id, False)

            dict_to_render[self.FORM_KEY] = stimuli_fragment
            dict_to_render[self.ACTION_KEY] = '/burst/set_stimulus'
            dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_surface'
            return dict_to_render

        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        session_stored_simulator.is_surface_simulation = True
        surface_index = ABCAdapter.load_entity_by_gid(surface_index_gid)

        # TODO library: Correct Cortex usage in library
        surface = Cortex()
        session_stored_simulator.surface = surface

        rm_fragment = SimulatorRMFragment('', common.get_current_project().id, surface_index)
        dict_to_render[self.FORM_KEY] = rm_fragment
        dict_to_render[self.ACTION_KEY] = '/burst/set_cortex'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_surface'
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_cortex(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        rm_fragment = SimulatorRMFragment()
        rm_fragment.fill_from_post(data)

        session_stored_simulator.surface.coupling_strength = rm_fragment.coupling_strength.data

        lc_gid = rm_fragment.lc.value
        if lc_gid == 'None':
            lc_index = ABCAdapter.load_entity_by_gid(lc_gid)
            storage_path = self.files_helper.get_project_folder(lc_index.parent_operation.project,
                                                                str(lc_index.fk_from_operation))
            loader = DirLoader(storage_path)
            lc_path = loader.path_for(LocalConnectivityH5, lc_gid)
            lc = LocalConnectivity()
            with LocalConnectivityH5(lc_path) as lc_h5:
                lc_h5.load_into(lc)
            session_stored_simulator.surface.local_connectivity = lc

        rm_gid = rm_fragment.rm.value
        rm_index = ABCAdapter.load_entity_by_gid(rm_gid)
        storage_path = self.files_helper.get_project_folder(rm_index.parent_operation.project,
                                                            str(rm_index.fk_from_operation))
        loader = DirLoader(storage_path)
        rm_path = loader.path_for(RegionMappingH5, rm_gid)
        rm = RegionMapping()
        with RegionMappingH5(rm_path) as rm_h5:
            rm_h5.load_into(rm)
        session_stored_simulator.surface.region_mapping_data = rm

        model_fragment = SimulatorModelFragment('', common.get_current_project().id)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = model_fragment
        dict_to_render[self.ACTION_KEY] = '/burst/set_model'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_cortex'
        dict_to_render[self.IS_MODEL_FRAGMENT_KEY] = True
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_stimulus(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)

        # TODO: show surface stimulus if surface was prev chosen, otherwise, show region stimulus
        model_fragment = SimulatorModelFragment('', common.get_current_project().id)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = model_fragment
        dict_to_render[self.ACTION_KEY] = '/burst/set_model'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_stimulus'
        dict_to_render[self.IS_MODEL_FRAGMENT_KEY] = True
        # if session_stored_simulator.is_surface_simulation:
        dict_to_render[self.IS_SURFACE_SIMULATION_KEY] = True
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_model(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        form = SimulatorModelFragment()
        form.fill_from_post(data)

        session_stored_simulator.model = form.model.value()

        form = get_form_for_model(type(session_stored_simulator.model))()

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = form
        dict_to_render[self.ACTION_KEY] = '/burst/set_model_params'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_model'
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_model_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        form = get_form_for_model(type(session_stored_simulator.model))()
        form.fill_from_post(data)

        form.fill_trait(session_stored_simulator.model)

        integrator_fragment = SimulatorIntegratorFragment('', common.get_current_project().id)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = integrator_fragment
        dict_to_render[self.ACTION_KEY] = '/burst/set_integrator'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_model_params'
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
        fragment = SimulatorIntegratorFragment()
        fragment.fill_from_post(data)

        session_stored_simulator.integrator = fragment.integrator.value()

        form = get_form_for_integrator(type(session_stored_simulator.integrator))()

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = form
        dict_to_render[self.ACTION_KEY] = '/burst/set_integrator_params'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_integrator'
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_integrator_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        form = get_form_for_integrator(type(session_stored_simulator.integrator))()
        form.fill_from_post(data)

        form.fill_trait(session_stored_simulator.integrator)

        integrator_noise_fragment = get_form_for_noise(type(session_stored_simulator.integrator.noise))()

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = integrator_noise_fragment
        dict_to_render[self.ACTION_KEY] = '/burst/set_noise_params'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_integrator_params'
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_noise_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        form = get_form_for_noise(type(session_stored_simulator.integrator.noise))()
        form.fill_from_post(data)

        form.fill_trait(session_stored_simulator.integrator.noise)

        if isinstance(session_stored_simulator.integrator.noise, Additive):
            monitor_fragment = SimulatorMonitorFragment('', common.get_current_project().id)

            dict_to_render = copy.deepcopy(self.dict_to_render)
            dict_to_render[self.FORM_KEY] = monitor_fragment
            dict_to_render[self.ACTION_KEY] = '/burst/set_monitors'
            dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_noise_params'
            return dict_to_render

        equation_form = get_form_for_equation(type(session_stored_simulator.integrator.noise.b))()

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = equation_form
        dict_to_render[self.ACTION_KEY] = '/burst/set_noise_equation_params'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_noise_params'
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_noise_equation_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        form = get_form_for_equation(type(session_stored_simulator.integrator.noise.b))()
        form.fill_from_post(data)

        form.fill_trait(session_stored_simulator.integrator.noise.b)

        monitor_fragment = SimulatorMonitorFragment('', common.get_current_project().id)

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = monitor_fragment
        dict_to_render[self.ACTION_KEY] = '/burst/set_monitors'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_noise_equation_params'
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_monitors(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)

        # TODO: handle multiple monitors
        fragment = SimulatorMonitorFragment()
        fragment.fill_from_post(data)

        session_stored_simulator.monitors = [fragment.monitor.value()]
        monitor = session_stored_simulator.monitors[0]

        form = get_form_for_monitor(type(monitor))('', common.get_current_project().id)
        session_stored_simulator.monitors = [monitor]

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = form
        dict_to_render[self.ACTION_KEY] = '/burst/set_monitor_params'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_monitors'
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_monitor_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        monitor = session_stored_simulator.monitors[0]
        form = get_form_for_monitor(type(monitor))()
        form.fill_from_post(data)

        form.fill_trait(monitor)

        if isinstance(monitor, Bold):
            next_form = get_form_for_equation(type(monitor.hrf_kernel))()

            dict_to_render = copy.deepcopy(self.dict_to_render)
            dict_to_render[self.FORM_KEY] = next_form
            dict_to_render[self.ACTION_KEY] = '/burst/set_monitor_equation'
            dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_monitor_params'
            return dict_to_render
        session_stored_simulator.monitors = [monitor]

        next_form = SimulatorLengthFragment()

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = next_form
        dict_to_render[self.ACTION_KEY] = '/burst/set_simulation_length'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_monitor_params'
        dict_to_render[self.IS_LAST_FRAGMENT_KEY] = True
        return dict_to_render

    @cherrypy.expose
    @using_jinja_template("wizzard_form")
    @handle_error(redirect=False)
    @check_user
    def set_monitor_equation(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)

        form = get_form_for_monitor(type(session_stored_simulator.monitor))()
        form.fill_from_post(data)
        form.fill_trait(session_stored_simulator.monitor.hrf_kernel)

        next_form = SimulatorLengthFragment()

        dict_to_render = copy.deepcopy(self.dict_to_render)
        dict_to_render[self.FORM_KEY] = next_form
        dict_to_render[self.ACTION_KEY] = '/burst/set_simulation_length'
        dict_to_render[self.PREVIOUS_ACTION_KEY] = '/burst/set_monitor_equation'
        dict_to_render[self.IS_LAST_FRAGMENT_KEY] = True
        return dict_to_render

    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def set_simulation_length(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)

        fragment = SimulatorLengthFragment()
        fragment.fill_from_post(data)

        session_stored_simulator.simulation_length = fragment.length.value

        return 'Config is finished'

    @expose_json
    def launch_simulation(self, launch_mode, burst_name):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)

        project = common.get_current_project()
        user = common.get_logged_user()

        # TODO: handle operation - simulator H5 relation
        simulator_index = SimulatorIndex()
        simulator_index = dao.store_entity(simulator_index)

        burst_config = common.get_from_session(common.KEY_BURST_CONFIG)
        burst_config.start_time = datetime.datetime.now()
        if burst_name != 'none_undefined':
            burst_config.name = burst_name

        # TODO: branch simulation name is different
        if burst_config.name is None:
            new_id = dao.get_max_burst_id() + 1
            burst_config.name = 'simulation_' + str(new_id)

        dao.store_entity(burst_config)

        simulator_service = SimulatorService()

        simulator_id = self.cached_simulator_algorithm.id
        operation = simulator_service._prepare_operation(project.id, user.id, simulator_id, simulator_index)

        simulator_index.fk_from_operation = operation.id
        dao.store_entity(simulator_index)

        storage_path = self.files_helper.get_project_folder(project, str(operation.id))

        simulator_gid = simulator_service.serialize_simulator(session_stored_simulator, simulator_index.gid,
                                                              storage_path)

        OperationService().launch_operation(operation.id, True)

    @expose_fragment('burst/burst_history')
    def load_burst_history(self):
        """
        Load the available burst that are stored in the database at this time.
        This is one alternative to 'chrome-back problem'.
        """
        session_burst = common.get_from_session(common.KEY_BURST_CONFIG)
        bursts = self.burst_service.get_available_bursts(common.get_current_project().id)
        self.burst_service.populate_burst_disk_usage(bursts)
        return {'burst_list': bursts,
                'selectedBurst': 1}

    @cherrypy.expose
    @handle_error(redirect=False)
    def copy_simulator_configuration(self, burst_config_id):
        # TODO:
        # 1. load burst by burst_config_id
        # 2. load simulator index
        # 3. deserialize simulator from H5
        # 4. add simulator to session
        burst_config = dao.get_burst_by_id(burst_config_id)
        simulator_index_id = burst_config.simulator_id
        simulator_index = dao.get_datatype_by_id(simulator_index_id)
        simulator_gid = simulator_index.gid

        project = common.get_current_project()
        storage_path = self.files_helper.get_project_folder(project, str(simulator_index.fk_from_operation))

        simulator_service = SimulatorService()
        simulator, = simulator_service.deserialize_simulator(simulator_gid, storage_path)

        session_stored_burst = simulator
        common.add2session(common.KEY_SIMULATOR_CONFIG, session_stored_burst)
        pass

    @cherrypy.expose
    def generate_form_for_copied_simulator_configuratin(self):
        # TODO for each simulator attribute:
        # 1. find the corresponding form
        # 2. fill it with proper values
        # 3. make fields read-only
        pass
