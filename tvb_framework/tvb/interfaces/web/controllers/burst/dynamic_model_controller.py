# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import json
import threading
import cherrypy
import numpy

import tvb.core.entities.model.model_burst as model_burst
from tvb.adapters.forms.equation_forms import get_form_for_equation, TemporalEquationsEnum
from tvb.adapters.forms.integrator_forms import get_form_for_integrator, NoiseTypesEnum
from tvb.adapters.forms.model_forms import get_form_for_model, ModelsEnum
from tvb.adapters.forms.noise_forms import get_form_for_noise
from tvb.adapters.forms.simulator_fragments import SimulatorModelFragment, SimulatorIntegratorFragment
from tvb.adapters.visualizers.phase_plane_interactive import phase_space_d3
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import TVBEnum
from tvb.core import utils
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.entities.file.simulator.view_model import HeunDeterministicViewModel, IntegratorStochasticViewModel, \
    IntegratorViewModelsEnum
from tvb.core.entities.storage import dao
from tvb.core.neotraits.forms import StrField
from tvb.core.neotraits.view_model import Str
from tvb.core.utils import TVBJSONEncoder
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.burst.matjax import configure_matjax_doc
from tvb.interfaces.web.controllers.decorators import expose_page, expose_json, expose_fragment, using_template, \
    handle_error, check_user
from tvb.simulator import models


class Dynamic(object):
    """
    Groups a model and an integrator.
    """

    def __init__(self, model=None, integrator=None):
        if model is None:
            model = models.Generic2dOscillator()
        if integrator is None:
            integrator = HeunDeterministicViewModel()

        model.configure()
        self.model = model
        self.integrator = integrator

        # Only one instance should exist for a browser page.
        # To achieve something close to that we store it here
        self.phase_plane = phase_space_d3(model, integrator)


class SessionCache(object):
    """
    A simple cache backed by the current cherrypy session.
    It does not expire it's contents.
    """
    SESSION_KEY = 'session_cache'

    @property
    def _cache(self):
        cache = common.get_from_session(self.SESSION_KEY)
        if cache is None:
            cache = {}
            common.add2session(self.SESSION_KEY, cache)
        return cache

    def __contains__(self, key):
        return key in self._cache

    def __getitem__(self, key):
        return self._cache[key]

    def __setitem__(self, key, value):
        self._cache[key] = value


class _InputTreeFragment(ABCAdapterForm):
    def __init__(self):
        super(_InputTreeFragment, self).__init__()
        self.dynamic_name = StrField(Str(label='Parameter configuration name',
                                         doc="""The name of this parameter configuration"""), name='dynamic_name')


@traced('fill_default_attributes', exclude=True)
class DynamicModelController(BurstBaseController):
    KEY_CACHED_DYNAMIC_MODEL = 'cache.DynamicModelController'
    LOGGER = get_logger(__name__)

    def __init__(self):
        BurstBaseController.__init__(self)
        self.cache = SessionCache()
        # Work around a numexpr thread safety issue. See TVB-1639.
        self.traj_lock = threading.Lock()

    def get_cached_dynamic(self, dynamic_gid):
        """
        Creating the model per request will be expensive.
        So we cache it in session.
        If there is nothing cached it returns the default dynamic.
        """
        # TODO: The cached objects expire only with the session. Invalidate the cache earlier.
        if dynamic_gid not in self.cache:
            dynamic = Dynamic()
            self.cache[dynamic_gid] = dynamic
        return self.cache[dynamic_gid]

    @expose_page
    def index(self):
        dynamic_gid = utils.generate_guid()
        model_name_fragment = _InputTreeFragment()
        model_fragment = self.algorithm_service.prepare_adapter_form(form_instance=SimulatorModelFragment())
        integrator_fragment = self.algorithm_service.prepare_adapter_form(form_instance=SimulatorIntegratorFragment())
        model_description = configure_matjax_doc()

        params = {
            'title': "Dynamic model",
            'mainContent': 'burst/dynamic',
            'model_name_fragment': self.render_adapter_form(model_name_fragment),
            'model_form': self.render_adapter_form(model_fragment),
            'integrator_form': self.render_adapter_form(integrator_fragment),
            'dynamic_gid': dynamic_gid,
            'model_description': model_description
        }
        self.fill_default_attributes(params)

        dynamic = self.get_cached_dynamic(dynamic_gid)
        self._configure_integrator_noise(dynamic.integrator, dynamic.model)
        return params

    def fill_default_attributes(self, param):
        return BurstBaseController.fill_default_attributes(self, param, subsection='phaseplane')

    @expose_json
    def model_changed(self, dynamic_gid, name):
        """
        Resets the phase plane and returns the ui model for the slider area.
        """
        dynamic = self.get_cached_dynamic(dynamic_gid)
        dynamic.model = TVBEnum.string_to_enum(list(ModelsEnum), name).instance
        dynamic.model.configure()
        self._configure_integrator_noise(dynamic.integrator, dynamic.model)
        dynamic.phase_plane = phase_space_d3(dynamic.model, dynamic.integrator)
        mp_params = DynamicModelController._get_model_parameters_ui_model(dynamic.model)
        graph_params = DynamicModelController._get_graph_ui_model(dynamic)
        return {
            'params': mp_params, 'graph_params': graph_params,
            'model_param_sliders_fragment': self._model_param_sliders_fragment(dynamic_gid),
            'axis_sliders_fragment': self._axis_sliders_fragment(dynamic_gid)
        }

    def _update_integrator(self, dynamic, integrator):
        self._configure_integrator_noise(integrator, dynamic.model)
        dynamic.integrator = integrator
        dynamic.phase_plane.integrator = integrator

    def _change_integrator(self, dynamic, field_value):
        integrator = TVBEnum.string_to_enum(list(IntegratorViewModelsEnum), field_value).instance
        self._update_integrator(dynamic, integrator)

    def _change_noise(self, dynamic, field_value):
        noise = TVBEnum.string_to_enum(list(NoiseTypesEnum), field_value).instance
        integrator = dynamic.integrator
        integrator.noise = noise
        self._update_integrator(dynamic, integrator)

    def _change_equation(self, dynamic, field_value):
        equation = TVBEnum.string_to_enum(list(TemporalEquationsEnum), field_value).instance
        integrator = dynamic.integrator
        integrator.noise.b = equation
        self._update_integrator(dynamic, integrator)

    def _update_integrator_on_dynamic(self, dynamic_gid, field_value):
        dynamic = self.get_cached_dynamic(dynamic_gid)
        if field_value in list(map(str, IntegratorViewModelsEnum)):
            self._change_integrator(dynamic, field_value)
            return IntegratorViewModelsEnum, get_form_for_integrator
        elif field_value in list(map(str, NoiseTypesEnum)):
            self._change_noise(dynamic, field_value)
            return NoiseTypesEnum, get_form_for_noise
        elif field_value in list(map(str, TemporalEquationsEnum)):
            self._change_equation(dynamic, field_value)
            return TemporalEquationsEnum, get_form_for_equation

    @cherrypy.expose
    @using_template('form_fields/form_field')
    @handle_error(redirect=False)
    @check_user
    def refresh_subform(self, dynamic_gid, field_value):
        enum_class, get_form_method = self._update_integrator_on_dynamic(dynamic_gid, field_value)
        integrator_class = TVBEnum.string_to_enum(list(enum_class), field_value).value
        subform = get_form_method(integrator_class)()
        return {'adapter_form': subform}

    @cherrypy.expose
    def integrator_parameters_changed(self, dynamic_gid, type, **param):
        dynamic = self.get_cached_dynamic(dynamic_gid)
        integrator = dynamic.integrator
        changed_params = list(param.keys())
        if type == 'INTEGRATOR':
            integrator_form_class = get_form_for_integrator(integrator.__class__)
            integrator_form = integrator_form_class()
            integrator_form.fill_from_post(param)
            integrator_form.fill_trait_partially(integrator, changed_params)
        if type == 'NOISE':
            noise_form_class = get_form_for_noise(integrator.noise.__class__)
            noise_form = noise_form_class()
            noise_form.fill_from_post(param)
            noise_form.fill_trait_partially(integrator.noise, changed_params)
        if type == 'EQUATION':
            eq_form_class = get_form_for_equation(integrator.noise.b.__class__)
            eq_form = eq_form_class()
            eq_form.fill_from_post(param)
            eq_form.fill_trait_partially(integrator.noise.b, changed_params)
        self._update_integrator(dynamic, integrator)

    @staticmethod
    def _configure_integrator_noise(integrator, model):
        """
        This function has to be called after integrator construction.
        Without it the noise instance is not in a good state.

        Should the Integrator __init__ not take care of this? Or noise_framework.buildnoise?
        Should I call noise.configure() as well?
        similar to simulator.configure_integrator_noise
        """
        if isinstance(integrator, IntegratorStochasticViewModel):
            shape = (model.nvar, 1, model.number_of_modes)
            integrator.noise.reset_random_stream()
            if integrator.noise.ntau > 0.0:
                integrator.noise.configure_coloured(integrator.dt, shape)
            else:
                integrator.noise.configure_white(integrator.dt, shape)

    @expose_json
    def parameters_changed(self, dynamic_gid, params):
        with self.traj_lock:
            params = json.loads(params)
            dynamic = self.get_cached_dynamic(dynamic_gid)
            model = dynamic.model
            for name, value in params.items():
                param_type = float
                if numpy.issubdtype(getattr(model, name).dtype, numpy.integer):
                    param_type = int
                setattr(model, name, numpy.array([param_type(value)]))
            model.configure()
            return dynamic.phase_plane.compute_phase_plane()

    @expose_json
    def graph_changed(self, dynamic_gid, graph_state):
        with self.traj_lock:
            graph_state = json.loads(graph_state)
            dynamic = self.get_cached_dynamic(dynamic_gid)
            self._configure_integrator_noise(dynamic.integrator, dynamic.model)
            dynamic.phase_plane.update_axis(**graph_state)
            return dynamic.phase_plane.compute_phase_plane()

    @expose_json
    def trajectories(self, dynamic_gid, starting_points, integration_steps):
        with self.traj_lock:
            starting_points = json.loads(starting_points)
            dynamic = self.get_cached_dynamic(dynamic_gid)
            trajectories, signals = dynamic.phase_plane.trajectories(starting_points, int(integration_steps))

            for t in trajectories:
                if not numpy.isfinite(t).all():
                    self.logger.warning('Denaturated point %s on a trajectory')
                    return {'finite': False}

            return {'trajectories': trajectories, 'signals': signals, 'finite': True}

    @staticmethod
    def _get_model_parameters_ui_model(model):
        """
        For each model parameter return the representation used by the ui (template & js)
        """
        ret = []
        model_form_class = get_form_for_model(type(model))
        for param in model_form_class().get_params_configurable_in_phase_plane():
            attr = getattr(type(model), param.name)
            ranger = attr.domain
            if ranger is None:
                DynamicModelController.LOGGER.warning("Param %s doesn't have a domain specified" % param.name)
                continue
            default = float(attr.default)

            ret.append({
                'name': param.name,
                'label': attr.label,
                'description': attr.doc,
                'min': ranger.lo,
                'max': ranger.hi,
                'step': ranger.step,
                'default': default
            })
        return ret

    @staticmethod
    def _get_graph_ui_model(dynamic):
        model = dynamic.model
        sv_model = []
        for sv in range(model.nvar):
            name = model.state_variables[sv]
            min_val, max_val, lo, hi = dynamic.phase_plane.get_axes_ranges(name)
            sv_model.append({
                'name': name,
                'label': ':math:`%s`' % name,
                'description': 'state variable ' + name,
                'lo': lo,
                'hi': hi,
                'min': min_val,
                'max': max_val,
                'step': (hi - lo) / 1000.0,  # todo check if reasonable
                'default': (hi + lo) / 2
            })

        ret = {
            'modes': list(range(model.number_of_modes)),
            'state_variables': sv_model,
            'default_mode': dynamic.phase_plane.mode
        }

        if model.nvar > 1:
            ret['default_sv'] = [model.state_variables[dynamic.phase_plane.svx_ind],
                                 model.state_variables[dynamic.phase_plane.svy_ind]]
            ret['integration_steps'] = {'default': 512, 'min': 32, 'max': 2048}
        else:
            ret['default_sv'] = [model.state_variables[0]]
        return ret

    @using_template('burst/dynamic_axis_sliders')
    def _axis_sliders_fragment(self, dynamic_gid):
        dynamic = self.get_cached_dynamic(dynamic_gid)
        model = dynamic.model
        ps_params = self._get_graph_ui_model(dynamic)
        templ_var = ps_params
        templ_var.update({'showOnlineHelp': True,
                          'one_dimensional': len(model.state_variables) == 1})
        return templ_var

    @using_template('burst/dynamic_mp_sliders')
    def _model_param_sliders_fragment(self, dynamic_gid):
        dynamic = self.get_cached_dynamic(dynamic_gid)
        model = dynamic.model
        mp_params = self._get_model_parameters_ui_model(model)
        templ_var = {'parameters': mp_params, 'showOnlineHelp': True}
        return templ_var

    @expose_json
    def submit(self, dynamic_gid, dynamic_name):
        if dao.get_dynamic_by_name(dynamic_name):
            return {'saved': False, 'msg': 'There is another configuration with the same name'}

        dynamic = self.get_cached_dynamic(dynamic_gid)
        model = dynamic.model
        integrator = dynamic.integrator

        model_parameters = []

        model_form_class = get_form_for_model(type(model))
        for param in model_form_class().get_params_configurable_in_phase_plane():
            value = getattr(model, param.name)[0]
            model_parameters.append((param.name, value))

        entity = model_burst.Dynamic(
            dynamic_name,
            common.get_logged_user().id,
            model.__class__.__name__,
            json.dumps(model_parameters, cls=TVBJSONEncoder),
            integrator.__class__.__name__,
            None
            # todo: serialize integrator parameters
            # json.dumps(integrator.raw_ui_integrator_parameters)
        )

        dao.store_entity(entity)
        return {'saved': True}

    @expose_fragment('burst/dynamic_minidetail')
    def dynamic_detail(self, dynamic_id):
        dynamic = dao.get_dynamic(dynamic_id)
        model_parameters = dict(json.loads(dynamic.model_parameters))
        return {'model_parameters': model_parameters}
