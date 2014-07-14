# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import json
import numpy
from tvb.adapters.visualizers.phase_plane_interactive import PhasePlaneInteractive
from tvb.basic.traits import core, types_basic, traited_interface
from tvb.basic.traits.parameters_factory import get_traited_subclasses
from tvb.basic.traits.util import multiline_math_directives_to_matjax
from tvb.core import utils
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.storage import dao
import tvb.core.entities.model
from tvb.datatypes import noise_framework
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.decorators import expose_page, expose_json, expose_fragment, profile_func
from tvb.simulator import models, integrators


class Dynamic(object):
    """
    Groups a model and an integrator.
    """
    def __init__(self, model=None, integrator=None):
        if model is None:
            model = models.Generic2dOscillator()
        if integrator is None:
            integrator = integrators.HeunDeterministic()
        self.model = model
        self.integrator = integrator

        # The phase plane holds a global reference to a mplh5 figure
        # Only one instance should exist for a browser page.
        # To achieve something close to that we store it here
        self.phase_plane = PhasePlaneInteractive(model, integrator)


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


class _InputTreeFragment(core.Type):
    """
    This trait-ed class is used to build the input tree for the integrator.
    """
    dynamic_name = types_basic.String(
        label = "Dynamic name",
        required = True,
        order=1,
        doc = """The name of the dynamic""")

    integrator = integrators.Integrator(
        label = "integrator",
        required = True,
        order=2,
        default = integrators.HeunDeterministic,
        doc = """The integrator"""
        )


class _LeftFragmentAdapter(ABCAdapter):
    """
    This adapter is used only to generate the left input tree.
    """
    def __init__(self, available_models):
        ABCAdapter.__init__(self)
        self.available_models = available_models

    def launch(self):
        pass

    def get_output(self):
        pass

    def get_required_memory_size(self):
        return -1

    def get_required_disk_size(self):
        return 0

    def get_input_tree(self):
        models_sub_tree = {
            'name': 'model_type', 'type': 'select', 'required': True, 'label': 'model',
            'default' : 'Generic2dOscillator',
            'options': []
        }

        for clz_name, clz in self.available_models.iteritems():
            models_sub_tree['options'].append({
                'name': clz._ui_name, # ui-name instaead
                'value': clz_name,
                'inline_description': self._dfun_math_directives_to_matjax(clz),
                'description' : clz.__doc__
            })

        fragment = _InputTreeFragment()
        fragment.trait.bound = traited_interface.INTERFACE_ATTRIBUTES_ONLY

        input_tree = fragment.interface[traited_interface.INTERFACE_ATTRIBUTES]
        input_tree.insert(1, models_sub_tree)
        #self.log.warn(json.dumps(input_tree, indent=2, sort_keys=1))
        return input_tree


    @staticmethod
    def _dfun_math_directives_to_matjax(model):
        """
        Looks for sphinx math directives if the docstring of the dfun function of a model.
        It converts them in html text that will be interpreted by mathjax
        The parsing is simplistic, not a full rst parser.
        """
        dfun = getattr(model, 'dfun', None)

        if dfun:
            return multiline_math_directives_to_matjax(dfun.__doc__).replace('&', '&amp;').replace('.. math::','')
        else:
            return ''



class DynamicModelController(BurstBaseController):
    KEY_CACHED_DYNAMIC_MODEL = 'cache.DynamicModelController'

    def __init__(self):
        BurstBaseController.__init__(self)
        self.available_models = get_traited_subclasses(models.Model)
        self.available_integrators = get_traited_subclasses(integrators.Integrator)
        self.cache = SessionCache()

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
        adapter = _LeftFragmentAdapter(self.available_models)
        input_tree = adapter.get_input_tree()
        #WARN: If this input tree will contain data type references then to render it correctly we have to use flow_service.prepare_parameters
        input_tree = adapter.prepare_param_names(input_tree)

        params = {
            'title': "Dynamic model",
            'mainContent': 'burst/dynamic',
            'input_tree': input_tree,
            'dynamic_gid': dynamic_gid
        }
        self.fill_default_attributes(params)

        dynamic = self.get_cached_dynamic(dynamic_gid)
        params.update(dynamic.phase_plane.draw_phase_plane())
        return params


    def fill_default_attributes(self, param):
        BurstBaseController.fill_default_attributes(self, param, subsection='dynamic')


    @expose_json
    def model_changed(self, dynamic_gid, name):
        dynamic = self.get_cached_dynamic(dynamic_gid)
        dynamic.model = self.available_models[name]()

        dynamic.phase_plane.reset(dynamic.model, dynamic.integrator)
        return self.slider_options(dynamic.model)


    @expose_json
    def integrator_changed(self, dynamic_gid, **kwargs):
        adapter = _LeftFragmentAdapter(self.available_models)
        tree = adapter.convert_ui_inputs(kwargs, validation_required=False)
        integrator_name = tree['integrator']
        integrator_parameters = tree['integrator_parameters']

        noise_framework.build_noise(integrator_parameters)
        integrator = self.available_integrators[integrator_name](**integrator_parameters)

        dynamic = self.get_cached_dynamic(dynamic_gid)
        dynamic.integrator = integrator
        dynamic.model.integrator = integrator
        dynamic.model.configure()
        self._configure_integrator_noise(integrator, dynamic.model)

        dynamic.phase_plane.reset(dynamic.model, dynamic.integrator)


    @staticmethod
    def _configure_integrator_noise(integrator, model):
        """
        This function has to be called after integrator construction.
        Without it the noise instance is not in a good state.

        Should the Integrator __init__ not take care of this? Or noise_framework.buildnoise?
        Should I call noise.configure() as well?
        """
        #todo: When refactoring phase_plane_interactive.py let it use this.

        if isinstance(integrator, integrators.IntegratorStochastic):
            shape = (1, model.nvar, 1, model.number_of_modes)
            if integrator.noise.ntau > 0.0:
                integrator.noise.configure_coloured(integrator.dt, shape)
            else:
                integrator.noise.configure_white(integrator.dt, shape)


    @expose_json
    def parameter_changed(self, dynamic_gid, name, value):
        dynamic = self.get_cached_dynamic(dynamic_gid)
        model = dynamic.model
        setattr(model, name, numpy.array([float(value)]))
        model.configure()
        dynamic.phase_plane.refresh()


    @expose_fragment('burst/dynamic_sliders')
    def sliders_fragment(self, dynamic_gid):
        dynamic = self.get_cached_dynamic(dynamic_gid)
        model = dynamic.model
        ret = []
        for name in model.ui_configurable_parameters:
            trait_kwd = model.trait[name].trait.inits.kwd
            ret.append({
                'name': name,
                'label': trait_kwd.get('label'),
                'default': trait_kwd.get('default'),
                'description': trait_kwd.get('doc'),
            })
        return {'parameters' : ret, 'showOnlineHelp' : True}

    @staticmethod
    def slider_options(model):
        ret = []
        for name in model.ui_configurable_parameters:
            parameter_trait = model.trait[name].trait
            ranger = parameter_trait.range_interval
            default = float(parameter_trait.inits.kwd.get('default'))

            ret.append({
                'name': name,
                'min': ranger.lo,
                'max': ranger.hi,
                'step': ranger.step,
                'default': default
            })
        return {'options' : ret}


    @expose_json
    def submit(self, dynamic_gid, dynamic_name):
        dynamic = self.get_cached_dynamic(dynamic_gid)
        model = dynamic.model
        integrator = dynamic.integrator

        model_parameters = []

        for name in model.ui_configurable_parameters:
            value = getattr(model, name).tolist()
            model_parameters.append((name, value))

        entity = tvb.core.entities.model.Dynamic(
            dynamic_name,
            common.get_logged_user().id,
            model.__class__.__name__,
            json.dumps(model_parameters),
            integrator.__class__.__name__,
            None
            # todo: serialize integrator parameters
            # json.dumps(integrator.raw_ui_integrator_parameters)
        )

        dao.store_entity(entity)


    @staticmethod
    def group_parameter_values_by_name(model_parameters_list):
        """
        Given a list of model parameters like this:
            ["a", [2.0]], ['b', [1.0]
            ["a", [3.0]], ['b', [7.0]
        Group them by param name to get:
        {'a': [2.0, 3.0], 'b': [1.0, 7.0]}
        """
        ret = {}
        for model_parameters in model_parameters_list:
            for param_name, param_vals in model_parameters:
                if param_name not in ret:
                    ret[param_name] = []
                ret[param_name].extend(param_vals)
        return ret


    @expose_fragment('burst/dynamic_minidetail')
    def dynamic_detail(self, dynamic_id):
        dynamic = dao.get_dynamic(dynamic_id)
        model_parameters = self.group_parameter_values_by_name([json.loads(dynamic.model_parameters)])
        return {'model_parameters': model_parameters}