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
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import urllib
import cherrypy
from tvb.config import DISCRETE_PSE_ADAPTER_MODULE, DISCRETE_PSE_ADAPTER_CLASS
from tvb.config import ISOCLINE_PSE_ADAPTER_CLASS, ISOCLINE_PSE_ADAPTER_MODULE
from tvb.core.services.project_service import ProjectService
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.transient.filtering import FilterChain
from tvb.interfaces.web.controllers.decorators import handle_error, expose_fragment, check_user, using_template
from tvb.interfaces.web.controllers.base_controller import BaseController


PSE_FLOT = "FLOT"
PSE_ISO = "ISO"

REDIRECT_MSG = '/burst/explore/pse_error?adapter_name=%s&message=%s'



class ParameterExplorationController(BaseController):
    """
    Controller to handle PSE actions.
    """


    def __init__(self):
        BaseController.__init__(self)
        self.project_service = ProjectService()


    @cherrypy.expose
    @handle_error(redirect=False)
    def get_default_pse_viewer(self, datatype_group_gid):
        """
        For a given DataTypeGroup, check first if the discrete PSE is compatible.
        If this is not the case fallback to the continous PSE viewer.
        If none are available return: None.
        """
        algorithm = self.flow_service.get_algorithm_by_module_and_class(DISCRETE_PSE_ADAPTER_MODULE,
                                                                        DISCRETE_PSE_ADAPTER_CLASS)[0]
        if self._is_compatible(algorithm, datatype_group_gid):
            return PSE_FLOT

        algorithm = self.flow_service.get_algorithm_by_module_and_class(ISOCLINE_PSE_ADAPTER_MODULE,
                                                                        ISOCLINE_PSE_ADAPTER_CLASS)[0]
        if self._is_compatible(algorithm, datatype_group_gid):
            return PSE_ISO

        return None


    def _is_compatible(self, algorithm, datatype_group_gid):
        """
        Check if PSE view filters are compatible with current DataType.
        :param algorithm: Algorithm instance to get filters from it.
        :param datatype_group_gid: Current DataTypeGroup to validate against.
        :returns: True when DataTypeGroup can be displayed with current algorithm, False when incompatible.
        """
        datatype_group = ABCAdapter.load_entity_by_gid(datatype_group_gid)
        filter_chain = FilterChain.from_json(algorithm.datatype_filter)
        if datatype_group and (not filter_chain or filter_chain.get_python_filter_equivalent(datatype_group)):
            return True
        return False


    @cherrypy.expose
    @handle_error(redirect=True)
    @using_template('visualizers/pse_discrete/burst_preview')
    @check_user
    def draw_discrete_exploration(self, datatype_group_gid, back_page, color_metric=None, size_metric=None):
        """
        Create new data for when the user chooses to refresh from the UI.
        """
        if color_metric == 'None':
            color_metric = None
        if size_metric == 'None':
            size_metric = None

        algorithm, group = self.flow_service.get_algorithm_by_module_and_class(DISCRETE_PSE_ADAPTER_MODULE,
                                                                               DISCRETE_PSE_ADAPTER_CLASS)
        adapter = self.flow_service.build_adapter_instance(group)
        if self._is_compatible(algorithm, datatype_group_gid):
            try:
                pse_context = adapter.prepare_parameters(datatype_group_gid, back_page, color_metric, size_metric)
                pse_context.prepare_individual_jsons()
                return pse_context
            except LaunchException, ex:
                error_msg = urllib.quote(ex.message)
        else:
            error_msg = urllib.quote("Discrete PSE is incompatible (most probably due to result size being too large).")

        name = urllib.quote(adapter._ui_name)
        raise cherrypy.HTTPRedirect(REDIRECT_MSG % (name, error_msg))


    @cherrypy.expose
    @handle_error(redirect=True)
    @using_template('visualizers/pse_isocline/burst_preview')
    @check_user
    def draw_isocline_exploration(self, datatype_group_gid, width=None, height=None):

        if width is not None:
            width = int(width)
        if height is not None:
            height = int(height)

        algorithm, group = self.flow_service.get_algorithm_by_module_and_class(ISOCLINE_PSE_ADAPTER_MODULE,
                                                                               ISOCLINE_PSE_ADAPTER_CLASS)
        adapter = self.flow_service.build_adapter_instance(group)
        if self._is_compatible(algorithm, datatype_group_gid):
            try:
                return adapter.burst_preview(datatype_group_gid, width, height)
            except LaunchException, ex:
                self.logger.error(ex.message)
                error_msg = urllib.quote(ex.message)
        else:
            error_msg = urllib.quote("Isocline PSE requires a 2D range of floating point values.")

        name = urllib.quote(adapter._ui_name)
        raise cherrypy.HTTPRedirect(REDIRECT_MSG % (name, error_msg))


    @expose_fragment('burst/burst_pse_error')
    def pse_error(self, adapter_name, message):
        return {'adapter_name': adapter_name, 'message': message}

