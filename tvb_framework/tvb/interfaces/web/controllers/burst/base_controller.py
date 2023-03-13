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

from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.structure import WebStructure


class BurstBaseController(BaseController):

    def fill_default_attributes(self, template_dictionary, subsection='burst'):

        template_dictionary[common.KEY_PARAMETERS_CONFIG] = False
        template_dictionary[common.KEY_SECTION] = WebStructure.SECTION_BURST
        template_dictionary[common.KEY_SUBMENU_LIST] = self.burst_submenu

        if common.KEY_SUB_SECTION not in template_dictionary:
            template_dictionary[common.KEY_SUB_SECTION] = subsection

        BaseController.fill_default_attributes(self, template_dictionary)
        return template_dictionary