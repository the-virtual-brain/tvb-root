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
Used by FlowController.
It will store in current user's session information about 

.. moduleauthor:: Adrian Dordea <adrian.dordea@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from tvb.interfaces.web.controllers import common


class SelectedAdapterContext(object):
    """
    Responsible for storing/retrieving/removing from session info about currently selected algorithm.
    """

    KEY_VIEW_MODEL = "viewModel"

    def add_view_model_to_session(self, view_model):
        """
        Put in session information about the view_model
        """
        common.add2session(self.KEY_VIEW_MODEL, view_model)

    def get_view_model_from_session(self):
        """
        Get view_model from session
        """
        return common.get_from_session(self.KEY_VIEW_MODEL)

    def clean_from_session(self):
        """
        Remove info about selected algo from session
        """
        common.remove_from_session(self.KEY_VIEW_MODEL)
