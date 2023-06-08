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
Change for TVB version 2.0.9.

.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""
from tvb.basic.config.stored import KEY_ADMIN_DISPLAY_NAME, KEY_ENABLE_KC_LOGIN, KEY_KC_WEB_CONFIGURATION, \
    KEY_KC_CONFIGURATION
from tvb.basic.profile import TvbProfile


def update():
    """
    Add new parameters to the tvb.configuration file and delete 'URL_WEB'
    """

    new_stored_settings = {KEY_ADMIN_DISPLAY_NAME: 'Administrator', KEY_ENABLE_KC_LOGIN: False,
                           KEY_KC_WEB_CONFIGURATION: '', KEY_KC_CONFIGURATION: 'add_keycloak_path_here'}
    manager = TvbProfile.current.manager
    manager.add_entries_to_config_file(new_stored_settings)
    manager.delete_entries_from_config_file(['URL_WEB'])

