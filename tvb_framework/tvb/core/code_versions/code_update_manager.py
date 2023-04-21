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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from tvb.basic.profile import TvbProfile
from tvb.basic.config import stored
from tvb.core.code_versions.base_classes import UpdateManager
import tvb.core.code_versions.code_update_scripts as code_versions


class CodeUpdateManager(UpdateManager):
    """
    A manager that goes through all the scripts that are newer than the version number 
    written in the .tvb.basic.config.setting configuration file.
    """

    def __init__(self):
        super(CodeUpdateManager, self).__init__(code_versions, TvbProfile.current.version.CODE_CHECKED_TO_VERSION,
                                                TvbProfile.current.version.REVISION_NUMBER)

    def run_update_script(self, script_name):
        """
        Add specific code after every update script.
        """
        super(CodeUpdateManager, self).run_update_script(script_name)
        # After each update mark the update in cfg file. 
        # In case one update script fails, the ones before will not be repeated.
        TvbProfile.current.manager.add_entries_to_config_file(
            {stored.KEY_LAST_CHECKED_CODE_VERSION: script_name.split('_')[0]})

    def run_all_updates(self):
        """
        Upgrade the code to current version. 
        Go through all update scripts with lower SVN version than the current running version.
        """
        if TvbProfile.is_first_run():
            # We've just started with a clean TVB. No need to upgrade anything.
            return

        super(CodeUpdateManager, self).run_all_updates()

        if self.checked_version < self.current_version:
            TvbProfile.current.manager.add_entries_to_config_file(
                {stored.KEY_LAST_CHECKED_CODE_VERSION: TvbProfile.current.version.REVISION_NUMBER})
