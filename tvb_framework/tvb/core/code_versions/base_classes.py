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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
import importlib
import os
from tvb.basic.logger.builder import get_logger
from tvb.basic.exceptions import TVBException


class InvalidUpgradeScriptException(TVBException):
    """
    Raised in case an update script is present but does not comply to TVB conventions.
    """
    pass


class UpdateManager(object):
    """
    An update manager pattern.
    Goes through all the scripts, and based on the current version executed the one in need
    """

    def __init__(self, module_scripts, check_version, current_version):
        self.log = get_logger(self.__class__.__module__)
        self.update_scripts_module = module_scripts
        self.checked_version = check_version
        self.current_version = current_version

    def get_update_scripts(self, checked_version=None):
        """
        Return all update scripts that need to be executed in order to bring code up to date.
        """
        unprocessed_scripts = []
        if checked_version is None:
            checked_version = self.checked_version

        for file_n in os.listdir(os.path.dirname(self.update_scripts_module.__file__)):
            if '_update' in file_n and file_n.endswith('.py'):
                version_nr = int(file_n.split('_')[0])
                if version_nr > checked_version:
                    unprocessed_scripts.append(file_n)

        result = sorted(unprocessed_scripts, key=lambda x: int(x.split('_')[0]))
        if result:
            self.log.info("Found unprocessed update scripts: %s." % result)
        return result

    def run_update_script(self, script_name, **kwargs):
        """
        Run one script file.
        """
        script_module_name = self.update_scripts_module.__name__ + '.' + script_name.split('.')[0]
        script_module = importlib.import_module(script_module_name)

        if not hasattr(script_module, 'update'):
            raise InvalidUpgradeScriptException("Code update scripts should expose a 'update()' method.")

        script_module.update(**kwargs)

    def run_all_updates(self, **kwargs):
        """
        Upgrade the code to current version. 
        """
        if self.checked_version < self.current_version:
            for script_name in self.get_update_scripts():
                self.log.info("We will run update %s" % script_name)
                self.run_update_script(script_name, **kwargs)
