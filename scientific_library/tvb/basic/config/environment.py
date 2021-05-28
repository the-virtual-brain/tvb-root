# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
Environment related checks or operations are to be defined here.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import os
import sys
from subprocess import Popen, PIPE
from tvb.basic.config.settings import VersionSettings


class Environment(object):
    PYTHON_FOLDER = "python%s.%s" % sys.version_info[:2]

    def is_framework_present(self):
        """
        :return: True when framework classes are present and can be imported.
        """
        framework_present = True
        try:
            from tvb.config.profile_settings import WebSettingsProfile
        except ImportError:
            framework_present = False

        return framework_present

    @staticmethod
    def is_distribution():
        """
        Return True when TVB_Distribution package, False when used from a GitHub clone, Pypi, Conda or Docker
        """

        try:
            import tvb_bin
        except ImportError:
            # No tvb_bin, it means usage from Pypi or Conda Forge
            return False

        try:
            _proc = Popen(["git", "status"], stdout=PIPE, stderr=PIPE)
            if "On branch " in str(_proc.communicate()):
                # usage from GitHub clone directly
                return False
        except Exception:
            pass

        externals_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(tvb_bin.__file__)))),
            "externals")
        if os.path.exists(externals_path):
            # usage from GitHub clone without got cmd or inside a Docker container (as a mounted volume)
            return False

        # We default as usage from TVB_Distribution
        return True

    def is_linux_deployment(self):
        """
        Return True if current run is not development and is running on Linux.
        """
        return self.is_linux() and self.is_distribution()

    def is_mac_deployment(self):
        """
        Return True if current run is not development and is running on Mac OS X
        """
        return self.is_mac() and self.is_distribution()

    def is_windows_deployment(self):
        """
        Return True if current run is not development and is running on Windows.
        """
        return self.is_windows() and self.is_distribution()

    def is_linux(self):
        return not self.is_windows() and not self.is_mac()

    @staticmethod
    def is_mac():
        return sys.platform == 'darwin'

    @staticmethod
    def is_windows():
        return sys.platform.startswith('win')

    def get_library_folder(self, default_mac):
        """
        Return top level library folder. Will be use for setting paths
        """
        if self.is_windows_deployment():
            return os.path.dirname(sys.executable)
        if self.is_mac_deployment():
            return os.path.dirname(default_mac)
        if self.is_linux_deployment():
            return os.path.dirname(sys.executable)

    def setup_tk_tcl_environ(self, root_folder):
        """
        Given a root folder to look in, find the required configuration files for TCL/TK and set the proper
        environmental variables so everything works fine in the distribution package.

        :param root_folder: the top folder from which to start looking for the required configuration files
        """
        tk_folder = self._find_file('tk.tcl', root_folder)
        if tk_folder:
            os.environ['TK_LIBRARY'] = tk_folder

        tcl_folder = self._find_file('init.tcl', root_folder)
        if tcl_folder:
            os.environ['TCL_LIBRARY'] = tcl_folder

    def _find_file(self, target_file, root_folder):
        """
        Search for a file in a folder directory. Return the folder in which the file can be found.

        :param target_file: the name of the file that is searched
        :param root_folder: the top lever folder from which to start searching in all it's subdirectories
        :returns: the name of the folder in which the file can be found
        """
        for root, _, files in os.walk(root_folder):
            for file_n in files:
                if file_n == target_file:
                    return root

    def setup_python_path(self, *paths):
        """
        Set PYTHONPATH
        :param paths: list of absolute folder paths to join.
        """
        os.environ['PYTHONPATH'] = os.pathsep.join(paths)

    def append_to_path(self, *paths):
        """
        Set PATH
        :param paths: list of absolute folder paths to join and add BEFORE the current PATH
        """
        paths = list(paths)
        paths.append(os.environ.get('PATH', ''))
        os.environ['PATH'] = os.pathsep.join(paths)
