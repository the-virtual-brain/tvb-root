# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
Environment related checks are to be done here.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import os
import sys


class Environment():


    def is_framework_present(self):
        """
        :return: True when framework classes are present and can be imported.
        """
        framework_present = True
        try:
            from tvb.config.settings import DevelopmentProfile
        except ImportError:
            framework_present = False

        return framework_present


    @staticmethod
    def is_development():
        """
        Return True when TVB is used with Python installed natively.
        """
        try:
            import tvb_bin
            bin_folder = os.path.dirname(os.path.abspath(tvb_bin.__file__))
            tvb_version_file = os.path.join(bin_folder, "tvb.version")
            if os.path.exists(tvb_version_file):
                return False
            return True
        except ImportError:
            return True


    def is_linux_deployment(self):
        """
        Return True if current run is not development and is running on Linux.
        """
        return self.is_linux() and not self.is_development()


    def is_mac_deployment(self):
        """
        Return True if current run is not development and is running on Mac OS X
        """
        return self.is_mac() and not self.is_development()


    def is_windows_deployment(self):
        """
        Return True if current run is not development and is running on Windows.
        """
        return self.is_windows() and not self.is_development()


    def is_linux(self):
        return not self.is_windows() and not self.is_mac()


    @staticmethod
    def is_mac():
        return sys.platform == 'darwin'


    @staticmethod
    def is_windows():
        return sys.platform.startswith('win')
