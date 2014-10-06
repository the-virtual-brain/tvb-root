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
TVB Profile.

This class is responsible for referring towards application settings, based on running environment or developer choice.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>

"""

import os
import sys
import copy
from tvb.basic.config.utils import LibraryImportError
from tvb.basic.config.environment import Environment
from tvb.basic.config.settings import BaseSettingsProfile



class TvbProfile():
    """
    ENUM-like class with current TVB profile and accepted values.
    """

    DEVELOPMENT_PROFILE = "DEVELOPMENT_PROFILE"
    DEPLOYMENT_PROFILE = "DEPLOYMENT_PROFILE"
    LIBRARY_PROFILE = "LIBRARY_PROFILE"
    COMMAND_PROFILE = "COMMAND_PROFILE"
    TEST_LIBRARY_PROFILE = "TEST_LIBRARY_PROFILE"
    TEST_POSTGRES_PROFILE = "TEST_POSTGRES_PROFILE"
    TEST_SQLITE_PROFILE = "TEST_SQLITE_PROFILE"
    DESKTOP_PROFILE = "DESKTOP_PROFILE"

    ALL = [DEVELOPMENT_PROFILE, DEPLOYMENT_PROFILE, LIBRARY_PROFILE, COMMAND_PROFILE,
           TEST_POSTGRES_PROFILE, TEST_SQLITE_PROFILE, TEST_LIBRARY_PROFILE, DESKTOP_PROFILE]

    REGISTERED_PROFILES = {}

    CURRENT_PROFILE_NAME = None

    current = BaseSettingsProfile(False)
    env = Environment()

    _old_meta_path = copy.deepcopy(sys.meta_path)


    @classmethod
    def set_profile(cls, selected_profile, try_reload=False):
        """
        Sets TVB profile from script_argv and specify UTF-8 and encoding.

        :param selected_profile: String representing profile to be set.
        :param try_reload: When set to true, try to reload all tvb.* modules
                        This is needed when setting a profile different that default requires previously loaded tvb
                        modules to get different (e.g. from deployment to contributor we have a different
                        tvb.interface path, already loaded as the starting point is tvb.interfaces.run)
        """

        ### Ensure Python is using UTF-8 encoding (otherwise default encoding is ASCII)
        ### We should make sure UTF-8 gets set before reading from any TVB files
        ### e.g. TVB_STORAGE will differ if the .tvb.configuration file contains non-ascii bytes
        ### most of the comments in the simulator are having pieces outside of ascii coverage
        if cls.env.is_development() and sys.getdefaultencoding().lower() != 'utf-8':
            reload(sys)
            sys.setdefaultencoding('utf-8')

        if selected_profile is not None:
            ## Restore sys.meta_path, as some profiles (Library) are adding something
            sys.meta_path = copy.deepcopy(cls._old_meta_path)
            cls._load_framework_profiles(selected_profile)
            if selected_profile in cls.REGISTERED_PROFILES:
                current_class = cls.REGISTERED_PROFILES[selected_profile]
                cls.current = current_class()
                cls.current.initialize_profile()
            else:
                raise Exception("Invalid profile %s" % selected_profile)

            cls.CURRENT_PROFILE_NAME = selected_profile

        if try_reload:
            # To make sure in case of contributor setup the external TVB is the one
            # we get, we need to reload all tvb related modules, since any call done
            # python -m will always consider the current folder as the first to search in.
            sys.path = os.environ.get("PYTHONPATH", "").split(os.pathsep) + sys.path
            for key in sys.modules.keys():
                if key.startswith("tvb") and sys.modules[key] and not key.startswith("tvb.basic.profile"):
                    try:
                        reload(sys.modules[key])
                    except LibraryImportError:
                        pass


    @classmethod
    def _load_framework_profiles(cls, new_profile):

        from tvb.basic.config.settings import LibrarySettingsProfile, TestLibrarySettingsProfile
        cls.REGISTERED_PROFILES[TvbProfile.LIBRARY_PROFILE] = LibrarySettingsProfile
        cls.REGISTERED_PROFILES[TvbProfile.TEST_LIBRARY_PROFILE] = TestLibrarySettingsProfile

        if not cls.is_library_mode(new_profile):
            try:
                from tvb.config.settings import CommandProfile, DeploymentProfile, DevelopmentProfile
                from tvb.config.settings import TestPostgresProfile, TestSQLiteProfile

                cls.REGISTERED_PROFILES[TvbProfile.COMMAND_PROFILE] = CommandProfile
                cls.REGISTERED_PROFILES[TvbProfile.DEPLOYMENT_PROFILE] = DeploymentProfile
                cls.REGISTERED_PROFILES[TvbProfile.DEVELOPMENT_PROFILE] = DevelopmentProfile
                cls.REGISTERED_PROFILES[TvbProfile.TEST_POSTGRES_PROFILE] = TestPostgresProfile
                cls.REGISTERED_PROFILES[TvbProfile.TEST_SQLITE_PROFILE] = TestSQLiteProfile

            except ImportError:
                pass


    @staticmethod
    def is_library_mode(new_profile=None):

        lib_profiles = [TvbProfile.LIBRARY_PROFILE, TvbProfile.TEST_LIBRARY_PROFILE]
        return (new_profile in lib_profiles
                or (new_profile is None and TvbProfile.CURRENT_PROFILE_NAME in lib_profiles)
                or not TvbProfile.env.is_framework_present())


    @staticmethod
    def is_first_run():

        return TvbProfile.current.manager.is_first_run()