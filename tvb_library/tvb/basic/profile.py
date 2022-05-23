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
TVB Profile Manager (top level in TVB profile & settings).

This class is responsible for referring towards application settings,
based on current running environment (e.g. dev vs deployment), or developer profile choice (e.g. web vs console).

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>

"""

from tvb.basic.config.environment import Environment
from tvb.basic.config.profile_settings import BaseSettingsProfile


class TvbProfile(object):
    """
    ENUM-like class with current TVB profile and accepted values.
    """

    LIBRARY_PROFILE = "LIBRARY_PROFILE"
    COMMAND_PROFILE = "COMMAND_PROFILE"
    WEB_PROFILE = "WEB_PROFILE"
    MATLAB_PROFILE = "MATLAB_PROFILE"

    TEST_LIBRARY_PROFILE = "TEST_LIBRARY_PROFILE"
    TEST_POSTGRES_PROFILE = "TEST_POSTGRES_PROFILE"
    TEST_SQLITE_PROFILE = "TEST_SQLITE_PROFILE"

    ALL = [LIBRARY_PROFILE, COMMAND_PROFILE, WEB_PROFILE, MATLAB_PROFILE,
           TEST_POSTGRES_PROFILE, TEST_SQLITE_PROFILE, TEST_LIBRARY_PROFILE]

    REGISTERED_PROFILES = {}

    CURRENT_PROFILE_NAME = None

    current = BaseSettingsProfile()
    env = Environment()

    @classmethod
    def set_profile(cls, selected_profile, in_operation=False, run_init=True):
        """
        Sets TVB profile and do related initializations.
        """
        if selected_profile is not None:
            cls._load_framework_profiles(selected_profile)
            cls._build_profile_class(selected_profile, in_operation, run_init)

    @classmethod
    def _build_profile_class(cls, selected_profile, in_operation=False, run_init=True):
        """
        :param selected_profile: Profile name to be loaded.
        """

        if selected_profile in cls.REGISTERED_PROFILES:
            current_class = cls.REGISTERED_PROFILES[selected_profile]

            cls.current = current_class()
            cls.CURRENT_PROFILE_NAME = selected_profile

            if in_operation:
                # set flags IN_OPERATION,  before initialize** calls, to avoid LoggingBuilder being created there
                cls.current.prepare_for_operation_mode()

            if cls.env.is_distribution():
                # initialize deployment first, because in case of a contributor setup this tried to reload
                # and initialize_profile loads already too many tvb modules,
                # making the reload difficult and prone to more failures
                cls.current.initialize_for_deployment()
            if run_init:
                cls.current.initialize_profile()

        else:
            msg = "Invalid profile name %r, expected one of %r"
            msg %= (selected_profile, cls.ALL)
            raise Exception(msg)

    @classmethod
    def _load_framework_profiles(cls, new_profile):

        from tvb.basic.config.profile_settings import LibrarySettingsProfile, TestLibraryProfile, MATLABLibraryProfile
        cls.REGISTERED_PROFILES[TvbProfile.LIBRARY_PROFILE] = LibrarySettingsProfile
        cls.REGISTERED_PROFILES[TvbProfile.TEST_LIBRARY_PROFILE] = TestLibraryProfile
        cls.REGISTERED_PROFILES[TvbProfile.MATLAB_PROFILE] = MATLABLibraryProfile

        if not cls.is_library_mode(new_profile):
            try:
                from tvb.config.profile_settings import CommandSettingsProfile, WebSettingsProfile
                from tvb.config.profile_settings import TestPostgresProfile, TestSQLiteProfile

                cls.REGISTERED_PROFILES[TvbProfile.COMMAND_PROFILE] = CommandSettingsProfile
                cls.REGISTERED_PROFILES[TvbProfile.WEB_PROFILE] = WebSettingsProfile
                cls.REGISTERED_PROFILES[TvbProfile.TEST_POSTGRES_PROFILE] = TestPostgresProfile
                cls.REGISTERED_PROFILES[TvbProfile.TEST_SQLITE_PROFILE] = TestSQLiteProfile

            except ImportError:
                pass

    @staticmethod
    def is_library_mode(new_profile=None):

        lib_profiles = [TvbProfile.LIBRARY_PROFILE, TvbProfile.TEST_LIBRARY_PROFILE]
        result = (new_profile in lib_profiles
                  or (new_profile is None and TvbProfile.CURRENT_PROFILE_NAME in lib_profiles)
                  or not TvbProfile.env.is_framework_present())

        # Make sure default settings are not failing because we are not finding some modules
        if (new_profile is None and TvbProfile.CURRENT_PROFILE_NAME is None and
                not TvbProfile.env.is_framework_present()):
            TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

        return result

    @staticmethod
    def is_first_run():

        return TvbProfile.current.manager.is_first_run()
