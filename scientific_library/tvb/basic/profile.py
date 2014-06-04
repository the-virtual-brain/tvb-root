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
ENUM used for choosing current TVB Profile.

Contains functionality which allows a user to set a certain profile for TVB.
"""
import os
import sys


class LibraryModulesFinder():
    """
    In case users run TVB in 'library' profile access should be restricted
    to some parts of tvb.
    """
    restricted_modules = ['tvb.interfaces',
                          'tvb.datatype_removers',
                          'tvb.core',
                          'tvb.config',
                          'tvb.adapters']
    
    
    def find_module(self, fullname, path=None):
        if fullname in self.restricted_modules:
            return self
        
        
    def load_module(self, module_name):
        info_message = str("You are trying to import the module `%s` in library mode."
                           "The library profile is a lightweight version of TVB and you "
                           "only have access to the simulator, analyzers and datatypes packages."
                           "If you want to use the entire TVB Framework start it either in command "
                           "or web interface profile." % module_name)
        raise ImportError(info_message)
    

class TvbProfile():
    """
    ENUM-like class with current TVB profile values.
    """

    SUBPARAM_PROFILE = "-profile"

    # Existing profiles:
    LIBRARY_PROFILE = "LIBRARY_PROFILE"
    DEVELOPMENT_PROFILE = "DEVELOPMENT_PROFILE"
    TEST_POSTGRES_PROFILE = "TEST_POSTGRES_PROFILE"
    TEST_SQLITE_PROFILE = "TEST_SQLITE_PROFILE"
    DEPLOYMENT_PROFILE = "DEPLOYMENT_PROFILE"
    CONSOLE_PROFILE = "CONSOLE_PROFILE"

    # Used for setting the current TVB profile
    CURRENT_SELECTED_PROFILE = None


    @staticmethod
    def get_profile(script_argv):
        """
        Returns the user given profile or None if the user didn't specify a profile.

        :param script_argv: represents a list of string arguments.
                If the script_argv contains the string '-profile',
                than TVB profile will be set to the next element.

        E.g.: if script_argv=['$param1', ..., '-profile', 'TEST_SQLITE_PROFILE', ...] 
               than TVB profile will be set to 'TEST_SQLITE_PROFILE'
        """
        if TvbProfile.SUBPARAM_PROFILE in script_argv:
            index = script_argv.index(TvbProfile.SUBPARAM_PROFILE)

            if len(script_argv) > index + 1:
                return script_argv[index + 1]

        return None


    @staticmethod
    def set_profile(script_argv, remove_from_args=False, try_reload=True):
        """
        Sets TVB profile from script_argv.

        :param script_argv: represents a list of string arguments.
                      If the script_argv contains the string '-profile' 
                      than the TVB profile will be set to the next element.
        :param remove_from_args: when True, script_argv will get stripped of profile strings.

        :param try_reload: When set to true, try to reload all tvb.* modules
                        This is needed when setting a profile different that default requires previously loaded tvb
                        modules to get different (e.g. from deployment to developer we have a different
                        tvb.interface path, already loaded as the starting point is tvb.interfaces.run)
        
        E.g.: if script_argv = ['$param1', ..., '-profile', 'TEST_SQLITE_PROFILE', ...] 
              than the  profile will be set to 'TEST_SQLITE_PROFILE'
        """
        selected_profile = TvbProfile.get_profile(script_argv)
        
        if try_reload:
            # To make sure in case of contributor setup the external TVB is the one
            # we get, we need to reload all tvb related modules, since any call done
            # python -m will always consider the current folder as the first to search in.
            sys.path = os.environ.get("PYTHONPATH", "").split(os.pathsep) + sys.path
            for key in sys.modules.keys():
                if key.startswith("tvb") and sys.modules[key]:
                    reload(sys.modules[key])

        if selected_profile is not None:
            TvbProfile.CURRENT_SELECTED_PROFILE = selected_profile

            if remove_from_args:
                script_argv.remove(selected_profile)
                script_argv.remove(TvbProfile.SUBPARAM_PROFILE)
                
            if selected_profile == TvbProfile.LIBRARY_PROFILE:
                sys.meta_path.append(LibraryModulesFinder())
        

    @staticmethod
    def is_library_mode():
        """
        Fall-back to LibraryProfile either if this was the profile passed as argument or if TVB Framework is not found.
        
        :return: True when currently selected profile is LibraryProfile, 
                 or when the framework classes are not present, and we should enforce the library profile.
        """
        framework_present = True
        try:
            from tvb.config.settings import FrameworkSettings
        except ImportError:
            framework_present = False

        return TvbProfile.CURRENT_SELECTED_PROFILE == TvbProfile.LIBRARY_PROFILE or not framework_present     
            
            
            