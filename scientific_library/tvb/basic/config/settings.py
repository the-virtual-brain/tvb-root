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
TVB-Simulator-Library global configurations are defined here.

Also the generic TVB-Configuration gets set from this point 
(dependent on the previously user-specified profile).

"""

import os
from tvb.basic.config.utils import ClassProperty, EnhancedDictionary
from tvb.basic.profile import TvbProfile


class LibraryProfile():
    """
    Profile needed at the level of TVB Simulator Library.
    It needs to respect a minimal pattern, common to all TVB available profiles.
    """
    
    ## Number used for estimation of TVB used storage space
    MAGIC_NUMBER = 9
    
    ## Maximum number of vertices accepted on a Surface object.
    ## Used for validation purposes.
    MAX_SURFACE_VERTICES_NUMBER = 300000
    
    ## Default value of folder where to store logging files.
    TVB_STORAGE = os.path.expanduser(os.path.join("~", "TVB" + os.sep))
    
    ## Temporary add-on used in DataTypes_Framework.
    ## When DataType_Framework classes will get moved, this should be removed.
    WEB_VISUALIZERS_URL_PREFIX = ""
    
    ## Way of functioning traits is different when using storage or not.
    ## Storage set to false is valid when using TVB_Simulator_Library stand-alone.
    TRAITS_CONFIGURATION = EnhancedDictionary()
    TRAITS_CONFIGURATION.interface_method_name = 'interface'
    TRAITS_CONFIGURATION.use_storage = False
    
    ## Name of file where logging configuration is stored.
    LOGGER_CONFIG_FILE_NAME = "library_logger.conf"
    
    @ClassProperty
    @staticmethod
    def TVB_TEMP_FOLDER():
        """ 
        Represents a temporary folder, where to store things for a while.
        Content of this folder can be deleted at any time.
        """
        tmp_path = os.path.join(LibraryProfile.TVB_STORAGE, "TEMP")
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        return tmp_path
    
    
    @ClassProperty
    @staticmethod
    def TVB_LOG_FOLDER():
        """ Return a folder, where all log files are to be stored. """
        tmp_path = os.path.join(LibraryProfile.TVB_STORAGE, "logs")
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        return tmp_path
    
   
    @classmethod
    def initialize_profile(cls):
        """No initialization needed for this particular profile. But useful in general"""
        pass
    

    
###
###  Dependent of the selected profile and framework classes being present or not, load the correct configuration.
###    
    
if TvbProfile.is_library_mode():
    ## TVB-Simulator-Library is used stand-alone.
    TVBSettings = LibraryProfile
    
else:
    ## Initialization based on profile is further done in Framework.
    from tvb.config.settings import FrameworkSettings
    TVBSettings = FrameworkSettings
        
TVBSettings.initialize_profile()


