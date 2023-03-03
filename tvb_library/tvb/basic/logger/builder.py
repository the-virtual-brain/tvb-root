# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
Singleton logging builder.


.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""
import importlib
import os
import weakref
import logging
import logging.config
from tvb.basic.profile import TvbProfile
from tvb.basic.config.profile_settings import MATLABLibraryProfile
from logging import StreamHandler


class LoggerBuilder(object):
    """
    Class taking care of uniform Python logger initialization. 
    It uses the Python native logging package. 
    It's purpose is just to offer a common mechanism for initializing all modules in a package.
    """

    def __init__(self, config_root):
        """
        Prepare Python logger based on a configuration file.
        :param: config_root - current package to configure logger for it.

        """
        if not isinstance(TvbProfile.current, MATLABLibraryProfile):
            config_file_name = TvbProfile.current.LOGGER_CONFIG_FILE_NAME
            package = importlib.import_module(config_root)
            package_path = package.__path__[0]
            # Specify logging configuration file for current package.
            logging.config.fileConfig(os.path.join(package_path, config_file_name), disable_existing_loggers=False)
        else:
            logging.basicConfig(level=logging.DEBUG)
        self._loggers = weakref.WeakValueDictionary()

    def build_logger(self, parent_module):
        """
        Build a logger instance and return it
        """
        self._loggers[parent_module] = logger = logging.getLogger(parent_module)
        return logger

    def set_loggers_level(self, level):
        for logger in self._loggers.values():
            logger.setLevel(level)
            for handler in logger.handlers:
                if isinstance(handler, StreamHandler) and handler.stream.name == 'stdout':
                    handler.setLevel(min(level, handler.level))


# We make sure a single instance of logger-builder is created.
if "GLOBAL_LOGGER_BUILDER" not in globals():

    if TvbProfile.is_library_mode():
        GLOBAL_LOGGER_BUILDER = LoggerBuilder('tvb.basic.logger')
    else:
        GLOBAL_LOGGER_BUILDER = LoggerBuilder('tvb.config.logger')


def get_logger(parent_module=''):
    """
    Function to retrieve a new Python logger instance for current module.
    
    :param parent_module: module name for which to create logger.
    """
    return GLOBAL_LOGGER_BUILDER.build_logger(parent_module)


def set_loggers_level(level):
    """
    Function to set the logging level for the loggers and their console handlers

    :param level: the level to be set
    """
    GLOBAL_LOGGER_BUILDER.set_loggers_level(level)
