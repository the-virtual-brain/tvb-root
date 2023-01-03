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
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

import os
import shutil
from functools import wraps
from types import FunctionType
import decorator

from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.storage.storage_interface import StorageInterface

LOGGER = get_logger(__name__)


class BaseStorageTestCase(object):

    @staticmethod
    def delete_projects_folders():
        projects_folder = os.path.join(TvbProfile.current.TVB_STORAGE, StorageInterface.PROJECTS_FOLDER)
        if os.path.exists(projects_folder):
            for current_file in os.listdir(projects_folder):
                full_path = os.path.join(TvbProfile.current.TVB_STORAGE, StorageInterface.PROJECTS_FOLDER, current_file)
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path, ignore_errors=True)


def storage_test(func, callback=None):
    """
    A decorator to be used in tests which makes sure all stored files and folders are deleted at the end of the test.
    """
    if func.__name__.startswith('test_'):

        @wraps(func)
        def dec(func, *args, **kwargs):
            try:
                if hasattr(args[0], 'storage_setup_method_TVB'):
                    LOGGER.debug(args[0].__class__.__name__ + "->" + func.__name__
                                 + "- Storage SETUP starting...")
                    args[0].storage_setup_method_TVB()
                result = func(*args, **kwargs)
            finally:
                if hasattr(args[0], 'storage_teardown_method_TVB'):
                    LOGGER.debug(args[0].__class__.__name__ + "->" + func.__name__
                                 + "- Storage TEARDOWN starting...")
                    args[0].storage_teardown_method_TVB()

            if callback is not None:
                callback(*args, **kwargs)
            return result

        return decorator.decorator(dec, func)
    else:
        return func


class StorageTestMeta(type):

    def __new__(mcs, classname, bases, class_dict):
        """
        Called when a new class gets instantiated.
        """
        new_class_dict = {}
        for attr_name, attribute in class_dict.items():
            if (type(attribute) == FunctionType and not (attribute.__name__.startswith('__')
                                                         and attribute.__name__.endswith('__'))):
                if attr_name.startswith('test_'):
                    attribute = storage_test(attribute)
                if attr_name in ('storage_setup_method', 'storage_teardown_method'):
                    new_class_dict[attr_name + '_TVB'] = attribute
                else:
                    new_class_dict[attr_name] = attribute
            else:
                new_class_dict[attr_name] = attribute
        return type.__new__(mcs, classname, bases, new_class_dict)


class StorageTestCase(BaseStorageTestCase, metaclass=StorageTestMeta):
    """
    This class makes sure that any test case that has 'storage_setup_method' and
    'storage_teardown_method' will have those methods called before each, and
    after each method respectively.
    """