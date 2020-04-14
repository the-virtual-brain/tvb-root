# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
import shutil
import sys
from functools import wraps
from types import FunctionType
from tvb.config.init.model_manager import reset_database
from tvb.config.init.initializer import initialize
from tvb.core.neocom.h5 import REGISTRY
from tvb.tests.framework.datatypes.dummy_datatype import DummyDataType
from tvb.tests.framework.datatypes.dummy_datatype2_index import DummyDataType2Index
from tvb.tests.framework.datatypes.dummy_datatype_h5 import DummyDataTypeH5
from tvb.tests.framework.datatypes.dummy_datatype_index import DummyDataTypeIndex


def init_test_env():
    """
    This method prepares all necessary data for tests execution
    """
    # Set a default test profile, for when running tests from dev-env.
    from tvb.basic.profile import TvbProfile
    if TvbProfile.CURRENT_PROFILE_NAME is None:
        profile = TvbProfile.TEST_SQLITE_PROFILE
        if len(sys.argv) > 1:
            for i in range(1, len(sys.argv) - 1):
                if "--profile=" in sys.argv[i]:
                    profile = sys.argv[i].split("=")[1]
        TvbProfile.set_profile(profile)
        print("Not expected to happen except from PyCharm: setting profile", TvbProfile.CURRENT_PROFILE_NAME)
        db_file = TvbProfile.current.db.DB_URL.replace('sqlite:///', '')
        if os.path.exists(db_file):
            os.remove(db_file)

    reset_database()
    initialize(skip_import=True)

    # Add Dummy DataType
    REGISTRY.register_datatype(DummyDataType, DummyDataTypeH5, DummyDataTypeIndex)
    REGISTRY.register_datatype(None, None, DummyDataType2Index)


# Following code is executed once / tests execution to reduce time spent in tests.
if "TEST_INITIALIZATION_DONE" not in globals():
    init_test_env()
    TEST_INITIALIZATION_DONE = True

from tvb.adapters.exporters.export_manager import ExportManager
from tvb.core.services.operation_service import OperationService
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.storage import dao
from tvb.core.entities.storage.session_maker import SessionMaker
from tvb.core.entities.model.model_project import *
from tvb.core.entities.model.model_datatype import *
import decorator
from tvb.core.adapters.abcdisplayer import ABCDisplayer

LOGGER = get_logger(__name__)


class BaseTestCase(object):
    """
    This class should implement basic functionality which is common to all TVB tests.
    """
    EXCLUDE_TABLES = ["ALGORITHMS", "ALGORITHM_CATEGORIES", "PORTLETS",
                      "MAPPED_INTERNAL__CLASS", "MAPPED_MAPPED_TEST_CLASS"]

    def assertEqual(self, expected, actual, message=""):
        assert expected == actual, message + " Expected %s but got %s." % (expected, actual)

    def clean_database(self, delete_folders=True):
        """
        Deletes data from all tables
        """
        self.cancel_all_operations()
        LOGGER.warning("Your Database content will be deleted.")
        try:
            session = SessionMaker()
            for table in reversed(Base.metadata.sorted_tables):
                # We don't delete data from some tables, because those are 
                # imported only during introspection which is done one time
                if table.name not in self.EXCLUDE_TABLES:
                    try:
                        session.open_session()
                        con = session.connection()
                        LOGGER.debug("Executing Delete From Table " + table.name)
                        con.execute(table.delete())
                        session.commit()
                    except Exception as e:
                        # We cache exception here, in case some table does not exists and
                        # to allow the others to be deleted
                        LOGGER.warning(e)
                        session.rollback()
                    finally:
                        session.close_session()
            LOGGER.info("Database was cleanup!")
        except Exception as excep:
            LOGGER.warning(excep)
            raise

        # Now if the database is clean we can delete also project folders on disk
        if delete_folders:
            self.delete_project_folders()
        dao.store_entity(
            User(TvbProfile.current.web.admin.SYSTEM_USER_NAME, TvbProfile.current.web.admin.SYSTEM_USER_NAME, None,
                 None, True, None))

    def cancel_all_operations(self):
        """
        To make sure that no running operations are left which could make some other
        test started afterwards to fail, cancel all operations after each test.
        """
        LOGGER.info("Stopping all operations.")
        op_service = OperationService()
        operations = self.get_all_entities(Operation)
        for operation in operations:
            op_service.stop_operation(operation.id)

    def delete_project_folders(self):
        """
        This method deletes folders for all projects from TVB folder.
        This is done without any check on database. You might get projects in DB but no folder for them on disk.
        """
        projects_folder = os.path.join(TvbProfile.current.TVB_STORAGE, FilesHelper.PROJECTS_FOLDER)
        if os.path.exists(projects_folder):
            for current_file in os.listdir(projects_folder):
                full_path = os.path.join(TvbProfile.current.TVB_STORAGE, FilesHelper.PROJECTS_FOLDER, current_file)
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path, ignore_errors=True)

        for folder in [os.path.join(TvbProfile.current.TVB_STORAGE, ExportManager.EXPORT_FOLDER_NAME),
                       os.path.join(TvbProfile.current.TVB_STORAGE, FilesHelper.TEMP_FOLDER)]:
            if os.path.exists(folder):
                shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder)

    @staticmethod
    def compute_recursive_h5_disk_usage(start_path='.'):
        """
        Computes the disk usage of all h5 files under the given directory.
        :param start_path:
        :return: A tuple of size in kiB and number of files
        """
        total_size = 0
        n_files = 0
        for dir_path, _, file_names in os.walk(start_path):
            for f in file_names:
                if f.endswith('.h5'):
                    fp = os.path.join(dir_path, f)
                    total_size += os.path.getsize(fp)
                    n_files += 1
        return int(round(total_size / 1024.)), n_files

    def count_all_entities(self, entity_type):
        """
        Count all entities of a given type currently stored in DB.
        """
        result = 0
        session = None
        try:
            session = SessionMaker()
            session.open_session()
            result = session.query(entity_type).count()
        except Exception as excep:
            LOGGER.warning(excep)
        finally:
            if session:
                session.close_session()
        return result

    def get_all_entities(self, entity_type):
        """
        Retrieve all entities of a given type.
        """
        result = []
        session = None
        try:
            session = SessionMaker()
            session.open_session()
            result = session.query(entity_type).all()
        except Exception as excep:
            LOGGER.warning(excep)
        finally:
            if session:
                session.close_session()
        return result

    def get_all_datatypes(self):
        """
        Return all DataType entities in DB or [].
        """
        return self.get_all_entities(DataType)

    def assert_compliant_dictionary(self, expected, found_dict):
        """
        Compare two dictionaries, especially as keys.
        When in expected_dictionary the value is not None, validate also to be found in found_dict.
        """
        for key, value in expected.items():
            assert key in found_dict, "%s not found in result" % key
            if value is not None:
                assert value == found_dict[key]


def transactional_test(func, callback=None):
    """
    A decorator to be used in tests which makes sure all database changes are reverted at the end of the test.
    """
    if func.__name__.startswith('test_'):

        @wraps(func)
        def dec(func, *args, **kwargs):
            session_maker = SessionMaker()
            TvbProfile.current.db.ALLOW_NESTED_TRANSACTIONS = True
            session_maker.start_transaction()
            ABCDisplayer.VISUALIZERS_ROOT = TvbProfile.current.web.VISUALIZERS_ROOT
            try:
                try:
                    if hasattr(args[0], 'transactional_setup_method_TVB'):
                        LOGGER.debug(args[0].__class__.__name__ + "->" + func.__name__
                                     + "- Transactional SETUP starting...")
                        args[0].transactional_setup_method_TVB()
                    result = func(*args, **kwargs)
                finally:
                    if hasattr(args[0], 'transactional_teardown_method_TVB'):
                        LOGGER.debug(args[0].__class__.__name__ + "->" + func.__name__
                                     + "- Transactional TEARDOWN starting...")
                        args[0].transactional_teardown_method_TVB()
                        args[0].delete_project_folders()
            finally:
                session_maker.rollback_transaction()
                session_maker.close_transaction()
                TvbProfile.current.db.ALLOW_NESTED_TRANSACTIONS = False

            if callback is not None:
                callback(*args, **kwargs)
            return result

        return decorator.decorator(dec, func)
    else:
        return func


class TransactionalTestMeta(type):
    """
    New MetaClass.
    """

    def __new__(mcs, classname, bases, class_dict):
        """
        Called when a new class gets instantiated.
        """
        new_class_dict = {}
        for attr_name, attribute in class_dict.items():
            if (type(attribute) == FunctionType and not (attribute.__name__.startswith('__')
                                                         and attribute.__name__.endswith('__'))):
                if attr_name.startswith('test_'):
                    attribute = transactional_test(attribute)
                if attr_name in ('transactional_setup_method', 'transactional_teardown_method'):
                    new_class_dict[attr_name + '_TVB'] = attribute
                else:
                    new_class_dict[attr_name] = attribute
            else:
                new_class_dict[attr_name] = attribute
        return type.__new__(mcs, classname, bases, new_class_dict)


class TransactionalTestCase(BaseTestCase, metaclass=TransactionalTestMeta):
    """
    This class makes sure that any test case it contains is ran in a transactional
    environment and a rollback is issued at the end of that transaction. This should
    improve performance for most cases.

    IMPORTANT: in case you want the unit-test method setup to be done in the same transaction
    as the unit-test (recommended situation), then define methods:
    - 'transactional_setup_method'
    - 'transactional_teardown_method'
    in your subclass of TransactionalTestCase!
    
    WARNING! Do not use this is any test class that has uses multiple threads to do
    dao related operations since that might cause errors/leave some dangling sessions.
    """
