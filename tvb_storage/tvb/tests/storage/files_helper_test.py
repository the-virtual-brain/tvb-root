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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

import os
import pytest

from tvb.basic.profile import TvbProfile
from tvb.storage.h5.file.exceptions import FileStructureException
from tvb.storage.h5.file.files_helper import FilesHelper
from tvb.storage.h5.file.xml_metadata_handlers import XMLReader
from tvb.storage.storage_interface import StorageInterface
from tvb.tests.storage.dummy.dummy_project import DummyProject
from tvb.tests.storage.dummy.dummy_storage_data_h5 import DummyStorageDataH5
from tvb.tests.storage.storage_test import StorageTestCase

root_storage = TvbProfile.current.TVB_STORAGE


class TestFilesHelper(StorageTestCase):
    """
    This class contains tests for the tvb.storage.h5.file.files_helper module.
    """

    def storage_setup_method(self):
        self.files_helper = FilesHelper()
        self.project_name = "test_proj"

    def storage_teardown_method(self):
        self.delete_projects_folders()

    def test_check_created(self):
        """ Test standard flows for check created. """
        self.files_helper.check_created(TvbProfile.current.TVB_STORAGE)
        assert os.path.exists(root_storage), "Storage not created!"

        self.files_helper.check_created(os.path.join(root_storage, "test"))
        assert os.path.exists(root_storage), "Storage not created!"
        assert os.path.exists(os.path.join(root_storage, "test")), "Test directory not created!"

    def test_get_project_folder(self):
        """
        Test the get_project_folder method which should create a folder in case
        it doesn't already exist.
        """
        project_path = self.files_helper.get_project_folder(self.project_name)
        assert os.path.exists(project_path), "Folder doesn't exist"

        folder_path = self.files_helper.get_project_folder(self.project_name, "43")
        assert os.path.exists(project_path), "Folder doesn't exist"
        assert os.path.exists(folder_path), "Folder doesn't exist"

    def test_rename_project_structure(self):
        """ Try to rename the folder structure of a project. Standard flow. """
        self.files_helper.get_project_folder(self.project_name)
        path, name = self.files_helper.rename_project_structure(self.project_name, "new_name")
        assert path != name, "Rename didn't take effect."

    def test_rename_structure_same_name(self):
        """ Try to rename the folder structure of a project. Same name. """
        self.files_helper.get_project_folder(self.project_name)

        with pytest.raises(FileStructureException):
            self.files_helper.rename_project_structure(self.project_name, self.project_name)

    def test_write_project_metadata(self):
        """  Write XML for test-project. """
        user_id = 1
        dummy_project = DummyProject(self.project_name, "description", 3, user_id)
        self.files_helper.write_project_metadata(dummy_project.to_dict(), StorageInterface.TVB_PROJECT_FILE)
        expected_file = self.files_helper.get_project_meta_file_path(self.project_name,
                                                                     StorageInterface.TVB_PROJECT_FILE)
        assert os.path.exists(expected_file)
        project_meta = XMLReader(expected_file).read_metadata_from_xml()
        loaded_project = DummyProject(None, None, None, None)
        loaded_project.from_dict(project_meta, user_id)
        assert dummy_project.name == loaded_project.name
        assert dummy_project.description == loaded_project.description
        assert dummy_project.gid == loaded_project.gid
        expected_dict = dummy_project.to_dict()
        del expected_dict['last_updated']
        found_dict = loaded_project.to_dict()
        del found_dict['last_updated']
        self._dictContainsSubset(expected_dict, found_dict)
        self._dictContainsSubset(found_dict, expected_dict)

    def test_move_datatype(self):
        """
        Make sure associated H5 file is moved to a correct new location.
        """
        path = self.files_helper.get_project_folder(self.project_name)
        old_h5_path = os.path.join(path, "dummy_datatype.h5")
        DummyStorageDataH5(old_h5_path)
        assert os.path.exists(old_h5_path), "Test file was not created!"
        self.files_helper.move_datatype(self.project_name + '2', "1", old_h5_path)

        assert not os.path.exists(old_h5_path), "Test file was not moved!"
        new_file_path = os.path.join(self.files_helper.get_project_folder(self.project_name + '2', "1"),
                                     os.path.basename(old_h5_path))
        assert os.path.exists(new_file_path), "Test file was not created!"

    def test_remove_files_valid(self):
        """
        Pass a valid list of files and check they are all removed.
        """
        file_list = ["test1", "test2", "test3"]
        for file_n in file_list:
            fp = open(file_n, 'w')
            fp.write('test')
            fp.close()
        for file_n in file_list:
            assert os.path.isfile(file_n)
        self.files_helper.remove_files(file_list, False)
        for file_n in file_list:
            assert not os.path.isfile(file_n)

    def test_remove_folder(self):
        """
        Pass an open file pointer, but ignore exceptions.
        """
        folder_name = "test_folder"
        os.mkdir(folder_name)
        assert os.path.isdir(folder_name), "Folder should be created."
        self.files_helper.remove_folder(folder_name, False)
        assert not os.path.isdir(folder_name), "Folder should be deleted."

    def test_remove_folder_non_existing_ignore_exc(self):
        """
        Pass an open file pointer, but ignore exceptions.
        """
        folder_name = "test_folder"
        assert not os.path.isdir(folder_name), "Folder should not exist before call."
        self.files_helper.remove_folder(folder_name, ignore_errors=True)

    def test_remove_folder_non_existing(self):
        """
        Pass an open file pointer, but ignore exceptions.
        """
        folder_name = "test_folder"
        assert not os.path.isdir(folder_name), "Folder should not exist before call."
        with pytest.raises(FileStructureException):
            self.files_helper.remove_folder(folder_name, ignore_errors=False)

    def _dictContainsSubset(self, expected, actual, msg=None):
        """Checks whether actual is a superset of expected."""
        missing = []
        mismatched = []
        for key, value in expected.items():
            if key not in actual:
                return False
            elif value != actual[key]:
                return False
        return True
