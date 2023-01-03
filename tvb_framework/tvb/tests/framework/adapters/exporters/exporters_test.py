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
.. moduleauthor:: calin.pavel <calin.pavel@codemart.ro>
"""
import os.path
import uuid
import zipfile
from contextlib import closing
import pytest

from tvb.adapters.exporters.exceptions import ExportException, InvalidExportDataException
from tvb.adapters.exporters.export_manager import ExportManager
from tvb.basic.profile import TvbProfile
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.services.burst_service import BurstService
from tvb.storage.storage_interface import StorageInterface
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestExporters(TransactionalTestCase):
    """
    Test export functionality.
    """
    TVB_EXPORTER = "TVBExporter"
    TVB_LINKED_EXPORTER = "TVBLinkedExporter"

    def transactional_setup_method(self):
        self.export_manager = ExportManager()
        self.test_user = TestFactory.create_user('Exporter_Tests_User1')
        self.test_project = TestFactory.create_project(self.test_user, 'Exporter_Tests_Project1')

    def transactional_teardown_method(self):
        """
        Clean-up tests data
        """
        # Remove EXPORT folder
        export_folder = os.path.join(TvbProfile.current.TVB_STORAGE, StorageInterface.EXPORT_FOLDER_NAME)
        StorageInterface.remove_folder(export_folder, True)

    def test_get_exporters_for_data(self, dummy_datatype_index_factory):
        """
        Test retrieval of exporters that can be used for a given data.
        """
        datatype = dummy_datatype_index_factory()
        exporters = self.export_manager.get_exporters_for_data(datatype)

        # Only TVB export can export any type of data type
        assert len(exporters) == 2, "Incorrect number of exporters."

    def test_get_exporters_for_data_with_no_data(self):
        """
        Test retrieval of exporters when data == None.
        """
        with pytest.raises(InvalidExportDataException):
            self.export_manager.get_exporters_for_data(None)

    def test_tvb_export_of_simple_datatype(self, dummy_datatype_index_factory):
        """
        Test export of a data type which has no data stored on file system
        """
        datatype = dummy_datatype_index_factory()
        _, file_path, _ = self.export_manager.export_data(datatype, self.TVB_EXPORTER, self.test_project)

        assert file_path is not None, "Export process should return path to export file"
        assert os.path.exists(file_path), "Could not find export file: %s on disk." % file_path

    @staticmethod
    def compare_files(original_path, decrypted_file_path):
        buffer_size = TvbProfile.current.hpc.CRYPT_BUFFER_SIZE
        with open(original_path, 'rb') as f_original:
            with open(decrypted_file_path, 'rb') as f_decrypted:
                while True:
                    original_content_chunk = f_original.read(buffer_size)
                    decrypted_content_chunk = f_decrypted.read(buffer_size)

                    assert original_content_chunk == decrypted_content_chunk, \
                        "Original and Decrypted chunks are not equal, so decryption hasn't been done correctly!"

                    # check if EOF was reached
                    if len(original_content_chunk) < buffer_size:
                        break

    def test_tvb_export_of_simple_datatype_with_encryption(self, dummy_datatype_index_factory):
        """
        Test export of an encrypted data type which has no data stored on file system
        """
        datatype = dummy_datatype_index_factory()
        storage_interface = StorageInterface()
        import_export_encryption_handler = StorageInterface.get_import_export_encryption_handler()
        import_export_encryption_handler.generate_public_private_key_pair(TvbProfile.current.TVB_TEMP_FOLDER)

        _, file_path, _ = self.export_manager.export_data(
            datatype, self.TVB_EXPORTER, self.test_project, os.path.join(
                TvbProfile.current.TVB_TEMP_FOLDER, import_export_encryption_handler.PUBLIC_KEY_NAME))

        assert file_path is not None, "Export process should return path to export file"
        assert os.path.exists(file_path), "Could not find export file: %s on disk." % file_path

        result = storage_interface.unpack_zip(file_path, TvbProfile.current.TVB_TEMP_FOLDER)
        encrypted_password_path = import_export_encryption_handler.extract_encrypted_password_from_list(result)

        decrypted_file_path = import_export_encryption_handler.decrypt_content(
            encrypted_password_path, result, os.path.join(TvbProfile.current.TVB_TEMP_FOLDER,
                                                          import_export_encryption_handler.PRIVATE_KEY_NAME))[0]

        original_path = h5.path_for_stored_index(datatype)
        self.compare_files(original_path, decrypted_file_path)

    def test_tvb_linked_export_of_simple_datatype(self, connectivity_index_factory, surface_index_factory,
                                                  region_mapping_index_factory):
        """
        Test export of a data type and its linked data types which have no data stored on file system
        """

        conn = connectivity_index_factory()
        _, surface = surface_index_factory(cortical=True)
        region_mapping_index = region_mapping_index_factory(conn_gid=conn.gid, surface_gid=surface.gid.hex)

        _, file_path, _ = self.export_manager.export_data(region_mapping_index, self.TVB_LINKED_EXPORTER,
                                                          self.test_project)

        assert file_path is not None, "Export process should return path to export file"
        assert os.path.exists(file_path), "Could not find export file;: %s on disk." % file_path

    def test_tvb_linked_export_of_simple_datatype_with_encryption(self, connectivity_index_factory,
                                                                  surface_index_factory, region_mapping_index_factory):
        """
        Test export of an encrypted data type and its linked data types which have no data stored on file system
        """

        conn = connectivity_index_factory()
        surface_idx, surface = surface_index_factory(cortical=True)
        region_mapping_index = region_mapping_index_factory(conn_gid=conn.gid, surface_gid=surface.gid.hex)

        storage_interface = StorageInterface()
        import_export_encryption_handler = StorageInterface.get_import_export_encryption_handler()
        import_export_encryption_handler.generate_public_private_key_pair(TvbProfile.current.TVB_TEMP_FOLDER)

        _, file_path, _ = self.export_manager.export_data(
            region_mapping_index, self.TVB_LINKED_EXPORTER, self.test_project,
            os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, import_export_encryption_handler.PUBLIC_KEY_NAME))

        assert file_path is not None, "Export process should return path to export file"
        assert os.path.exists(file_path), "Could not find export file;: %s on disk." % file_path

        result = storage_interface.unpack_zip(file_path, TvbProfile.current.TVB_TEMP_FOLDER)
        encrypted_password = import_export_encryption_handler.extract_encrypted_password_from_list(result)

        decrypted_file_paths = import_export_encryption_handler.decrypt_content(
            encrypted_password, result, os.path.join(TvbProfile.current.TVB_TEMP_FOLDER,
                                                     import_export_encryption_handler.PRIVATE_KEY_NAME))

        original_conn_path = h5.path_for_stored_index(conn)
        decrypted_conn_path, idx = (decrypted_file_paths[0], 0) if 'Connectivity' in decrypted_file_paths[0] else \
            (decrypted_file_paths[1], 1) if 'Connectivity' in decrypted_file_paths[1] else (decrypted_file_paths[2], 2)

        self.compare_files(original_conn_path, decrypted_conn_path)

        original_surface_path = h5.path_for_stored_index(surface_idx)
        del decrypted_file_paths[idx]
        decrypted_surface_path, idx = (decrypted_file_paths[0], 0) if 'Surface' in decrypted_file_paths[0] else \
            (decrypted_file_paths[1], 1)
        self.compare_files(original_surface_path, decrypted_surface_path)

        original_rm_path = h5.path_for_stored_index(region_mapping_index)
        del decrypted_file_paths[idx]
        self.compare_files(original_rm_path, decrypted_file_paths[0])

    def test_tvb_export_of_datatype_with_storage(self, dummy_datatype_index_factory):
        """
        Test export of a data type which has no data stored on file system
        """
        datatype = dummy_datatype_index_factory()
        _, file_path, _ = self.export_manager.export_data(datatype, self.TVB_EXPORTER, self.test_project)

        assert file_path is not None, "Export process should return path to export file"
        assert os.path.exists(file_path), "Could not find export file;: %s on disk." % file_path

    def test_export_datatype_with_links(self, region_mapping_index_factory, user_factory, project_factory):
        """
        This is a test for exporting region mapping with links, that results in importing:
        connectivity, surface and region mapping all from one zip.
        """
        self.test_user = user_factory()
        self.test_project = project_factory(self.test_user)

        region_mapping_index = region_mapping_index_factory()

        export_manager = ExportManager()
        _, exported_h5_file, _ = export_manager.export_data(region_mapping_index, self.TVB_LINKED_EXPORTER,
                                                            self.test_project)
        assert zipfile.is_zipfile(exported_h5_file), "Generated file is not a valid ZIP file"

        with zipfile.ZipFile(exported_h5_file, 'r') as zipObj:
            # Get list of files names in zip
            dts_in_zip = len(zipObj.namelist())
            assert 3 == dts_in_zip

            has_conn = False
            has_surface = False
            for dt_file in zipObj.namelist():
                if dt_file.find(region_mapping_index.fk_connectivity_gid) > -1:
                    has_conn = True
                if dt_file.find(region_mapping_index.fk_surface_gid):
                    has_surface = True

        assert has_conn is True, "Connectivity was exported in zip"
        assert has_surface is True, "Surface was exported in zip"

    def test_tvb_export_for_datatype_group(self, datatype_group_factory):
        """
        This method checks export of a data type group
        """
        ts_datatype_group, dm_datatype_group = datatype_group_factory(project=self.test_project, store_vm=True)
        file_name, file_path, _ = self.export_manager.export_data(dm_datatype_group, self.TVB_EXPORTER,
                                                                  self.test_project)

        assert file_name is not None, "Export process should return a file name"
        assert file_path is not None, "Export process should return path to export file"
        assert os.path.exists(file_path), "Could not find export file: %s on disk." % file_path

        # Now check if the generated file is a correct ZIP file
        assert zipfile.is_zipfile(file_path), "Generated file is not a valid ZIP file"

        with closing(zipfile.ZipFile(file_path)) as zip_file:
            list_of_files = zip_file.namelist()

            list_of_folders = []
            for file in list_of_files:
                dir_name = os.path.dirname(file)
                if dir_name not in list_of_folders:
                    list_of_folders.append(dir_name)

            count_datatypes = dao.count_datatypes_in_group(ts_datatype_group.id)
            count_datatypes += dao.count_datatypes_in_group(dm_datatype_group.id)

            # Check if ZIP files contains files for data types and view models (multiple H5 files in case of a Sim)
            assert count_datatypes == len(list_of_folders)
            assert (count_datatypes / 2) * 6 + (count_datatypes / 2) * 2 == len(list_of_files)

    def test_tvb_export_for_datatype_group_with_links(self, datatype_group_factory):
        """
        This method checks export of a data type group with Links
        """
        ts_datatype_group, dm_datatype_group = datatype_group_factory(project=self.test_project, store_vm=True,
                                                                      use_time_series_region=True)
        file_name, file_path, _ = self.export_manager.export_data(ts_datatype_group, self.TVB_LINKED_EXPORTER,
                                                                  self.test_project)

        assert file_name is not None, "Export process should return a file name"
        assert file_path is not None, "Export process should return path to export file"
        assert os.path.exists(file_path), "Could not find export file: %s on disk." % file_path

        # Now check if the generated file is a correct ZIP file
        assert zipfile.is_zipfile(file_path), "Generated file is not a valid ZIP file"

        with closing(zipfile.ZipFile(file_path)) as zip_file:
            list_of_files = zip_file.namelist()

            list_of_folders = []
            links_folder_found = False
            for file in list_of_files:
                dir_name = os.path.dirname(file)
                if not links_folder_found:
                    if "Links" in dir_name:
                        links_folder_found = True
                        assert file_path is not None, "Export process should return path to export file"

                if dir_name not in list_of_folders:
                    list_of_folders.append(dir_name)

            assert links_folder_found is not None, "Links folder was not exported"

            count_datatypes = dao.count_datatypes_in_group(ts_datatype_group.id)
            count_datatypes += dao.count_datatypes_in_group(dm_datatype_group.id)

            # Check if ZIP files contains files for data types and view models (multiple H5 files in case of a Sim)
            # +1 For Links folder
            assert count_datatypes + 1 == len(list_of_folders)
            # +3 for the 3 files in Links folder: Connectivity, Surface, Region Mapping
            # time series have 6 files, datatype measures have 2 files
            assert (count_datatypes / 2) * 6 + (count_datatypes / 2) * 2 + 3 == len(list_of_files)

    def test_tvb_export_for_encrypted_datatype_group_with_links(self, datatype_group_factory):
        """
        This method checks export of an encrypted data type group
        """

        ts_datatype_group, dm_datatype_group = datatype_group_factory(project=self.test_project, store_vm=True)

        storage_interface = StorageInterface()
        import_export_encryption_handler = StorageInterface.get_import_export_encryption_handler()
        import_export_encryption_handler.generate_public_private_key_pair(TvbProfile.current.TVB_TEMP_FOLDER)

        file_name, file_path, _ = self.export_manager.export_data(
            dm_datatype_group, self.TVB_EXPORTER, self.test_project, os.path.join(
                TvbProfile.current.TVB_TEMP_FOLDER, import_export_encryption_handler.PUBLIC_KEY_NAME))

        assert file_name is not None, "Export process should return a file name"
        assert file_path is not None, "Export process should return path to export file"
        assert os.path.exists(file_path), "Could not find export file: %s on disk." % file_path

        # Now check if the generated file is a correct ZIP file
        assert zipfile.is_zipfile(file_path), "Generated file is not a valid ZIP file"

        result = storage_interface.unpack_zip(file_path, TvbProfile.current.TVB_TEMP_FOLDER)
        encrypted_password = import_export_encryption_handler.extract_encrypted_password_from_list(result)
        decrypted_file_paths = import_export_encryption_handler.decrypt_content(
            encrypted_password, result, os.path.join(TvbProfile.current.TVB_TEMP_FOLDER,
                                                     import_export_encryption_handler.PRIVATE_KEY_NAME))
        # Here we only test if the length of decrypted_file_paths is the one expected
        assert len(decrypted_file_paths) == len(result), "Number of decrypted data type group files is not correct!"

    def test_export_with_invalid_data(self, dummy_datatype_index_factory):
        """
        Test scenarios when data provided to export method is invalid
        """
        # Test with no datatype
        with pytest.raises(InvalidExportDataException):
            self.export_manager.export_data(None, self.TVB_EXPORTER, self.test_project)
        # Test with no exporter 
        datatype = dummy_datatype_index_factory()
        with pytest.raises(ExportException):
            self.export_manager.export_data(datatype, None, self.test_project)

        # test with wrong exporter
        with pytest.raises(ExportException):
            self.export_manager.export_data(datatype, "wrong_exporter", self.test_project)

        # test with no project folder
        with pytest.raises(ExportException):
            self.export_manager.export_data(datatype, self.TVB_EXPORTER, None)

    def test_export_project_failure(self):
        """
        This method tests export of project with None data
        """
        with pytest.raises(ExportException):
            self.export_manager.export_project(None)

    def test_export_project(self, project_factory, user_factory):
        """
        Test export of a project
        """
        user = user_factory(username='test_user2')
        project = project_factory(user)
        export_file = self.export_manager.export_project(project)

        assert export_file is not None, "Export process should return path to export file"
        assert os.path.exists(export_file), "Could not find export file: %s on disk." % export_file
        # Now check if the generated file is a correct ZIP file
        assert zipfile.is_zipfile(export_file), "Generated file is not a valid ZIP file"

    def test_export_simulator_configuration(self, operation_factory, connectivity_index_factory):
        """
        Test export of a simulator configuration
        """
        conn_gid = uuid.UUID(connectivity_index_factory().gid)
        operation = operation_factory(is_simulation=True, store_vm=True, test_project=self.test_project,
                                      conn_gid=conn_gid)

        burst_configuration = BurstConfiguration(self.test_project.id)
        burst_configuration.fk_simulation = operation.id
        burst_configuration.simulator_gid = operation.view_model_gid
        burst_configuration.name = "Test_burst"
        burst_configuration = dao.store_entity(burst_configuration)

        BurstService().store_burst_configuration(burst_configuration)

        export_file = self.export_manager.export_simulator_configuration(burst_configuration.id)

        assert export_file is not None, "Export process should return path to export file"
        assert os.path.exists(export_file), "Could not find export file: %s on disk." % export_file
        assert zipfile.is_zipfile(export_file), "Generated file is not a valid ZIP file"
