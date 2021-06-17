# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
This module is an interface for the tvb storage module.
All calls to methods from this module must be done through this class.

.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

from datetime import datetime
import os

from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.storage.h5.encryption.data_encryption_handler import DataEncryptionHandler, FoldersQueueConsumer, \
    encryption_handler
from tvb.storage.h5.encryption.encryption_handler import EncryptionHandler
from tvb.storage.h5.file.exceptions import RenameWhileSyncEncryptingException
from tvb.storage.h5.file.files_helper import FilesHelper, TvbZip
from tvb.storage.h5.file.hdf5_storage_manager import HDF5StorageManager
from tvb.storage.h5.file.xml_metadata_handlers import XMLReader, XMLWriter


class StorageInterface:
    TEMP_FOLDER = "TEMP"
    IMAGES_FOLDER = "IMAGES"
    PROJECTS_FOLDER = "PROJECTS"

    ROOT_NODE_PATH = "/"
    TVB_FILE_EXTENSION = XMLWriter.FILE_EXTENSION
    TVB_STORAGE_FILE_EXTENSION = ".h5"
    TVB_ZIP_FILE_EXTENSION = ".zip"

    TVB_PROJECT_FILE = "Project" + TVB_FILE_EXTENSION

    ZIP_FILE_EXTENSION = "zip"

    OPERATION_FOLDER_PREFIX = "Operation_"

    logger = get_logger(__name__)

    def __init__(self):
        self.files_helper = FilesHelper()
        self.data_encryption_handler = encryption_handler
        self.folders_queue_consumer = FoldersQueueConsumer()

        # object attributes which have parameters in their constructor will be lazily instantiated
        self.tvb_zip = None
        self.storage_manager = None
        self.xml_reader = None
        self.xml_writer = None
        self.encryption_handler = None

    # FilesHelper methods start here #

    def check_created(self, path=TvbProfile.current.TVB_STORAGE):
        self.files_helper.check_created(path)

    @staticmethod
    def get_projects_folder():
        return FilesHelper.get_projects_folder()

    def get_project_folder(self, project_name, *sub_folders):
        return self.files_helper.get_project_folder(project_name, *sub_folders)

    def get_temp_folder(self, project_name):
        return self.files_helper.get_project_folder(project_name, self.TEMP_FOLDER)

    def remove_project_structure(self, project_name):
        self.files_helper.remove_project_structure(project_name)

    def get_project_meta_file_path(self, project_name):
        return self.files_helper.get_project_meta_file_path(project_name)

    def read_project_metadata(self, project_path):
        return FilesHelper.read_project_metadata(project_path, self.TVB_PROJECT_FILE)

    def write_project_metadata_from_dict(self, project_path, meta_entity):
        FilesHelper.write_project_metadata_from_dict(project_path, meta_entity, self.TVB_PROJECT_FILE)

    def write_project_metadata(self, meta_dictionary):
        self.files_helper.write_project_metadata(meta_dictionary, self.TVB_PROJECT_FILE)

    def remove_operation_data(self, project_name, operation_id):
        self.files_helper.remove_operation_data(project_name, operation_id)

    def remove_datatype_file(self, h5_file):
        self.files_helper.remove_datatype_file(h5_file)
        self.push_folder_to_sync(FilesHelper.get_project_folder_from_h5(h5_file))

    def get_images_folder(self, project_name):
        return self.files_helper.get_images_folder(project_name, self.IMAGES_FOLDER)

    def write_image_metadata(self, figure, meta_entity):
        self.files_helper.write_image_metadata(figure, meta_entity, self.IMAGES_FOLDER)

    def remove_image_metadata(self, figure):
        self.files_helper.remove_image_metadata(figure, self.IMAGES_FOLDER)

    def get_allen_mouse_cache_folder(self, project_name):
        return self.files_helper.get_allen_mouse_cache_folder(project_name)

    def get_tumor_dataset_folder(self):
        return self.files_helper.get_tumor_dataset_folder()

    def zip_folders(self, all_datatypes, project_name, zip_full_path):
        operation_folders = []
        for data_type in all_datatypes:
            operation_folder = self.get_project_folder(project_name, str(data_type.fk_from_operation))
            operation_folders.append(operation_folder)
        FilesHelper.zip_folders(zip_full_path, operation_folders, self.OPERATION_FOLDER_PREFIX)

    @staticmethod
    def zip_folder(result_name, folder_root):
        FilesHelper.zip_folder(result_name, folder_root)

    def unpack_zip(self, uploaded_zip, folder_path):
        return self.files_helper.unpack_zip(uploaded_zip, folder_path)

    @staticmethod
    def copy_file(source, dest, dest_postfix=None, buffer_size=1024 * 1024):
        FilesHelper.copy_file(source, dest, dest_postfix, buffer_size)

    @staticmethod
    def remove_files(file_list, ignore_exception=False):
        FilesHelper.remove_files(file_list, ignore_exception)

    @staticmethod
    def remove_folder(folder_path, ignore_errors=False):
        FilesHelper.remove_folder(folder_path, ignore_errors)

    @staticmethod
    def compute_size_on_disk(file_path):
        return FilesHelper.compute_size_on_disk(file_path)

    @staticmethod
    def compute_recursive_h5_disk_usage(start_path='.'):
        return FilesHelper.compute_recursive_h5_disk_usage(start_path)

    # TvbZip methods start here #

    def write_zip_folder(self, dest_path, folder, exclude=None):
        self.tvb_zip = TvbZip(dest_path, "w")
        self.tvb_zip.write_zip_folder(folder, exclude)
        self.tvb_zip.close()

    def write_zip_folder_with_links(self, dest_path, folder, linked_paths, op, exclude=None):
        self.tvb_zip = TvbZip(dest_path, "w")
        self.tvb_zip.write_zip_folder(folder, exclude)

        self.logger.debug("Done exporting files, now we will export linked DTs")

        if linked_paths is not None:
            self.export_datatypes(linked_paths, op)

        self.tvb_zip.close()

    def get_filenames_in_zip(self, dest_path, mode="r"):
        self.tvb_zip = TvbZip(dest_path, mode)
        name_list = self.tvb_zip.namelist()
        self.tvb_zip.close()
        return name_list

    def open_tvb_zip(self, dest_path, name, mode="r"):
        self.tvb_zip = TvbZip(dest_path, mode)
        file = self.tvb_zip.open(name)
        self.tvb_zip.close()
        return file

    # Return a HDF5 storage methods to call the methods from there #

    @staticmethod
    def get_storage_manager(file_full_path):
        return HDF5StorageManager(file_full_path)

    # XML methods start here #

    def read_metadata_from_xml(self, xml_path):
        self.xml_reader = XMLReader(xml_path)
        return self.xml_reader.read_metadata_from_xml()

    def write_metadata_in_xml(self, entity, final_path):
        self.xml_writer = XMLWriter(entity)
        return self.xml_writer.write_metadata_in_xml(final_path)

    # Encryption Handler methods start here #

    def cleanup_encryption_handler(self, dir_gid):
        self.encryption_handler = EncryptionHandler(dir_gid)
        self.encryption_handler.cleanup_encryption_handler()

    @staticmethod
    def generate_random_password(pass_size):
        return EncryptionHandler.generate_random_password(pass_size)

    def get_encrypted_dir(self, dir_gid):
        self.encryption_handler = EncryptionHandler(dir_gid)
        return self.encryption_handler.get_encrypted_dir()

    def get_password_file(self, dir_gid):
        self.encryption_handler = EncryptionHandler(dir_gid)
        return self.encryption_handler.get_password_file()

    def encrypt_inputs(self, dir_gid, files_to_encrypt, subdir=None):
        self.encryption_handler = EncryptionHandler(dir_gid)
        return self.encryption_handler.encrypt_inputs(files_to_encrypt, subdir)

    def decrypt_results_to_dir(self, dir_gid, dir, from_subdir=None):
        self.encryption_handler = EncryptionHandler(dir_gid)
        return self.encryption_handler.decrypt_results_to_dir(dir, from_subdir)

    def decrypt_files_to_dir(self, dir_gid, files, dir):
        self.encryption_handler = EncryptionHandler(dir_gid)
        return self.encryption_handler.decrypt_files_to_dir(files, dir)

    def get_current_enc_dirname(self, dir_gid):
        self.encryption_handler = EncryptionHandler(dir_gid)
        return self.encryption_handler.current_enc_dirname

    # Data Encryption Handler methods start here #

    def inc_project_usage_count(self, folder):
        return self.data_encryption_handler.inc_project_usage_count(folder)

    def inc_running_op_count(self, folder):
        return self.data_encryption_handler.inc_running_op_count(folder)

    def dec_running_op_count(self, folder):
        return self.data_encryption_handler.dec_running_op_count(folder)

    def check_and_delete(self, folder):
        self.data_encryption_handler.dec_running_op_count(folder)
        return self.data_encryption_handler.check_and_delete(folder)

    @staticmethod
    def sync_folders(folder):
        DataEncryptionHandler.sync_folders(folder)

    def set_project_active(self, project, linked_dt=None):
        self.data_encryption_handler.set_project_active(project, linked_dt)

    def set_project_inactive(self, project):
        self.data_encryption_handler.set_project_inactive(project.name)

    def push_folder_to_sync(self, project_name):
        project_folder = self.get_project_folder(project_name)
        self.data_encryption_handler.push_folder_to_sync(project_folder)

    @staticmethod
    def encryption_enabled():
        return DataEncryptionHandler.encryption_enabled()

    @staticmethod
    def startup_cleanup():
        return DataEncryptionHandler.startup_cleanup()

    # Folders Queue Consumer methods start here #

    def run(self):
        return self.folders_queue_consumer.run()

    def start(self):
        self.folders_queue_consumer.start()

    def mark_stop(self):
        self.folders_queue_consumer.mark_stop()

    def join(self):
        self.folders_queue_consumer.join()

    # Generic methods start here

    def ends_with_tvb_file_extension(self, file):
        return file.endswith(self.TVB_FILE_EXTENSION)

    def ends_with_tvb_storage_file_extension(self, file):
        return file.endswith(self.TVB_STORAGE_FILE_EXTENSION)

    def is_in_usage(self, project_folder):
        return self.data_encryption_handler.is_in_usage(project_folder)

    def rename_project(self, current_proj_name, new_name):
        project_folder = self.get_project_folder(current_proj_name)
        if self.encryption_enabled() and not self.data_encryption_handler.is_in_usage(project_folder):
            raise RenameWhileSyncEncryptingException(
                "A project can not be renamed while sync encryption operations are running")
        self.files_helper.rename_project_structure(current_proj_name, new_name)
        encrypted_path = DataEncryptionHandler.compute_encrypted_folder_path(project_folder)
        if os.path.exists(encrypted_path):
            new_encrypted_path = DataEncryptionHandler.compute_encrypted_folder_path(
                self.get_project_folder(new_name))
            os.rename(encrypted_path, new_encrypted_path)

    def remove_project(self, project):
        project_folder = self.get_project_folder(project.name)
        self.remove_project_structure(project.name)
        encrypted_path = DataEncryptionHandler.compute_encrypted_folder_path(project_folder)
        if os.path.exists(encrypted_path):
            self.remove_folder(encrypted_path)
        if os.path.exists(DataEncryptionHandler.project_key_path(project.id)):
            os.remove(DataEncryptionHandler.project_key_path(project.id))

    def move_datatype_with_sync(self, to_project, to_project_path, new_op_id, full_path, vm_full_path):
        self.set_project_active(to_project)
        self.sync_folders(to_project_path)

        self.files_helper.move_datatype(to_project.name, str(new_op_id), full_path)
        self.files_helper.move_datatype(to_project.name, str(new_op_id), vm_full_path)

        self.sync_folders(to_project_path)
        self.set_project_inactive(to_project)

    def export_datatypes(self, paths, operation):
        op_folder = self.get_project_folder(operation.project.name, operation.id)
        op_folder_name = os.path.basename(op_folder)

        # add linked datatypes to archive in the import operation
        for pth in paths:
            zip_pth = op_folder_name + '/' + os.path.basename(pth)
            self.tvb_zip.write(pth, zip_pth)

        # remove these files, since we only want them in export archive
        self.remove_folder(op_folder)

    def build_data_export_folder(self, data, export_folder):
        now = datetime.now()
        date_str = "%d-%d-%d_%d-%d-%d_%d" % (now.year, now.month, now.day, now.hour,
                                             now.minute, now.second, now.microsecond)
        tmp_str = date_str + "@" + data.gid
        data_export_folder = os.path.join(export_folder, tmp_str)
        self.check_created(data_export_folder)

        return data_export_folder

    def export_project(self, project, folders_to_exclude, export_folder, linked_paths, op):
        project_folder = self.get_project_folder(project.name)
        folders_to_exclude.append("TEMP")

        # Compute path and name of the zip file
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M")
        zip_file_name = "%s_%s.%s" % (date_str, project.name, self.ZIP_FILE_EXTENSION)

        export_folder = self.build_data_export_folder(project, export_folder)
        result_path = os.path.join(export_folder, zip_file_name)

        # Pack project [filtered] content into a ZIP file:
        self.logger.debug("Done preparing, now we will write the folder.")
        self.logger.debug(project_folder)
        self.write_zip_folder_with_links(result_path, project_folder, linked_paths, op, folders_to_exclude)

        # Make sure the Project.xml file gets copied:
        self.logger.debug("Done, closing")

        return result_path
