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
This module is an interface for the tvb storage module.
All calls to methods from this module must be done through this class.

.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

import os
import uuid
from datetime import datetime
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.storage.h5.encryption.data_encryption_handler import FoldersQueueConsumer, encryption_handler
from tvb.storage.h5.encryption.encryption_handler import EncryptionHandler
from tvb.storage.h5.encryption.import_export_encryption_handler import ImportExportEncryptionHandler
from tvb.storage.h5.file.exceptions import RenameWhileSyncEncryptingException, FileStructureException
from tvb.storage.h5.file.files_helper import FilesHelper, TvbZip
from tvb.storage.h5.file.hdf5_storage_manager import HDF5StorageManager
from tvb.storage.h5.file.xml_metadata_handlers import XMLReader, XMLWriter


class StorageInterface:
    TEMP_FOLDER = "TEMP"
    IMAGES_FOLDER = "IMAGES"
    PROJECTS_FOLDER = "PROJECTS"

    ROOT_NODE_PATH = "/"
    TVB_STORAGE_FILE_EXTENSION = ".h5"
    TVB_ZIP_FILE_EXTENSION = ".zip"
    TVB_XML_FILE_EXTENSION = ".xml"

    TVB_PROJECT_FILE = "Project" + TVB_XML_FILE_EXTENSION
    FILE_NAME_STRUCTURE = '{}_{}.h5'
    OPERATION_FOLDER_PREFIX = "Operation_"

    EXPORTED_SIMULATION_NAME = "exported_simulation"
    EXPORTED_SIMULATION_DTS_DIR = "datatypes"

    EXPORT_FOLDER_NAME = "EXPORT_TMP"
    EXPORT_FOLDER = os.path.join(TvbProfile.current.TVB_STORAGE, EXPORT_FOLDER_NAME)

    logger = get_logger(__name__)

    def __init__(self):
        self.files_helper = FilesHelper()
        self.data_encryption_handler = encryption_handler
        self.folders_queue_consumer = FoldersQueueConsumer()
        self.import_export_encryption_handler = ImportExportEncryptionHandler()

        # object attributes which have parameters in their constructor will be lazily instantiated
        self.tvb_zip = None
        self.xml_reader = None
        self.xml_writer = None

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

    def get_images_folder(self, project_name):
        return self.files_helper.get_images_folder(project_name, self.IMAGES_FOLDER)

    def write_image_metadata(self, figure, meta_entity):
        self.files_helper.write_image_metadata(figure, meta_entity, self.IMAGES_FOLDER)

    def remove_figure(self, figure):
        self.files_helper.remove_figure(figure, self.IMAGES_FOLDER)
        self.push_folder_to_sync(figure.project.name)

    def get_allen_mouse_cache_folder(self, project_name):
        return self.files_helper.get_allen_mouse_cache_folder(project_name)

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

    def write_zip_folder(self, dest_path, folder, linked_paths=None, op=None, exclude=[]):
        self.tvb_zip = TvbZip(dest_path, "w")
        self.tvb_zip.write_zip_folder(folder, exclude)

        self.logger.debug("Done exporting files, now we will export linked DTs")

        if linked_paths is not None and op is not None:
            self.__export_datatypes(linked_paths, op)

        self.tvb_zip.close()

    def unpack_zip(self, uploaded_zip, folder_path):
        self.tvb_zip = TvbZip(uploaded_zip, "r")
        unpacked_folder = self.tvb_zip.unpack_zip(folder_path)
        self.tvb_zip.close()
        return unpacked_folder

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

    # Return a HDF5StorageManager object to call the methods from there #

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

    # Method for preparing encryption
    def prepare_encryption(self, project_name):
        # Generate path to public_key
        temp_folder = self.get_temp_folder(project_name)
        public_key_file_name = "public_key_" + uuid.uuid4().hex + ".pem"
        public_key_file_path = os.path.join(temp_folder, public_key_file_name)

        # Generate a random password for the files
        password = EncryptionHandler.generate_random_password()

        return public_key_file_path, password

    # Return an EncryptionHandler object to call the methods from there #
    @staticmethod
    def get_encryption_handler(dir_gid):
        return EncryptionHandler(dir_gid)

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

    def sync_folders(self, folder):
        self.data_encryption_handler.sync_folders(folder)

    def set_project_active(self, project, linked_dt=None):
        self.data_encryption_handler.set_project_active(project, linked_dt)

    def set_project_inactive(self, project):
        self.data_encryption_handler.set_project_inactive(project.name)

    def push_folder_to_sync(self, project_name):
        project_folder = self.get_project_folder(project_name)
        self.data_encryption_handler.push_folder_to_sync(project_folder)

    @staticmethod
    def encryption_enabled():
        return encryption_handler.encryption_enabled()

    @staticmethod
    def app_encryption_handler():
        return encryption_handler.app_encryption_handler()

    def startup_cleanup(self):
        return self.data_encryption_handler.startup_cleanup()

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

    def get_filename(self, class_name, gid):
        return self.FILE_NAME_STRUCTURE.format(class_name, gid.hex)

    def path_for(self, op_id, h5_file_class, gid, project_name, dt_class):
        operation_dir = self.files_helper.get_project_folder(project_name, str(op_id))
        return self.path_by_dir(operation_dir, h5_file_class, gid, dt_class)

    def path_by_dir(self, base_dir, h5_file_class, gid, dt_class):
        if isinstance(gid, str):
            gid = uuid.UUID(gid)
        fname = self.get_filename(dt_class or h5_file_class.file_name_base(), gid)
        return os.path.join(base_dir, fname)

    def ends_with_tvb_file_extension(self, file):
        return file.endswith(self.TVB_XML_FILE_EXTENSION)

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
        encrypted_path = self.data_encryption_handler.compute_encrypted_folder_path(project_folder)
        if os.path.exists(encrypted_path):
            new_encrypted_path = self.data_encryption_handler.compute_encrypted_folder_path(
                self.get_project_folder(new_name))
            os.rename(encrypted_path, new_encrypted_path)

    def remove_project(self, project, sync_for_encryption=False):
        project_folder = self.get_project_folder(project.name)
        if sync_for_encryption:
            self.sync_folders(project_folder)
        try:
            self.remove_folder(project_folder)
            self.logger.debug("Project folders were removed for " + project.name)
        except OSError:
            self.logger.exception("A problem occurred while removing folder.")
            raise FileStructureException("Permission denied. Make sure you have write access on TVB folder!")

        encrypted_path = self.data_encryption_handler.compute_encrypted_folder_path(project_folder)
        FilesHelper.remove_files([encrypted_path, self.data_encryption_handler.project_key_path(project.id)], True)

    def move_datatype_with_sync(self, to_project, to_project_path, new_op_id, path_list):
        self.set_project_active(to_project)
        self.sync_folders(to_project_path)

        for path in path_list:
            self.files_helper.move_datatype(to_project.name, str(new_op_id), path)

        self.sync_folders(to_project_path)
        self.set_project_inactive(to_project)

    # Return an ImportExportEncryptionHandler object  to call the methods from there #

    @staticmethod
    def get_import_export_encryption_handler():
        return ImportExportEncryptionHandler()

    # Exporting related methods start here

    def __export_datatypes(self, paths, operation):
        op_folder = self.get_project_folder(operation.project.name, operation.id)
        op_folder_name = os.path.basename(op_folder)

        # add linked datatypes to archive in the import operation
        for pth in paths:
            zip_pth = op_folder_name + '/' + os.path.basename(pth)
            self.tvb_zip.write(pth, zip_pth)

        # remove these files, since we only want them in export archive
        self.remove_folder(op_folder)

    def __build_data_export_folder(self, data, export_folder):
        now = datetime.now()
        date_str = "%d-%d-%d_%d-%d-%d_%d" % (now.year, now.month, now.day, now.hour,
                                             now.minute, now.second, now.microsecond)
        tmp_str = date_str + "@" + data.gid
        data_export_folder = os.path.join(export_folder, tmp_str)
        self.check_created(data_export_folder)

        return data_export_folder

    def export_project(self, project, folders_to_exclude, linked_paths, op):
        """
        This method is used to export a project as a ZIP file.
        :param project: project to be exported.
        :param folders_to_exclude: a list of paths to folders inside of a project folder which should not be exported.
        :param linked_paths: a list of links to datatypes for the project to be exported
        :param op: operation for links to exported datatypes (if any)
        """

        project_folder = self.get_project_folder(project.name)
        folders_to_exclude.append("TEMP")

        # Compute path and name of the zip file
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M")
        zip_file_name = "%s_%s.%s" % (date_str, project.name, self.TVB_ZIP_FILE_EXTENSION)

        export_folder = self.__build_data_export_folder(project, self.EXPORT_FOLDER)
        result_path = os.path.join(export_folder, zip_file_name)

        # Pack project [filtered] content into a ZIP file:
        self.logger.debug("Done preparing, now we will write the folder.")
        self.logger.debug(project_folder)
        self.write_zip_folder(result_path, project_folder, linked_paths, op, folders_to_exclude)

        # Make sure the Project.xml file gets copied:
        self.logger.debug("Done, closing")

        return result_path

    def export_simulator_configuration(self, burst, all_view_model_paths, all_datatype_paths, zip_filename):
        """
        This method is used to export a simulator configuration as a ZIP file
        :param burst: BurstConfiguration of the simulation to be exported
        :param all_view_model_paths: a list of paths to all view model files of the simulation
        :param all_datatype_paths: a list of paths to all datatype files of the simulation
        :param zip_filename: name of the file to be exported
        """

        tmp_export_folder = self.__build_data_export_folder(burst, self.EXPORT_FOLDER)
        tmp_sim_folder = os.path.join(tmp_export_folder, self.EXPORTED_SIMULATION_NAME)

        if not os.path.exists(tmp_sim_folder):
            os.makedirs(tmp_sim_folder)

        for vm_path in all_view_model_paths:
            dest = os.path.join(tmp_sim_folder, os.path.basename(vm_path))
            self.copy_file(vm_path, dest)

        for dt_path in all_datatype_paths:
            dest = os.path.join(tmp_sim_folder, self.EXPORTED_SIMULATION_DTS_DIR, os.path.basename(dt_path))
            self.copy_file(dt_path, dest)

        result_path = os.path.join(tmp_export_folder, zip_filename)
        self.write_zip_folder(result_path, tmp_sim_folder)
        self.remove_folder(tmp_sim_folder)

        return result_path

    def __copy_datatypes(self, dt_path_list, data, password):
        export_folder = self.__build_data_export_folder(data, self.EXPORT_FOLDER)

        for dt_path in dt_path_list:
            file_destination = os.path.join(export_folder, os.path.basename(dt_path))
            if not os.path.exists(file_destination):
                self.copy_file(dt_path, file_destination)
                self.get_storage_manager(file_destination).remove_metadata('parent_burst', check_existence=True)

                if password is not None:
                    self.import_export_encryption_handler.encrypt_data_at_export(file_destination, password)
                    os.remove(file_destination)

        return export_folder

    def export_datatypes(self, dt_path_list, data, download_file_name, public_key_path=None, password=None):
        """
        This method is used to export a list of datatypes as a ZIP file.
        :param dt_path_list: a list of paths to be exported (there are more than one when exporting with links)
        :param data: data to be exported
        :param download_file_name: name of the zip file to be downloaded
        :param public_key_path: path to public key that will be used for encrypting the password by TVB
        :param password: password used for encrypting the files before exporting
        """

        export_folder = self.__copy_datatypes(dt_path_list, data, password)

        if len(dt_path_list) == 1 and password is None:
            return os.path.join(export_folder, os.path.basename(dt_path_list[0]))

        if password is not None:
            download_file_name = download_file_name.replace('.h5', '.zip')
            self.import_export_encryption_handler.encrypt_and_save_password(public_key_path, password, export_folder)

        export_data_zip_path = os.path.join(os.path.dirname(export_folder), download_file_name)
        self.write_zip_folder(export_data_zip_path, export_folder)
        return export_data_zip_path

    def export_datatypes_structure(self, op_file_dict, data, download_file_name, public_key_path, password,
                                   links_tuple_for_copy=None):
        """
        This method is used to export a list of datatypes as a ZIP file, while preserving the folder structure
        (eg: operation folders). It is only used during normal tvb exporting for datatype groups.
        :param op_file_dict: a dictionary where keys are operation folders and the values are lists of files inside
            that operation folder
        :param data: data to be exported
        :param download_file_name: name of the ZIP file to be exported
        :param public_key_path: path to public key that will be used for encrypting the password by TVB
        :param password: password used for encrypting the files before exporting
        :param links_tuple_for_copy: a tuple containing two elements: a list of paths to be copied and the first
         datatype of the group

        """
        if links_tuple_for_copy is not None:
            export_folder = self.__copy_datatypes(links_tuple_for_copy[0], links_tuple_for_copy[1], password)
        else:
            export_folder = self.__build_data_export_folder(data, self.EXPORT_FOLDER)

        for op_folder, files in op_file_dict.items():
            tmp_op_folder_path = os.path.join(export_folder, os.path.basename(op_folder))
            for file in files:
                dest_path = os.path.join(tmp_op_folder_path, os.path.basename(file))

                if not os.path.exists(dest_path):
                    self.copy_file(file, os.path.join(tmp_op_folder_path, os.path.basename(file)))
                    self.get_storage_manager(dest_path).remove_metadata('parent_burst', check_existence=True)

                    if password is not None:
                        self.import_export_encryption_handler.encrypt_data_at_export(dest_path, password)
                        os.remove(dest_path)

        dest_path = os.path.join(os.path.dirname(export_folder), download_file_name)
        if password is not None:
            self.import_export_encryption_handler.encrypt_and_save_password(public_key_path, password, export_folder)
        self.write_zip_folder(dest_path, export_folder)

        return dest_path

    def export_datatype_from_rest_server(self, dt, data, download_file_name, public_key_path):
        password = EncryptionHandler.generate_random_password()
        dest_path = self.export_datatypes([dt], data, download_file_name, public_key_path, password)

        return dest_path

    def import_datatype_to_rest_client(self, file_path, temp_folder, private_key_path):
        import_export_encryption_handler = self.get_import_export_encryption_handler()
        result = self.unpack_zip(file_path, temp_folder)
        encrypted_password_path = import_export_encryption_handler.extract_encrypted_password_from_list(result)

        decrypted_file_path = import_export_encryption_handler.decrypt_content(encrypted_password_path,
                                                                               result, private_key_path)[0]
        os.remove(file_path)
        return decrypted_file_path
