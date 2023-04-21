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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
import shutil
from threading import Lock
from zipfile import ZipFile, ZIP_DEFLATED, BadZipfile
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.storage.h5.decorators import synchronized
from tvb.storage.h5.file.exceptions import FileStructureException
from tvb.storage.h5.file.xml_metadata_handlers import XMLWriter, XMLReader

LOCK_CREATE_FOLDER = Lock()


class FilesHelper(object):
    """
    This class manages all Structure related operations, using File storage.
    It will handle creating meaning-full entities and retrieving existent ones.
    """
    PROJECTS_FOLDER = "PROJECTS"
    ALLEN_MOUSE_CONNECTIVITY_CACHE_FOLDER = "ALLEN_MOUSE_CONNECTIVITY_CACHE"

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)

    # PROJECT RELATED methods ##################################

    @synchronized(LOCK_CREATE_FOLDER)
    def check_created(self, path):
        """
        Check that the given folder exists, otherwise create it, with the entire tree of parent folders.
        This method is synchronized, for parallel access from events, to avoid conflicts.
        """
        try:
            # if this is meant to be used concurrently it might be better to catch OSError 17 then checking exists
            if not os.path.exists(path):
                self.logger.debug("Creating folder:" + str(path))
                os.makedirs(path, mode=TvbProfile.current.ACCESS_MODE_TVB_FILES)
                os.chmod(path, TvbProfile.current.ACCESS_MODE_TVB_FILES)
        except OSError:
            self.logger.exception("COULD NOT CREATE FOLDER! CHECK ACCESS ON IT!")
            raise FileStructureException("Could not create Folder" + str(path))

    @staticmethod
    def get_projects_folder():
        base_path = TvbProfile.current.TVB_STORAGE
        if TvbProfile.current.web.ENCRYPT_STORAGE and TvbProfile.current.web.CAN_ENCRYPT_STORAGE \
                and TvbProfile.current.web.DECRYPT_PATH:
            base_path = TvbProfile.current.web.DECRYPT_PATH

        return os.path.join(base_path, FilesHelper.PROJECTS_FOLDER)

    def get_project_folder(self, project_name, *sub_folders):
        """
        Retrieve the root path for the given project.
        If root folder is not created yet, will create it.
        """
        complete_path = os.path.join(self.get_projects_folder(), project_name)
        if sub_folders is not None:
            complete_path = os.path.join(complete_path, *sub_folders)
        if not os.path.exists(complete_path):
            self.check_created(complete_path)
        return complete_path

    @staticmethod
    def get_project_folder_from_h5(h5_file):
        op_folder = os.path.dirname(h5_file)
        return os.path.dirname(op_folder)

    def rename_project_structure(self, project_name, new_name):
        """ Rename Project folder or THROW FileStructureException. """
        try:
            path = self.get_project_folder(project_name)
            folder = os.path.split(path)[0]
            new_full_name = os.path.join(folder, new_name)

            if os.path.exists(new_full_name):
                raise IOError("Path exists %s " % new_full_name)
            os.rename(path, new_full_name)
            return path, new_full_name
        except Exception:
            self.logger.exception("Could not rename node!")
            raise FileStructureException("Could not rename to %s" % new_name)

    def get_project_meta_file_path(self, project_name, project_file):
        """
        Retrieve project meta info file path.
        
        :returns: File path for storing Project meta-data
            File might not exist yet, but parent folder is created after this method call.
            
        """
        complete_path = self.get_project_folder(project_name)
        complete_path = os.path.join(complete_path, project_file)
        return complete_path

    @staticmethod
    def read_project_metadata(project_path, project_file):
        project_cfg_file = os.path.join(project_path, project_file)
        return XMLReader(project_cfg_file).read_metadata_from_xml()

    @staticmethod
    def write_project_metadata_from_dict(project_path, meta_entity, project_file):
        project_cfg_file = os.path.join(project_path, project_file)
        XMLWriter(meta_entity).write_metadata_in_xml(project_cfg_file)
        os.chmod(project_path, TvbProfile.current.ACCESS_MODE_TVB_FILES)

    def write_project_metadata(self, meta_dictionary, project_file):
        """
        :param meta_dictionary: Project metadata
        :param project_file: Project file path
        """
        proj_path = self.get_project_folder(meta_dictionary['name'])
        self.write_project_metadata_from_dict(proj_path, meta_dictionary, project_file)

    # OPERATION related METHODS Start Here #########################

    def remove_operation_data(self, project_name, operation_id):
        """
        Remove H5 storage fully.
        """
        try:
            complete_path = self.get_project_folder(project_name, str(operation_id))
            self.logger.debug("Removing: " + str(complete_path))
            self.remove_folder(complete_path, False)
        except FileStructureException:
            self.logger.exception("Could not remove files")
            raise FileStructureException("Could not remove files for OP" + str(operation_id))

    # DATA-TYPES METHODS Start Here #####################
    def move_datatype(self, new_project_name, new_op_id, full_path):
        """
        Move H5 storage into a new location
        """
        try:
            folder = self.get_project_folder(new_project_name, str(new_op_id))
            full_new_file = os.path.join(folder, os.path.split(full_path)[1])
            os.rename(full_path, full_new_file)
        except Exception:
            self.logger.exception("Could not move file")
            raise FileStructureException("Could not move " + full_path)

    # IMAGES METHODS Start Here #######################
    def get_images_folder(self, project_name, images_folder):
        """
        Computes the name/path of the folder where to store images.
        """
        project_folder = self.get_project_folder(project_name)
        images_folder = os.path.join(project_folder, images_folder)
        self.check_created(images_folder)
        return images_folder

    def write_image_metadata(self, figure, meta_entity, images_folder):
        """
        Writes figure meta-data into XML file
        """
        _, dict_data = figure.to_dict()
        XMLWriter(meta_entity).write_metadata_in_xml(self._compute_image_metadata_file(figure, images_folder))

    def remove_figure(self, figure, images_folder):
        """
        Remove the file storing image and its meta data
        """
        figures_folder = self.get_images_folder(figure.project.name, images_folder)
        path2figure = os.path.join(figures_folder, figure.file_path)
        metadata_file = self._compute_image_metadata_file(figure, images_folder)
        self.remove_files([path2figure, metadata_file])

    def _compute_image_metadata_file(self, figure, images_folder):
        """
        Computes full path of image meta data XML file. 
        """
        name = figure.file_path.split('.')[0]
        images_folder = self.get_images_folder(figure.project.name, images_folder)
        return os.path.join(TvbProfile.current.TVB_STORAGE, images_folder, name + XMLWriter.FILE_EXTENSION)

    def get_allen_mouse_cache_folder(self, project_name):
        project_folder = self.get_project_folder(project_name)
        folder = os.path.join(project_folder, self.ALLEN_MOUSE_CONNECTIVITY_CACHE_FOLDER)
        self.check_created(folder)
        return folder

    # GENERIC METHODS Start Here #######################

    @staticmethod
    def copy_file(source, dest, dest_postfix, buffer_size):
        """
        Copy a file from source to dest. source and dest can either be strings or 
        any object with a read or write method, like StringIO for example.
        """
        should_close_source = False
        should_close_dest = False

        try:
            if not hasattr(source, 'read'):
                source = open(source, 'rb')
                should_close_source = True

            if not hasattr(dest, 'write'):
                if dest_postfix is not None:
                    dest = os.path.join(dest, dest_postfix)
                if not os.path.exists(os.path.dirname(dest)):
                    os.makedirs(os.path.dirname(dest))
                dest = open(dest, 'wb')
                should_close_dest = True

            shutil.copyfileobj(source, dest, length=buffer_size)

        finally:
            if should_close_source:
                source.close()
            if should_close_dest:
                dest.close()

    @staticmethod
    def remove_files(file_list, ignore_exception=False):
        """
        :param file_list: list of file paths to be removed.
        :param ignore_exception: When True and one of the specified files could not be removed, an exception is raised.  
        """
        for file_ in file_list:
            try:
                if os.path.exists(file_):
                    if os.path.isfile(file_):
                        os.remove(file_)
                    if os.path.isdir(file_):
                        shutil.rmtree(file_)
            except Exception:
                logger = get_logger(__name__)
                logger.exception("Could not remove " + str(file_))
                if not ignore_exception:
                    raise

    @staticmethod
    def remove_folder(folder_path, ignore_errors):
        """
        Given a folder path, try to remove that folder from disk.
        :param folder_path: Folder to be removed
        :param ignore_errors: When False throw FileStructureException if folder_path is invalid.
        """
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path, ignore_errors)
            return
        if not ignore_errors:
            raise FileStructureException("Given path does not exists, or is not a folder " + str(folder_path))

    @staticmethod
    def compute_size_on_disk(file_path):
        """
        Given a file's path, return size occupied on disk by that file.
        Size should be a number, representing size in KB.
        """
        if os.path.isfile(file_path):
            return int(os.path.getsize(file_path) / 1024)
        return 0

    @staticmethod
    def compute_recursive_h5_disk_usage(start_path):
        """
        Computes the disk usage of all h5 files under the given directory.
        :param start_path:
        :return: A tuple of size in kiB
        """
        total_size = 0
        n_files = 0
        for dir_path, _, file_names in os.walk(start_path):
            for f in file_names:
                if f.endswith('.h5'):
                    fp = os.path.join(dir_path, f)
                    total_size += os.path.getsize(fp)
                    n_files += 1
        return int(round(total_size / 1024.))


class TvbZip(ZipFile):
    def __init__(self, dest_path, mode):
        try:
            ZipFile.__init__(self, dest_path, mode, ZIP_DEFLATED, True)
        except Exception as excep:
            raise FileStructureException("Could not open the given ZIP file..." + str(excep))
        self.logger = get_logger(self.__class__.__module__)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()

    def write_zip_folder(self, folder, exclude, need_parent_folder=False):
        """
        write folder contents in archive
        :param folder: root folder in archive. Defaults to "" the archive root
        :param exclude: a list of file or folder names that will be recursively excluded
        :param need_parent_folder: if it is True, the parent_folder will be added as well to the zip file name
        """
        parent_folder = folder
        if need_parent_folder:
            parent_folder, _ = os.path.split(folder)

        for root, dirs, files in os.walk(folder):
            for ex in exclude:
                ex = str(ex)
                if ex in dirs:
                    dirs.remove(ex)
                if ex in files:
                    files.remove(ex)

            for file_n in files:
                abs_file_n = os.path.join(root, file_n)
                zip_file_n = abs_file_n[len(parent_folder) + len(os.sep):]
                self.write(abs_file_n, zip_file_n)

    def write_zip_folders(self, folders, exclude):
        """
        This method creates a ZIP file with all folders provided as parameters
        :param folders: array with the FULL names/path of the folders to add into ZIP
        :param exclude: a list of file or folder names that will be recursively excluded
        """
        for folder in set(folders):
            self.write_zip_folder(folder, exclude, True)

    def unpack_zip(self, folder_path):
        """ Simple method to unpack ZIP archive in a given folder. """

        def to_be_excluded(name):
            excluded_paths = ["__MACOSX/", ".DS_Store"]
            for excluded in excluded_paths:
                if name.startswith(excluded) or name.find('/' + excluded) >= 0:
                    return True
            return False

        try:
            result = []
            for filename in self.namelist():
                if not to_be_excluded(filename):
                    result.append(self.extract(filename, folder_path))
            return result
        except BadZipfile as excep:
            self.logger.exception("Could not process zip file")
            raise FileStructureException("Invalid ZIP file..." + str(excep))
        except Exception as excep:
            self.logger.exception("Could not process zip file")
            raise FileStructureException("Could not unpack the given ZIP file..." + str(excep))
