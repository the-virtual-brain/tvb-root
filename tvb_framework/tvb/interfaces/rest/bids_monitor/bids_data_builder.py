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

import os
import json
import uuid
from zipfile import ZipFile

from tvb.basic.logger.builder import get_logger

logger = get_logger(__name__)

SUBJECT_PREFIX = 'sub'
POSSIBLE_ATTR_WITH_PATHS = ["CoordsRows", "CoordsColumns", "ModelEq", "ModelParam", "Network"]


class BIDSDataBuilder:

    def __init__(self, bids_data_to_import=None, bids_root_dir=None, init_json_files=None):
        self.bids_data_to_import = bids_data_to_import
        self.bids_root_dir = os.path.normpath(bids_root_dir)
        self.init_json_files = init_json_files

    def create_archive(self, files_list):
        logger.info("Creating ZIP archive of {} files".format(len(files_list)))
        base_dir_name = os.path.dirname(os.path.abspath(self.bids_root_dir))
        with ZipFile(self.temp_bids_zip_dir, 'w') as myzip:
            for file_name in files_list:
                myzip.write(
                    file_name, arcname=file_name.split(base_dir_name)[1])

        return self.temp_bids_zip_dir

    def get_abs_path(self, bids_root_dir, sub, path):
        if os.path.exists(os.path.abspath(path)):
            return os.path.abspath(path)
        if '../..' in path:
            path1 = path.replace('../..', bids_root_dir)
            if os.path.exists(path1):
                return os.path.abspath(path1)
        elif '..' in path:
            path1 = path.replace('..', bids_root_dir + '/' + sub)
            path2 = path.replace('..', bids_root_dir)
            if os.path.exists(path1):
                return os.path.abspath(path1)
            if os.path.exists(path2):
                return os.path.abspath(path2)
        return os.path.abspath(path)

    def init_paths(self):
        self.bids_file_base_dir = os.path.abspath(
            os.path.join(self.bids_root_dir, os.pardir))
        self.bids_file_name = os.path.split(
            os.path.normpath(self.bids_root_dir))[1]
        self.temp_bids_dir_name = self.bids_file_name + \
            '-' + str(uuid.uuid4()).split("-")[4]
        self.temp_bids_zip_dir = os.path.join(
            self.bids_file_base_dir, self.temp_bids_dir_name) + '.zip'

    def get_connected_dependencies_paths(self, sub, paths_queue, json_files_processed, import_dependencies_paths):
        # Now, reading all dependencies json files and adding dependency again if found
        while len(paths_queue) != 0:
            path = paths_queue.pop(0)
            if json_files_processed[path] is False:
                json_files_processed[path] = True
                try:
                    json_data = json.load(
                        open(self.get_abs_path(self.bids_root_dir, sub, path)))

                    for possible_path_key in POSSIBLE_ATTR_WITH_PATHS:
                        if json_data.get(possible_path_key) != None:
                            if isinstance(json_data[possible_path_key], list):
                                for path1 in json_data[possible_path_key]:
                                    computed_path = self.get_abs_path(
                                        self.bids_root_dir, sub, path1)
                                    import_dependencies_paths.add(
                                        computed_path)
                                    paths_queue.append(computed_path)
                                    if computed_path not in json_files_processed:
                                        json_files_processed[computed_path] = False
                            else:
                                computed_path = self.get_abs_path(
                                    self.bids_root_dir, sub, json_data[possible_path_key])
                                import_dependencies_paths.add(computed_path)
                                paths_queue.append(computed_path)
                                if computed_path not in json_files_processed:
                                    json_files_processed[computed_path] = False

                except Exception as e:
                    logger.error("Exception occurred in reading json files: {}".format(
                        e.__class__.__name__))

        return import_dependencies_paths

    def get_data_files(self, import_dependencies_paths, sub):
        data_files = set()
        for p in import_dependencies_paths:
            path_ar = p.split(os.sep)
            file_dir = os.path.sep.join(path_ar[0:len(path_ar)-1])
            file_name = path_ar[len(path_ar)-1]
            files_to_copy = [fn for fn in os.listdir(file_dir) if os.path.splitext(file_name)[
                0] == os.path.splitext(fn)[0]]
            for f in files_to_copy:
                data_files.add(self.get_abs_path(
                    self.bids_root_dir, sub, os.path.join(file_dir, f)))
        return data_files

    def create_dataset_subjects(self):
        """
        Creates BIDS dataset zip with all subject folders and the provided data type
        """
        if self.subject_build_data_check() is False:
            return

        self.init_paths()

        logger.info("Creating BIDS dataset for {}".format(
            self.bids_data_to_import))

        files = os.listdir(self.bids_root_dir)
        subject_folders = []
        final_paths_set = set()

        # First we find subject parent folders
        for file_name in files:
            if os.path.basename(file_name).startswith(SUBJECT_PREFIX) and os.path.isdir(os.path.join(self.bids_root_dir, file_name)):
                subject_folders.append(file_name)

        logger.info("Found {} subject folders in the".format(
            len(subject_folders)))

        # For each subject we read its content and sub-dirs
        for sub in subject_folders:
            sub_contents_path = os.path.abspath(os.path.join(
                self.bids_root_dir, sub, self.bids_data_to_import))
            sub_contents = os.listdir(sub_contents_path)
            if len(sub_contents) == 0:
                continue

            # Set for keeping only unique file pahts
            # Dict for track of json files which are processed
            # queue is used for reading all other dependencies present in nested json files
            import_dependencies_paths = set()
            json_files_processed = {}
            paths_queue = []

            # Addding path of all files present in the alg(to import) dir
            for file_name in sub_contents:
                file_path = self.get_abs_path(
                    self.bids_root_dir, sub, os.path.join(sub_contents_path, file_name))
                import_dependencies_paths.add(file_path)
                if os.path.splitext(file_path)[1] != '.json':
                    continue
                paths_queue.append(file_path)
                json_files_processed[file_path] = False

            final_paths_set = self.get_complete_paths(
                sub, paths_queue, json_files_processed, import_dependencies_paths, final_paths_set)

        # Creating zip archive all paths present in the set
        return self.create_archive(final_paths_set)

    def create_dataset_json_files(self):
        """
        Creates BIDS dataset zip using the provided json files
        """
        if self.files_build_data_check() is False:
            return

        self.init_paths()

        json_files_count = 0
        for f in self.init_json_files.values():
            json_files_count += len(f)

        logger.info("Creating BIDS dataset with {} subjects and {} json files".format(
            len(self.init_json_files), json_files_count))

        final_paths_set = set()

        for sub in self.init_json_files.keys():

            import_dependencies_paths = set()
            json_files_processed = {}
            paths_queue = []

            for file_path in self.init_json_files[sub]:
                import_dependencies_paths.add(file_path)
                paths_queue.append(file_path)
                json_files_processed[file_path] = False

            final_paths_set = self.get_complete_paths(
                sub, paths_queue, json_files_processed, import_dependencies_paths, final_paths_set)

        # Creating zip archive all paths present in the set
        return self.create_archive(final_paths_set)

    def get_complete_paths(self, sub, paths_queue, json_files_processed, import_dependencies_paths, final_paths_set):
        import_dependencies_paths = self.get_connected_dependencies_paths(
            sub, paths_queue, json_files_processed, import_dependencies_paths)
        data_files = self.get_data_files(import_dependencies_paths, sub)

        for p in data_files:
            import_dependencies_paths.add(p)
        for p in import_dependencies_paths:
            final_paths_set.add(p)

        return final_paths_set

    def subject_build_data_check(self):
        if self.bids_root_dir is None:
            logger.info("BIDS root directory is empty")
            return False
        if self.bids_data_to_import is None:
            logger.info(
                "BIDS data to import from the directory is not provided")
            return False
        return True

    def files_build_data_check(self):
        if self.init_json_files is None:
            logger.info("Provided inital files are None")
            return False
        return True

