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
Manager for the file storage version updates.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

from datetime import datetime
import os
from sqlalchemy import text

import tvb.core.entities.file.file_update_scripts as file_update_scripts
from tvb.core.entities.storage import SA_SESSIONMAKER
from tvb.basic.config import stored
from tvb.basic.profile import TvbProfile
from tvb.core.code_versions.base_classes import UpdateManager
from tvb.core.entities.storage import dao
from tvb.core.neotraits.h5 import H5File
from tvb.core.utils import string2date
from tvb.storage.h5.file.exceptions import FileStructureException, MissingDataFileException, FileMigrationException
from tvb.storage.storage_interface import StorageInterface

FILE_STORAGE_VALID = 'valid'
FILE_STORAGE_INVALID = 'invalid'


class FilesUpdateManager(UpdateManager):
    """
    Manager for updating H5 files version, when code gets changed.
    """

    UPDATE_SCRIPTS_SUFFIX = "_update_files"
    PROJECTS_PAGE_SIZE = 20
    DATA_TYPES_PAGE_SIZE = 500
    STATUS = True
    MESSAGE = "Done"

    def __init__(self):
        super(FilesUpdateManager, self).__init__(file_update_scripts,
                                                 TvbProfile.current.version.DATA_CHECKED_TO_VERSION,
                                                 TvbProfile.current.version.DATA_VERSION)
        self.storage_interface = StorageInterface()

    def get_file_data_version(self, file_path):
        """
        Return the data version for the given file.

        :param file_path: the path on disk to the file for which you need the TVB data version
        :returns: a number representing the data version for which the input file was written
        """
        data_version = TvbProfile.current.version.DATA_VERSION_ATTRIBUTE
        return self.storage_interface.get_storage_manager(file_path).get_file_data_version(data_version)

    def is_file_up_to_date(self, file_path):
        """
        Returns True only if the data version of the file is equal with the
        data version specified into the TVB configuration file.
        """
        try:
            file_version = self.get_file_data_version(file_path)
        except MissingDataFileException as ex:
            self.log.exception(ex)
            return False
        except FileStructureException as ex:
            self.log.exception(ex)
            return False

        if file_version == TvbProfile.current.version.DATA_VERSION:
            return True
        return False

    def upgrade_file(self, input_file_name, datatype=None, burst_match_dict=None):
        """
        Upgrades the given file to the latest data version. The file will be upgraded
        sequentially, up until the current version from tvb.basic.config.settings.VersionSettings.DB_STRUCTURE_VERSION

        :param input_file_name the path to the file which needs to be upgraded
        :return True when update was successful and False when it resulted in an error.
        """
        if self.is_file_up_to_date(input_file_name):
            # Avoid running the DB update of size, when H5 is not being changed, to speed-up
            return True

        file_version = self.get_file_data_version(input_file_name)
        self.log.info("Updating from version %s , file: %s " % (file_version, input_file_name))
        for script_name in self.get_update_scripts(file_version):
            temp_file_path = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER,
                                          os.path.basename(input_file_name) + '.tmp')
            self.storage_interface.copy_file(input_file_name, temp_file_path)
            try:
                self.run_update_script(script_name, input_file=input_file_name, burst_match_dict=burst_match_dict)
            except FileMigrationException as excep:
                self.storage_interface.copy_file(temp_file_path, input_file_name)
                os.remove(temp_file_path)
                self.log.error(excep)
                return False

        if datatype:
            # Compute and update the disk_size attribute of the DataType in DB:
            datatype.disk_size = self.storage_interface.compute_size_on_disk(input_file_name)
            dao.store_entity(datatype)

        return True

    def __upgrade_h5_list(self, h5_files):
        """
        Upgrade a list of DataTypes to the current version.

        :returns: (nr_of_dts_upgraded_fine, nr_of_dts_ignored) a two-tuple of integers representing
            the number of DataTypes for which the upgrade worked fine, and the number of DataTypes for which
            the upgrade has failed.
        """
        nr_of_dts_upgraded_fine = 0
        nr_of_dts_failed = 0

        burst_match_dict = {}
        for path in h5_files:
            update_result = self.upgrade_file(path, burst_match_dict=burst_match_dict)

            if update_result:
                nr_of_dts_upgraded_fine += 1
            else:
                nr_of_dts_failed += 1

        return nr_of_dts_upgraded_fine, nr_of_dts_failed

    # TO DO: We should migrate the older scripts to Python 3 if we want to support migration for versions < 4
    def run_all_updates(self):
        """
        Upgrades all the data types from TVB storage to the latest data version.

        :returns: a two entry tuple (status, message) where status is a boolean that is True in case
            the upgrade was successfully for all DataTypes and False otherwise, and message is a status
            update message.
        """
        if TvbProfile.current.version.DATA_CHECKED_TO_VERSION < TvbProfile.current.version.DATA_VERSION:
            start_time = datetime.now()

            file_paths = self.get_all_h5_paths()
            total_count = len(file_paths)
            no_ok, no_error = self.__upgrade_h5_list(file_paths)

            self.log.info("Updated H5 files in total: %d [fine:%d, failed:%d in: %s min]" % (
                total_count, no_ok, no_error, int((datetime.now() - start_time).seconds / 60)))
            self._delete_old_burst_table_after_migration()

            # Now update the configuration file since update was done
            config_file_update_dict = {stored.KEY_LAST_CHECKED_FILE_VERSION: TvbProfile.current.version.DATA_VERSION}

            if no_error == 0:
                # Everything went fine
                config_file_update_dict[stored.KEY_FILE_STORAGE_UPDATE_STATUS] = FILE_STORAGE_VALID
                FilesUpdateManager.STATUS = True
                FilesUpdateManager.MESSAGE = ("File upgrade finished successfully for all %s entries. "
                                              "Thank you for your patience!" % total_count)
                self.log.info(FilesUpdateManager.MESSAGE)
            else:
                # Keep track of how many DataTypes were properly updated and how many
                # were marked as invalid due to missing files or invalid manager.
                config_file_update_dict[stored.KEY_FILE_STORAGE_UPDATE_STATUS] = FILE_STORAGE_INVALID
                FilesUpdateManager.STATUS = False
                FilesUpdateManager.MESSAGE = ("Out of %s stored DataTypes, %s were upgraded successfully, but %s had "
                                              "faults and were marked invalid" % (total_count, no_ok, no_error))
                self.log.warning(FilesUpdateManager.MESSAGE)

            TvbProfile.current.version.DATA_CHECKED_TO_VERSION = TvbProfile.current.version.DATA_VERSION
            TvbProfile.current.manager.add_entries_to_config_file(config_file_update_dict)

    @staticmethod
    def get_all_h5_paths():
        """
        This method returns a list of all h5 files and it is used in the migration from version 4 to 5.
        The h5 files inside a certain project are retrieved in numerical order (1, 2, 3 etc.).
        """
        h5_files = []
        projects_folder = StorageInterface().get_projects_folder()

        for project_path in os.listdir(projects_folder):
            # Getting operation folders inside the current project
            project_full_path = os.path.join(projects_folder, project_path)
            try:
                project_operations = os.listdir(project_full_path)
            except NotADirectoryError:
                continue
            project_operations_base_names = [os.path.basename(op) for op in project_operations]

            for op_folder in project_operations_base_names:
                try:
                    int(op_folder)
                    op_folder_path = os.path.join(project_full_path, op_folder)
                    for file in os.listdir(op_folder_path):
                        if StorageInterface().ends_with_tvb_storage_file_extension(file):
                            h5_file = os.path.join(op_folder_path, file)
                            try:
                                if FilesUpdateManager._is_empty_file(h5_file):
                                    continue
                                h5_files.append(h5_file)
                            except FileStructureException:
                                continue
                except ValueError:
                    pass

        # Sort all h5 files based on their creation date stored in the files themselves
        sorted_h5_files = sorted(h5_files, key=lambda h5_path: FilesUpdateManager._get_create_date_for_sorting(
            h5_path) or datetime.now())
        return sorted_h5_files

    @staticmethod
    def _is_empty_file(h5_file):
        return H5File.get_metadata_param(h5_file, 'Create_date') is None

    @staticmethod
    def _get_create_date_for_sorting(h5_file):
        create_date_str = str(H5File.get_metadata_param(h5_file, 'Create_date'), 'utf-8')
        create_date = string2date(create_date_str, date_format='datetime:%Y-%m-%d %H:%M:%S.%f')
        return create_date

    def _delete_old_burst_table_after_migration(self):
        session = SA_SESSIONMAKER()
        try:
            session.execute(text("""DROP TABLE "BURST_CONFIGURATION"; """))
            session.commit()
        except Exception as excep:
            session.close()
            session = SA_SESSIONMAKER()
            self.log.exception(excep)
            try:
                session.execute(text("""DROP TABLE if exists "BURST_CONFIGURATION" cascade; """))
                session.commit()
            except Exception as excep:
                self.log.exception(excep)
        finally:
            session.close()
