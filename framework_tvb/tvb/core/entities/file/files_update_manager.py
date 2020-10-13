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
Manager for the file storage version updates.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

import os
import tvb.core.entities.file.file_update_scripts as file_update_scripts
from datetime import datetime
from tvb.basic.config import stored
from tvb.basic.config.stored import KEY_ADMIN_DISPLAY_NAME, KEY_ENABLE_KC_LOGIN, KEY_KC_WEB_CONFIGURATION, \
    KEY_KC_CONFIGURATION
from tvb.basic.profile import TvbProfile
from tvb.core.code_versions.base_classes import UpdateManager
from tvb.core.entities.file.hdf5_storage_manager import HDF5StorageManager
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.exceptions import MissingDataFileException, FileStructureException, FileMigrationException
from tvb.core.entities.model.db_update_scripts.helper import delete_old_burst_table_after_migration
from tvb.core.entities.storage import dao


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
        self.files_helper = FilesHelper()

    def get_file_data_version(self, file_path):
        """
        Return the data version for the given file.
        
        :param file_path: the path on disk to the file for which you need the TVB data version
        :returns: a number representing the data version for which the input file was written
        """
        manager = self._get_manager(file_path)
        return manager.get_file_data_version()

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
        :return 0 when update was not necessary, 1 when update was successful and -1 when it resulted in an error.
        False, the the file is already up to date.

        """
        if self.is_file_up_to_date(input_file_name):
            # Avoid running the DB update of size, when H5 is not being changed, to speed-up
            return 0

        file_version = self.get_file_data_version(input_file_name)
        self.log.info("Updating from version %s , file: %s " % (file_version, input_file_name))
        for script_name in self.get_update_scripts(file_version):
            temp_file_path = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER,
                                          os.path.basename(input_file_name) + '.tmp')
            self.files_helper.copy_file(input_file_name, temp_file_path)
            try:
                self.run_update_script(script_name, input_file=input_file_name, burst_match_dict=burst_match_dict)
            except FileMigrationException as excep:
                self.files_helper.copy_file(temp_file_path, input_file_name)
                os.remove(temp_file_path)
                self.log.info(excep)
                return -1

        if datatype:
            # Compute and update the disk_size attribute of the DataType in DB:
            datatype.disk_size = self.files_helper.compute_size_on_disk(input_file_name)
            dao.store_entity(datatype)

        return 1

    def __upgrade_datatype_list(self, datatypes):
        """
        Upgrade a list of DataTypes to the current version.

        :param datatypes: The list of DataTypes that should be upgraded.

        :returns: (nr_of_dts_upgraded_fine, nr_of_dts_upgraded_fault) a two-tuple of integers representing
            the number of DataTypes for which the upgrade worked fine, and the number of DataTypes for which
            some kind of fault occurred
        """
        nr_of_dts_upgraded_fine = 0
        nr_of_dts_upgraded_fault = 0
        no_of_dts_ignored = 0

        for datatype in datatypes:
            try:
                from tvb.basic.traits.types_mapped import MappedType

                specific_datatype = dao.get_datatype_by_gid(datatype.gid, load_lazy=False)

                if specific_datatype is None:
                    datatype.invalid = True
                    dao.store_entity(datatype)
                    nr_of_dts_upgraded_fault += 1
                elif isinstance(specific_datatype, MappedType):
                    update_was_needed = self.upgrade_file(specific_datatype.get_storage_file_path(), specific_datatype)
                    if update_was_needed:
                        nr_of_dts_upgraded_fine += 1
                    else:
                        no_of_dts_ignored += 1
                else:
                    # Ignore DataTypeGroups
                    self.log.debug("We will ignore, due to type: " + str(specific_datatype))
                    no_of_dts_ignored += 1

            except Exception as ex:
                # The file/class is missing for some reason. Just mark the DataType as invalid.
                datatype.invalid = True
                dao.store_entity(datatype)
                nr_of_dts_upgraded_fault += 1
                self.log.exception(ex)

        return nr_of_dts_upgraded_fine, nr_of_dts_upgraded_fault, no_of_dts_ignored

    def __upgrade_h5_list(self, h5_files):
        """
        Upgrade a list of DataTypes to the current version (only from version 4 to 5).

        :returns: (nr_of_dts_upgraded_fine, nr_of_dts_ignored) a two-tuple of integers representing
            the number of DataTypes for which the upgrade worked fine, and the number of DataTypes for which
            upgrade was not needed.
        """
        nr_of_dts_upgraded_fine = 0
        nr_of_dts_failed = 0

        burst_match_dict = {}
        for path in h5_files:
            update_result = self.upgrade_file(path, burst_match_dict=burst_match_dict)

            if update_result == 1:
                nr_of_dts_upgraded_fine += 1
            else:
                nr_of_dts_failed += 1

        return nr_of_dts_upgraded_fine, nr_of_dts_failed

    def run_all_updates(self):
        """
        Upgrades all the data types from TVB storage to the latest data version.
        
        :returns: a two entry tuple (status, message) where status is a boolean that is True in case
            the upgrade was successfully for all DataTypes and False otherwise, and message is a status
            update message.
        """
        if TvbProfile.current.version.DATA_CHECKED_TO_VERSION < TvbProfile.current.version.DATA_VERSION:
            total_count = dao.count_all_datatypes()

            self.log.info("Starting to run H5 file updates from version %d to %d, for %d datatypes" % (
                TvbProfile.current.version.DATA_CHECKED_TO_VERSION,
                TvbProfile.current.version.DATA_VERSION, total_count))

            # Keep track of how many DataTypes were properly updated and how many 
            # were marked as invalid due to missing files or invalid manager.
            no_ok = 0
            no_error = 0
            no_ignored = 0
            start_time = datetime.now()

            if TvbProfile.current.version.DATA_CHECKED_TO_VERSION < 4:
                # Read DataTypes in pages to limit the memory consumption
                for current_idx in range(0, total_count, self.DATA_TYPES_PAGE_SIZE):
                    datatypes_for_page = dao.get_all_datatypes(current_idx, self.DATA_TYPES_PAGE_SIZE)
                    count_ok, count_error, count_ignored = self.__upgrade_datatype_list(datatypes_for_page)
                    no_ok += count_ok
                    no_error += count_error
                    no_ignored += count_ignored

                    self.log.info(
                        "Updated H5 files so far: %d [fine:%d, error:%d, ignored:%d of total:%d, in: %s min]" % (
                            current_idx + len(datatypes_for_page), no_ok, no_error, no_ignored, total_count,
                            int((datetime.now() - start_time).seconds / 60)))
            else:
                # Migrating the configuration file
                new_stored_settings = {KEY_ADMIN_DISPLAY_NAME: 'Administrator', KEY_ENABLE_KC_LOGIN: False,
                                       KEY_KC_WEB_CONFIGURATION: '', KEY_KC_CONFIGURATION: 'add_keycloak_path_here'}
                manager = TvbProfile.current.manager
                manager.add_entries_to_config_file(new_stored_settings)
                manager.delete_entries_from_config_file(['URL_WEB'])

                from tvb.core.neocom import h5
                file_paths = h5.get_all_h5_paths()
                total_count = len(file_paths)
                count_ok, count_error = self.__upgrade_h5_list(file_paths)

                self.log.info("Updated H5 files in total: %d [fine:%d, failed:%d in: %s min]" % (
                    total_count, count_ok, count_error, int((datetime.now() - start_time).seconds / 60)))
                delete_old_burst_table_after_migration()

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
                # Something went wrong
                config_file_update_dict[stored.KEY_FILE_STORAGE_UPDATE_STATUS] = FILE_STORAGE_INVALID
                FilesUpdateManager.STATUS = False
                FilesUpdateManager.MESSAGE = ("Out of %s stored DataTypes, %s were upgraded successfully, but %s had "
                                              "faults and were marked invalid" % (total_count, no_ok, no_error))
                self.log.warning(FilesUpdateManager.MESSAGE)

            TvbProfile.current.version.DATA_CHECKED_TO_VERSION = TvbProfile.current.version.DATA_VERSION
            TvbProfile.current.manager.add_entries_to_config_file(config_file_update_dict)

    @staticmethod
    def _get_manager(file_path):
        """
        Returns a storage manager.
        """
        folder, file_name = os.path.split(file_path)
        return HDF5StorageManager(folder, file_name)
