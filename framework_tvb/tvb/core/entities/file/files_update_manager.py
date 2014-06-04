# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
Manager for the file storage versioning updates.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os
import tvb.core.entities.file.file_update_scripts as file_update_scripts
from tvb.basic.config.settings import TVBSettings as cfg
from tvb.basic.traits.types_mapped import MappedType
from tvb.core.code_versions.base_classes import UpdateManager
from tvb.core.entities.file.hdf5_storage_manager import HDF5StorageManager
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.exceptions import MissingDataFileException, FileVersioningException, FileStructureException
from tvb.core.entities.storage import dao


FILE_STORAGE_VALID = 'valid'
FILE_STORAGE_INVALID = 'invalid'


class FilesUpdateManager(UpdateManager):
    """
    Manager for updating H5 files version, when code gets changed.
    """

    UPDATE_SCRIPTS_SUFFIX = "_update_files"
    PROJECTS_PAGE_SIZE = 20
    DATA_TYPES_PAGE_SIZE = 20
    
    
    def __init__(self):
        super(FilesUpdateManager, self).__init__(file_update_scripts, cfg.DATA_CHECKED_TO_VERSION, cfg.DATA_VERSION)


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
        except MissingDataFileException, ex:
            self.log.exception(ex)
            return False
        except FileStructureException, ex:
            self.log.exception(ex)
            return False

        if file_version == cfg.DATA_VERSION:
            return True
        return False


    def upgrade_file(self, input_file_name):
        """
        Upgrades the given file to the latest data version. The file will be upgraded
        sequencially up until the current version from tvb.basic.config.settings.
        
        :param input_file_name: the path to the file which needs to be upgraded
        """
        file_version = self.get_file_data_version(input_file_name)
        for script_name in self.get_update_scripts(file_version):
            self.run_update_script(script_name, input_file=input_file_name)
        self._update_datatype_disk_size(input_file_name)


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
        for datatype in datatypes:
            specific_datatype = dao.get_datatype_by_gid(datatype.gid)
            if isinstance(specific_datatype, MappedType):
                try:
                    self.upgrade_file(specific_datatype.get_storage_file_path())
                    nr_of_dts_upgraded_fine += 1
                except (MissingDataFileException, FileVersioningException) as ex:
                    # The file is missing for some reason. Just mark the DataType as invalid.
                    datatype.invalid = True
                    dao.store_entity(datatype)
                    nr_of_dts_upgraded_fault += 1
                    self.log.exception(ex)
        return nr_of_dts_upgraded_fine, nr_of_dts_upgraded_fault
                        
                        
    def upgrade_all_files_from_storage(self):
        """
        Upgrades all the data types from TVB storage to the latest data version.
        
        :returns: a two entry tuple (status, message) where status is a boolean that is True in case
            the upgrade was successfully for all DataTypes and False otherwise, and message is a status
            update message.
        """
        if cfg.DATA_CHECKED_TO_VERSION < cfg.DATA_VERSION:
            datatype_total_count = dao.count_all_datatypes()
            # Keep track of how many DataTypes were properly updated and how many 
            # were marked as invalid due to missing files or invalid manager.
            nr_of_dts_upgraded_fine = 0
            nr_of_dts_upgraded_fault = 0
            
            # Read DataTypes in pages just to spare memory consumption
            datatypes_nr_of_pages = datatype_total_count // self.DATA_TYPES_PAGE_SIZE
            if datatype_total_count % self.DATA_TYPES_PAGE_SIZE:
                datatypes_nr_of_pages += 1
            current_datatype_page = 1
            
            while current_datatype_page <= datatypes_nr_of_pages:
                
                datatype_start_idx = self.DATA_TYPES_PAGE_SIZE * (current_datatype_page - 1)
                if datatype_total_count >= datatype_start_idx + self.DATA_TYPES_PAGE_SIZE:
                    datatype_end_idx = self.DATA_TYPES_PAGE_SIZE * current_datatype_page  
                else: 
                    datatype_end_idx = datatype_total_count - datatype_start_idx
    
                datatypes_for_page = dao.get_all_datatypes(datatype_start_idx, datatype_end_idx)
                upgraded_fine_count, upgraded_fault_count = self.__upgrade_datatype_list(datatypes_for_page)
                nr_of_dts_upgraded_fine += upgraded_fine_count
                nr_of_dts_upgraded_fault += upgraded_fault_count
                current_datatype_page += 1
                
            # Now update the configuration file since update was done
            config_file_update_dict = {cfg.KEY_LAST_CHECKED_FILE_VERSION: cfg.DATA_VERSION}
            if nr_of_dts_upgraded_fault == 0:
                # Everything went fine
                config_file_update_dict[cfg.KEY_FILE_STORAGE_UPDATE_STATUS] = FILE_STORAGE_VALID
                return_status = True
                return_message = ("File upgrade finished successfully for all %s entries. "
                                  "Thank you for your patience" % nr_of_dts_upgraded_fine)
                self.log.info(return_message)
            else:
                # Something went wrong
                config_file_update_dict[cfg.KEY_FILE_STORAGE_UPDATE_STATUS] = FILE_STORAGE_INVALID
                return_status = False
                return_message = ("Out of %s stored DataTypes, %s were upgraded successfully "
                                  "and %s had faults and were marked invalid" % (datatype_total_count, 
                                  nr_of_dts_upgraded_fine, nr_of_dts_upgraded_fault))
                self.log.warning(return_message)
            cfg.add_entries_to_config_file(config_file_update_dict)
            return return_status, return_message
     

    @staticmethod
    def _get_manager(file_path):
        """
        Returns a storage manager.
        """
        folder, file_name = os.path.split(file_path)
        return HDF5StorageManager(folder, file_name)


    def _update_datatype_disk_size(self, file_path):
        """
        Computes and updates the disk_size attribute of the DataType, for which was created the given file.
        """
        file_handler = FilesHelper()
        datatype_gid = self._get_manager(file_path).get_gid_attribute()
        datatype = dao.get_datatype_by_gid(datatype_gid)
        
        if datatype is not None:
            datatype.disk_size = file_handler.compute_size_on_disk(file_path)
            dao.store_entity(datatype)
            
            
