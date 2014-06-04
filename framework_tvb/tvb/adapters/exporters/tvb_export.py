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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
from tvb.core.entities import model 
from tvb.adapters.exporters.abcexporter import ABCExporter
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.adapters.exporters.exceptions import ExportException 


class TVBExporter(ABCExporter):
    """ 
    This exporter simply provides for download data in TVB format
    """
    OPERATION_FOLDER_PREFIX = "Operation_"
    
    def get_supported_types(self):
        return [model.DataType]
    
    def get_label(self):
        return "TVB Format"
    
    def export(self, data, export_folder, project):
        """
        Exports data type:
        1. If data is a normal data type, simply exports storage file (HDF format)
        2. If data is a DataTypeGroup creates a zip with all files for all data types
        """
        download_file_name = self.get_export_file_name(data)
        files_helper = FilesHelper()
         
        if self.is_data_a_group(data):
            all_datatypes = self._get_all_data_types_arr(data)
            
            if all_datatypes is None or len(all_datatypes) == 0:
                raise ExportException("Could not export a data type group with no data")    
            
            zip_file = os.path.join(export_folder, download_file_name)
            
            # Now process each data type from group and add it to ZIP file
            operation_folders = []
            for data_type in all_datatypes:
                operation_folder = files_helper.get_operation_folder(project.name, data_type.fk_from_operation)
                operation_folders.append(operation_folder)
                
            # Create ZIP archive    
            files_helper.zip_folders(zip_file, operation_folders, self.OPERATION_FOLDER_PREFIX)
                        
            return download_file_name, zip_file, True

        else:
            project_folder = files_helper.get_project_folder(project)
            data_file = os.path.join(project_folder, data.get_storage_file_path())

            return download_file_name, data_file, False


    def get_export_file_extension(self, data):
        if self.is_data_a_group(data):
            return "zip"
        else:
            return "h5"
