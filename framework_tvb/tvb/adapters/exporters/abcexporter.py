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
Root class for export functionality.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
from datetime import datetime
from abc import ABCMeta, abstractmethod
from tvb.core.entities.model import DataTypeGroup
from tvb.core.services.project_service import ProjectService
from tvb.core.adapters.abcadapter import ABCAdapter

#List of DataTypes to be excluded from export due to not having a valid export mechanism implemented yet.
EXCLUDED_DATATYPES = ['Cortex', 'CortexActivity', 'CapEEGActivity', 'Cap', 'ValueWrapper', 'SpatioTermporalMask']


class ABCExporter:
    """
    Base class for all data type exporters
    This should provide common functionality for all TVB exporters.
    """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def get_supported_types(self):
        """
        This method specify what types are accepted by this exporter.
        Method should be implemented by each subclass and return
        an array with the supported types.
            
        :returns: an array with the supported data types.
        """
        pass
    
    def get_label(self):
        """
        This method returns a string to be used on the UI controls to initiate export
        
        :returns: string to be used on UI for starting this export.
                  By default class name is returned
        """
        return self.__class__.__name__
    
    def accepts(self, data):
        """
        This method specify if the current exporter can export provided data.
        :param data: data to be checked
        :returns: true if this data can be exported by current exporter, false otherwise.
        """
        effective_data_type = self._get_effective_data_type(data)
        
        # If no data present for export, makes no sense to show exporters
        if effective_data_type is None:
            return False
        
        # Now we should check if any data type is accepted by current exporter
        # Check if the data type is one of the global exclusions 
        if hasattr(effective_data_type, "type") and effective_data_type.type in EXCLUDED_DATATYPES:
            return False
            
        for supported_type in self.get_supported_types():
            if isinstance(effective_data_type, supported_type):
                return True

        return False
    
    def _get_effective_data_type(self, data):
        """
        This method returns the data type for the provided data.
        - If current data is a simple data type is returned.
        - If it is an data type group, we return the first element. Only one element is
        necessary since all group elements are the same type.
        """
        # first check if current data is a DataTypeGroup
        if self.is_data_a_group(data):
            data_types = ProjectService.get_datatypes_from_datatype_group(data.id)
            
            if data_types is not None and len(data_types) > 0:
                # Since all objects in a group are the same type it's enough 
                return ABCAdapter.load_entity_by_gid(data_types[0].gid)
            else:
                return None    
        else:
            return data
    
    def _get_all_data_types_arr(self, data):
        """
        This method builds an array with all data types to be processed later.
        - If current data is a simple data type is added to an array.
        - If it is an data type group all its children are loaded and added to array.
        """
        # first check if current data is a DataTypeGroup
        if self.is_data_a_group(data):
            data_types = ProjectService.get_datatypes_from_datatype_group(data.id)
            
            result = []
            if data_types is not None and len(data_types) > 0:
                for data_type in data_types:
                    entity = ABCAdapter.load_entity_by_gid(data_type.gid)
                    result.append(entity)
                     
            return result    
            
        else:
            return [data]
    
    def is_data_a_group(self, data):
        """
        Checks if the provided data, ready for export is a DataTypeGroup or not
        """
        return isinstance(data, DataTypeGroup)        
    
    @abstractmethod
    def export(self, data, export_folder, project):
        """
        Actual export method, to be implemented in each sub-class.

        :param data: data type to be exported

        :param export_folder: folder where to write results of the export if needed.
                              This is necessary in case new files are generated.

        :param project: project that contains data to be exported
            
        :returns: a tuple with the following elements:

                        1. name of the file to be shown to user
                        2. full path of the export file (available for download)
                        3. boolean which specify if file can be deleted after download
        """
        pass
    
    def get_export_file_name(self, data):
        """
        This method computes the name used to save exported data on user computer
        """
        file_ext = self.get_export_file_extension(data)
        data_type_name = data.__class__.__name__
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M")
        
        return "%s_%s.%s" % (date_str, data_type_name, file_ext)
        
    @abstractmethod
    def get_export_file_extension(self, data):
        """
        This method computes the extension of the export file
        :param data: data type to be exported
        :returns: the extension of the file to be exported (e.g zip or h5)
        """
        pass