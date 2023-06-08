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
"""

from tvb.adapters.exporters.abcexporter import ABCExporter
from tvb.core.entities.model.model_datatype import DataType, DataTypeGroup
from tvb.core.neocom import h5
from tvb.storage.storage_interface import StorageInterface


class TVBExporter(ABCExporter):
    """ 
    This exporter simply provides for download data in TVB format
    """

    def __init__(self):
        self.storage_interface = StorageInterface()

    def get_supported_types(self):
        return [DataType]

    def get_label(self):
        return "TVB Format"

    def export(self, data, project, public_key_path, password):
        """
        Exports data type:
        1. If data is a normal data type, simply exports storage file (HDF format)
        2. If data is a DataTypeGroup creates a zip with all files for all data types
        """
        download_file_name = self._get_export_file_name(data)

        if DataTypeGroup.is_data_a_group(data):
            _, op_file_dict = self.prepare_datatypes_for_export(data)

            # Create ZIP archive
            zip_file = self.storage_interface.export_datatypes_structure(op_file_dict, data, download_file_name,
                                                                         public_key_path, password)
            return download_file_name, zip_file, True
        else:
            data_path = h5.path_for_stored_index(data)
            data_file = self.storage_interface.export_datatypes([data_path], data, download_file_name,
                                                                public_key_path, password)

            return None, data_file, True

    def get_export_file_extension(self, data):
        if DataTypeGroup.is_data_a_group(data):
            return StorageInterface.TVB_ZIP_FILE_EXTENSION
        else:
            return StorageInterface.TVB_STORAGE_FILE_EXTENSION
