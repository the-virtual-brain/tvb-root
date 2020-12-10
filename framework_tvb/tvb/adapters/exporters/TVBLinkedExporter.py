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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
from tvb.adapters.exporters.abcexporter import ABCExporter
from tvb.core.entities.file.files_helper import FilesHelper, TvbZip
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.neotraits._h5core import H5File


class TVBLinkedExporter(ABCExporter):
    """
    """

    OPERATION_FOLDER_PREFIX = "Operation_"

    def get_supported_types(self):
        return [DataType]

    def get_label(self):
        return "TVB Linked Format"

    def export(self, data, data_export_folder, project):
        """
        Exports data type:
        1. If data is a normal data type, simply exports storage file (HDF format)
        2. If data is a DataTypeGroup creates a zip with all files for all data types
        """
        self.copy_dt_to_export_folder(data, data_export_folder)
        export_data_zip_path = self.get_export_data_zip_path(data, data_export_folder, self)
        return self.export_data_with_references(export_data_zip_path, data_export_folder)

    def get_export_data_zip_path(self, data, data_export_folder, exporter):
        zip_file_name = exporter.get_export_file_name(data)
        zip_file_name = zip_file_name.replace('.h5', '.zip')
        return os.path.join(os.path.dirname(data_export_folder), zip_file_name)

    def export_data_with_references(self, export_data_zip_path, data_export_folder):
        with TvbZip(export_data_zip_path, "w") as zip_file:
            for filename in os.listdir(data_export_folder):
                zip_file.write(os.path.join(data_export_folder, filename), filename)

        return None, export_data_zip_path, True

    def copy_dt_to_export_folder(self, data, data_export_folder):
        data_path = h5.path_for_stored_index(data)
        with H5File.from_file(data_path) as f:
            file_destination = os.path.join(data_export_folder, os.path.basename(data_path))
            if not os.path.exists(file_destination):
                FilesHelper().copy_file(data_path, file_destination)
            sub_dt_refs = f.gather_references()

            for reference in sub_dt_refs:
                if reference[1]:
                    dt = dao.get_datatype_by_gid(reference[1].hex)
                    self.copy_dt_to_export_folder(dt, data_export_folder)

    def get_export_file_extension(self, data):
        if self.is_data_a_group(data):
            return "zip"
        else:
            return "h5"

    def skip_group_datatypes(self):
        return True