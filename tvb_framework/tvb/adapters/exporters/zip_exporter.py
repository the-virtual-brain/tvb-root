# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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


from tvb.adapters.exporters.abcexporter import ABCExporter
from tvb.core.entities.model.model_datatype import ZipDatatype
from tvb.storage.storage_interface import StorageInterface


class ZipExporter(ABCExporter):
    """
    This exporter provides for download pipeline results in ZIP format
    """

    def __init__(self):
        self.storage_interface = StorageInterface()

    def get_supported_types(self):
        return [ZipDatatype]

    def get_label(self):
        return "ZIP Format"

    def export(self, data, project, public_key_path, password):
        """
        Downloads a ZIP file with pipeline results.
        """
        download_file_name = self._get_export_file_name(data)
        data_file = data.zip_path

        return download_file_name, data_file, False

    def get_export_file_extension(self, data):
        return self.storage_interface.TVB_ZIP_FILE_EXTENSION
