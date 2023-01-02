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
Class responsible for all TVB exports (datatype or project).

.. moduleauthor:: Calin Pavel <calin.pavel@codemat.ro
"""
import shutil
from cgi import FieldStorage
from cherrypy._cpreqbody import Part

from tvb.adapters.exporters.abcexporter import ABCExporter
from tvb.adapters.exporters.exceptions import ExportException, InvalidExportDataException
from tvb.adapters.exporters.tvb_export import TVBExporter
from tvb.adapters.exporters.tvb_linked_export import TVBLinkedExporter
from tvb.basic.logger.builder import get_logger
from tvb.config import TVB_IMPORTER_MODULE, TVB_IMPORTER_CLASS
from tvb.core.entities.model import model_operation
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.services.project_service import ProjectService
from tvb.storage.storage_interface import StorageInterface


class ExportManager(object):
    """
    This class provides basic methods for exporting data types of projects in different formats.
    """
    all_exporters = {}  # Dictionary containing all available exporters
    logger = get_logger(__name__)

    def __init__(self):
        # Here we register all available data type exporters
        # If new exporters supported, they should be added here
        self._register_exporter(TVBExporter())
        self._register_exporter(TVBLinkedExporter())
        self.storage_interface = StorageInterface()

    def _register_exporter(self, exporter):
        """
        This method register into an internal format available exporters.
        :param exporter: Instance of a data type exporter (extends ABCExporter)
        """
        if exporter is not None:
            self.all_exporters[exporter.__class__.__name__] = exporter

    def get_exporters_for_data(self, data):
        """
        Get available exporters for current data type.
        :returns: a dictionary with the {exporter_id : label}
        """
        if data is None:
            raise InvalidExportDataException("Could not detect exporters for null data")

        self.logger.debug("Trying to determine exporters valid for %s" % data.type)
        results = {}

        # No exporter for None data
        if data is None:
            return results

        for exporterId in self.all_exporters.keys():
            exporter = self.all_exporters[exporterId]
            if exporter.accepts(data):
                results[exporterId] = exporter.get_label()

        return results

    def export_data(self, data, exporter_id, project, user_public_key=None):
        """
        Export provided data using given exporter
        :param data: data type to be exported
        :param exporter_id: identifier of the exporter to be used
        :param project: project that contains data to be exported
        :param user_public_key: public key file used for encrypting data before exporting

        :returns: a tuple with the following elements
            1. name of the file to be shown to user
            2. full path of the export file (available for download)
            3. boolean which specify if file can be deleted after download
        """
        if data is None:
            raise InvalidExportDataException("Could not export null data. Please select data to be exported")

        if exporter_id is None:
            raise ExportException("Please select the exporter to be used for this operation")

        if exporter_id not in self.all_exporters:
            raise ExportException("Provided exporter identifier is not a valid one")

        exporter = self.all_exporters[exporter_id]

        if user_public_key is not None:
            public_key_path, encryption_password = self.storage_interface.prepare_encryption(project.name)
            if isinstance(user_public_key, (FieldStorage, Part)):
                with open(public_key_path, 'wb') as file_obj:
                    self.storage_interface.copy_file(user_public_key.file, file_obj)
            else:
                shutil.copy2(user_public_key, public_key_path)

        else:
            public_key_path, encryption_password = None, None

        if project is None:
            raise ExportException("Please provide the project where data files are stored")

        # Now we start the real export
        if not exporter.accepts(data):
            raise InvalidExportDataException("Current data can not be exported by specified exporter")

        # Now compute and create folder where to store exported data
        # This will imply to generate a folder which is unique for each export
        export_data = None
        try:
            self.logger.debug("Start export of data: %s" % data.type)
            export_data = exporter.export(data, project, public_key_path, encryption_password)
        except Exception:
            pass

        return export_data

    @staticmethod
    def _get_paths_of_linked_datatypes(project):
        linked_paths = ProjectService().get_linked_datatypes_storage_path(project)

        if not linked_paths:
            # do not export an empty operation
            return None, None

        # Make an import operation which will contain links to other projects
        algo = dao.get_algorithm_by_module(TVB_IMPORTER_MODULE, TVB_IMPORTER_CLASS)
        op = model_operation.Operation(None, None, project.id, algo.id)
        op.project = project
        op.algorithm = algo
        op.id = 'links-to-external-projects'
        op.start_now()
        op.mark_complete(model_operation.STATUS_FINISHED)

        return linked_paths, op

    def export_project(self, project):
        """
        Given a project root and the TVB storage_path, create a ZIP
        ready for export.
        :param project: project object which identifies project to be exported
        """
        if project is None:
            raise ExportException("Please provide project to be exported")

        folders_to_exclude = self._get_op_with_errors(project.id)
        linked_paths, op = self._get_paths_of_linked_datatypes(project)

        result_path = self.storage_interface.export_project(project, folders_to_exclude, linked_paths, op)

        return result_path

    @staticmethod
    def _get_op_with_errors(project_id):
        """
        Get the operation folders with error base name as list.
        """
        operations = dao.get_operations_with_error_in_project(project_id)
        op_with_errors = []
        for op in operations:
            op_with_errors.append(op.id)
        return op_with_errors

    def export_simulator_configuration(self, burst_id):
        burst = dao.get_burst_by_id(burst_id)
        if burst is None:
            raise InvalidExportDataException("Could not find burst with ID " + str(burst_id))

        op_folder = self.storage_interface.get_project_folder(burst.project.name, str(burst.fk_simulation))

        all_view_model_paths, all_datatype_paths = h5.gather_references_of_view_model(burst.simulator_gid, op_folder)

        burst_path = h5.determine_filepath(burst.gid, op_folder)
        all_view_model_paths.append(burst_path)

        zip_filename = ABCExporter.get_export_file_name(burst, self.storage_interface.TVB_ZIP_FILE_EXTENSION)
        result_path = self.storage_interface.export_simulator_configuration(burst, all_view_model_paths,
                                                                            all_datatype_paths, zip_filename)
        return result_path
