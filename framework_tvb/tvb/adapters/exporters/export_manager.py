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
Class responsible for all TVB exports (datatype or project).

.. moduleauthor:: Calin Pavel <calin.pavel@codemat.ro
"""

import os
from datetime import datetime

from tvb.adapters.exporters.exceptions import ExportException, InvalidExportDataException
from tvb.adapters.exporters.tvb_export import TVBExporter
from tvb.adapters.exporters.tvb_linked_export import TVBLinkedExporter
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.config import TVB_IMPORTER_MODULE, TVB_IMPORTER_CLASS
from tvb.file.files_helper import FilesHelper, TvbZip
from tvb.core.entities.model import model_operation
from tvb.core.entities.storage import dao
from tvb.neocom import h5
from tvb.neotraits.h5 import H5File
from tvb.core.services.project_service import ProjectService


class ExportManager(object):
    """
    This class provides basic methods for exporting data types of projects in different formats.
    """
    all_exporters = {}  # Dictionary containing all available exporters
    export_folder = None
    EXPORT_FOLDER_NAME = "EXPORT_TMP"
    EXPORTED_SIMULATION_NAME = "exported_simulation"
    EXPORTED_SIMULATION_DTS_DIR = "datatypes"
    ZIP_FILE_EXTENSION = "zip"
    logger = get_logger(__name__)

    def __init__(self):
        # Here we register all available data type exporters
        # If new exporters supported, they should be added here
        self._register_exporter(TVBExporter())
        self._register_exporter(TVBLinkedExporter())
        self.export_folder = os.path.join(TvbProfile.current.TVB_STORAGE, self.EXPORT_FOLDER_NAME)
        self.files_helper = FilesHelper()

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

    def export_data(self, data, exporter_id, project):
        """
        Export provided data using given exporter
        :param data: data type to be exported
        :param exporter_id: identifier of the exporter to be used
        :param project: project that contains data to be exported

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

        if project is None:
            raise ExportException("Please provide the project where data files are stored")

        # Now we start the real export
        if not exporter.accepts(data):
            raise InvalidExportDataException("Current data can not be exported by specified exporter")

        # Now compute and create folder where to store exported data
        # This will imply to generate a folder which is unique for each export
        data_export_folder = None
        try:
            data_export_folder = self._build_data_export_folder(data)
            self.logger.debug("Start export of data: %s" % data.type)
            export_data = exporter.export(data, data_export_folder, project)
        finally:
            # In case export did not generated any file delete folder
            if data_export_folder is not None and len(os.listdir(data_export_folder)) == 0:
                os.rmdir(data_export_folder)

        return export_data

    def _export_linked_datatypes(self, project, zip_file):
        linked_paths = ProjectService().get_linked_datatypes_storage_path(project)

        if not linked_paths:
            # do not export an empty operation
            return

        # Make a import operation which will contain links to other projects
        algo = dao.get_algorithm_by_module(TVB_IMPORTER_MODULE, TVB_IMPORTER_CLASS)
        op = model_operation.Operation(None, None, project.id, algo.id)
        op.project = project
        op.algorithm = algo
        op.id = 'links-to-external-projects'
        op.start_now()
        op.mark_complete(model_operation.STATUS_FINISHED)

        op_folder = self.files_helper.get_operation_folder(op.project.name, op.id)
        op_folder_name = os.path.basename(op_folder)

        # add linked datatypes to archive in the import operation
        for pth in linked_paths:
            zip_pth = op_folder_name + '/' + os.path.basename(pth)
            zip_file.write(pth, zip_pth)

        # remove these files, since we only want them in export archive
        self.files_helper.remove_folder(op_folder)

    def export_project(self, project, optimize_size=False):
        """
        Given a project root and the TVB storage_path, create a ZIP
        ready for export.
        :param project: project object which identifies project to be exported
        """
        if project is None:
            raise ExportException("Please provide project to be exported")

        project_folder = self.files_helper.get_project_folder(project)
        project_datatypes = dao.get_datatypes_in_project(project.id, only_visible=optimize_size)
        to_be_exported_folders = []
        considered_op_ids = []
        folders_to_exclude = self._get_op_with_errors(project.id)

        if optimize_size:
            # take only the DataType with visibility flag set ON
            for dt in project_datatypes:
                op_id = dt.fk_from_operation
                if op_id not in considered_op_ids:
                    to_be_exported_folders.append({'folder': self.files_helper.get_project_folder(project, str(op_id)),
                                                   'archive_path_prefix': str(op_id) + os.sep,
                                                   'exclude': folders_to_exclude})
                    considered_op_ids.append(op_id)

        else:
            folders_to_exclude.append("TEMP")
            to_be_exported_folders.append({'folder': project_folder,
                                           'archive_path_prefix': '', 'exclude': folders_to_exclude})

        # Compute path and name of the zip file
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M")
        zip_file_name = "%s_%s.%s" % (date_str, project.name, self.ZIP_FILE_EXTENSION)

        export_folder = self._build_data_export_folder(project)
        result_path = os.path.join(export_folder, zip_file_name)

        with TvbZip(result_path, "w") as zip_file:
            # Pack project [filtered] content into a ZIP file:
            self.logger.debug("Done preparing, now we will write folders " + str(len(to_be_exported_folders)))
            self.logger.debug(str(to_be_exported_folders))
            for pack in to_be_exported_folders:
                zip_file.write_folder(**pack)
            self.logger.debug("Done exporting files, now we will export linked DTs")
            self._export_linked_datatypes(project, zip_file)
            # Make sure the Project.xml file gets copied:
            if optimize_size:
                self.logger.debug("Done linked, now we write the project xml")
                zip_file.write(self.files_helper.get_project_meta_file_path(project.name),
                               self.files_helper.TVB_PROJECT_FILE)
            self.logger.debug("Done, closing")

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

    def _build_data_export_folder(self, data):
        """
        This method computes the folder where results of an export operation will be
        stored for a while (e.g until download is done; or for 1 day)
        """
        now = datetime.now()
        date_str = "%d-%d-%d_%d-%d-%d_%d" % (now.year, now.month, now.day, now.hour,
                                             now.minute, now.second, now.microsecond)
        tmp_str = date_str + "@" + data.gid
        data_export_folder = os.path.join(self.export_folder, tmp_str)
        self.files_helper.check_created(data_export_folder)

        return data_export_folder

    def export_simulator_configuration(self, burst_id):
        burst = dao.get_burst_by_id(burst_id)
        if burst is None:
            raise InvalidExportDataException("Could not find burst with ID " + str(burst_id))

        op_folder = self.files_helper.get_project_folder(burst.project, str(burst.fk_simulation))
        tmp_export_folder = self._build_data_export_folder(burst)
        tmp_sim_folder = os.path.join(tmp_export_folder, self.EXPORTED_SIMULATION_NAME)

        if not os.path.exists(tmp_sim_folder):
            os.makedirs(tmp_sim_folder)

        all_view_model_paths, all_datatype_paths = h5.gather_references_of_view_model(burst.simulator_gid, op_folder)

        burst_path = h5.determine_filepath(burst.gid, op_folder)
        all_view_model_paths.append(burst_path)

        for vm_path in all_view_model_paths:
            dest = os.path.join(tmp_sim_folder, os.path.basename(vm_path))
            self.files_helper.copy_file(vm_path, dest)

        for dt_path in all_datatype_paths:
            dest = os.path.join(tmp_sim_folder, self.EXPORTED_SIMULATION_DTS_DIR, os.path.basename(dt_path))
            self.files_helper.copy_file(dt_path, dest)

        main_vm_path = h5.determine_filepath(burst.simulator_gid, tmp_sim_folder)
        H5File.remove_metadata_param(main_vm_path, 'history_gid')

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M")
        zip_file_name = "%s_%s.%s" % (date_str, str(burst_id), self.ZIP_FILE_EXTENSION)

        result_path = os.path.join(tmp_export_folder, zip_file_name)
        with TvbZip(result_path, "w") as zip_file:
            zip_file.write_folder(tmp_sim_folder)

        self.files_helper.remove_folder(tmp_sim_folder)
        return result_path
