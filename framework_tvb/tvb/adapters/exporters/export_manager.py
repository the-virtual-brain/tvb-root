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

.. moduleauthor:: Calin Pavel
"""

import os
import json
from datetime import datetime
from tvb.adapters.exporters.tvb_export import TVBExporter
from tvb.adapters.exporters.exceptions import ExportException, InvalidExportDataException
from tvb.basic.profile import TvbProfile
from tvb.basic.logger.builder import get_logger
from tvb.config import TVB_IMPORTER_MODULE, TVB_IMPORTER_CLASS
from tvb.core.entities.model import model_operation
from tvb.core.entities.model.model_burst import BURST_INFO_FILE, BURSTS_DICT_KEY, DT_BURST_MAP
from tvb.core.entities.file.files_helper import FilesHelper, TvbZip
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5

LOG = get_logger(__name__)
BURST_PAGE_SIZE = 100
DATAYPES_PAGE_SIZE = 100

KEY_DT_GID = 'gid'
KEY_OPERATION_ID = 'operation'
KEY_BURST_ID = 'burst'
KEY_DT_DATE = "dt_date"


class ExportManager:
    """
    This class provides basic methods for exporting data types of projects in different formats.
    """
    all_exporters = {}  # Dictionary containing all available exporters
    export_folder = None
    EXPORT_FOLDER_NAME = "EXPORT_TMP"
    ZIP_FILE_EXTENSION = "zip"

    def __init__(self):
        # Here we register all available data type exporters
        # If new exporters supported, they should be added here
        # Todo: uploaders and visualizers are registered using a different method.
        self._register_exporter(TVBExporter())
        self.export_folder = os.path.join(TvbProfile.current.TVB_STORAGE, self.EXPORT_FOLDER_NAME)


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

        LOG.debug("Trying to determine exporters valid for %s" % data.type)

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

            # Here is the real export                    
            LOG.debug("Start export of data: %s" % data.type)
            export_data = exporter.export(data, data_export_folder, project)
        finally:
            # In case export did not generated any file delete folder
            if data_export_folder is not None and len(os.listdir(data_export_folder)) == 0:
                os.rmdir(data_export_folder)

        return export_data


    @staticmethod
    def _get_linked_datatypes_storage_path(project):
        """
        :return: the file paths to the datatypes that are linked in `project`
        """
        paths = []
        for lnk_dt in dao.get_linked_datatypes_in_project(project.id):
            # get datatype as a mapped type
            lnk_dt = dao.get_datatype_by_gid(lnk_dt.gid)
            path = h5.path_for_stored_index(lnk_dt)
            if path is not None:
                paths.append(path)
            else:
                LOG.warning("Problem when trying to retrieve path on %s:%s for export!" % (lnk_dt.type, lnk_dt.gid))
        return paths


    def _export_linked_datatypes(self, project, zip_file):
        files_helper = FilesHelper()
        linked_paths = self._get_linked_datatypes_storage_path(project)

        if not linked_paths:
            # do not export an empty operation
            return

        # Make a import operation which will contain links to other projects
        algo = dao.get_algorithm_by_module(TVB_IMPORTER_MODULE, TVB_IMPORTER_CLASS)
        op = model_operation.Operation(None, project.id, algo.id, '')
        op.project = project
        op.algorithm = algo
        op.id = 'links-to-external-projects'
        op.start_now()
        op.mark_complete(model_operation.STATUS_FINISHED)

        # write operation.xml to disk
        files_helper.write_operation_metadata(op)
        op_folder = files_helper.get_operation_folder(op.project.name, op.id)
        operation_xml = files_helper.get_operation_meta_file_path(op.project.name, op.id)
        op_folder_name = os.path.basename(op_folder)

        # add operation.xml
        zip_file.write(operation_xml, op_folder_name + '/' + os.path.basename(operation_xml))

        # add linked datatypes to archive in the import operation
        for pth in linked_paths:
            zip_pth = op_folder_name + '/' + os.path.basename(pth)
            zip_file.write(pth, zip_pth)

        # remove these files, since we only want them in export archive
        files_helper.remove_folder(op_folder)


    def _export_bursts(self, project, project_datatypes, zip_file):

        bursts_dict = {}

        bursts_count = dao.get_bursts_for_project(project.id, count=True)
        for start_idx in range(0, bursts_count, BURST_PAGE_SIZE):
            bursts = dao.get_bursts_for_project(project.id, page_start=start_idx, page_size=BURST_PAGE_SIZE)
            # for burst in bursts:
            #     one_info = self._build_burst_export_dict(burst)
            #     # Save data in dictionary form so we can just save it as a json later on
            #     bursts_dict[burst.id] = one_info

        datatype_burst_mapping = {}
        for dt in project_datatypes:
            datatype_burst_mapping[dt[KEY_DT_GID]] = dt[KEY_BURST_ID]

        project_folder = FilesHelper().get_project_folder(project)
        bursts_file_name = os.path.join(project_folder, BURST_INFO_FILE)
        burst_info = {BURSTS_DICT_KEY: bursts_dict,
                      DT_BURST_MAP: datatype_burst_mapping}

        zip_file.writestr(os.path.basename(bursts_file_name), json.dumps(burst_info))


    def export_project(self, project, optimize_size=False):
        """
        Given a project root and the TVB storage_path, create a ZIP
        ready for export.
        :param project: project object which identifies project to be exported
        """
        if project is None:
            raise ExportException("Please provide project to be exported")

        files_helper = FilesHelper()
        project_folder = files_helper.get_project_folder(project)
        project_datatypes = self._gather_project_datatypes(project, optimize_size)
        to_be_exported_folders = []
        considered_op_ids = []

        if optimize_size:
            # take only the DataType with visibility flag set ON
            for dt in project_datatypes:
                if dt[KEY_OPERATION_ID] not in considered_op_ids:
                    to_be_exported_folders.append({'folder': files_helper.get_project_folder(project,
                                                                                             str(dt[KEY_OPERATION_ID])),
                                                   'archive_path_prefix': str(dt[KEY_OPERATION_ID]) + os.sep})
                    considered_op_ids.append(dt[KEY_OPERATION_ID])

        else:
            to_be_exported_folders.append({'folder': project_folder,
                                           'archive_path_prefix': '', 'exclude': ["TEMP"]})

        # Compute path and name of the zip file
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M")
        zip_file_name = "%s_%s.%s" % (date_str, project.name, self.ZIP_FILE_EXTENSION)

        export_folder = self._build_data_export_folder(project)
        result_path = os.path.join(export_folder, zip_file_name)

        with TvbZip(result_path, "w") as zip_file:
            # Pack project [filtered] content into a ZIP file:
            LOG.debug("Done preparing, now we will write folders " + str(len(to_be_exported_folders)))
            LOG.debug(str(to_be_exported_folders))
            for pack in to_be_exported_folders:
                zip_file.write_folder(**pack)
            LOG.debug("Done exporting files, now we will write the burst configurations...")
            self._export_bursts(project, project_datatypes, zip_file)
            LOG.debug("Done exporting burst configurations, now we will export linked DTs")
            self._export_linked_datatypes(project, zip_file)
            ## Make sure the Project.xml file gets copied:
            if optimize_size:
                LOG.debug("Done linked, now we write the project xml")
                zip_file.write(files_helper.get_project_meta_file_path(project.name), files_helper.TVB_PROJECT_FILE)
            LOG.debug("Done, closing")

        return result_path


    @staticmethod
    def _gather_project_datatypes(project, only_visible):

        project_datatypes = []

        dts = dao.get_datatypes_in_project(project.id, only_visible=only_visible)
        for dt in dts:
            project_datatypes.append({KEY_DT_GID: dt.gid,
                                      KEY_BURST_ID: dt.fk_parent_burst,
                                      KEY_OPERATION_ID: dt.fk_from_operation,
                                      KEY_DT_DATE: dt.create_date})
        return project_datatypes

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
        files_helper = FilesHelper()
        files_helper.check_created(data_export_folder)

        return data_export_folder

    def export_simulator_configuration(self, burst_id):
        burst = dao.get_burst_by_id(burst_id)
        if burst is None:
            raise InvalidExportDataException("Could not find burst with ID " + str(burst_id))

        op_folder = FilesHelper().get_project_folder(burst.project, str(burst.fk_simulation_id))

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M")
        zip_file_name = "%s_%s.%s" % (date_str, str(burst_id), self.ZIP_FILE_EXTENSION)
        tmp_export_folder = self._build_data_export_folder(burst)
        result_path = os.path.join(tmp_export_folder, zip_file_name)

        with TvbZip(result_path, "w") as zip_file:
            for filename in os.listdir(op_folder):
                zip_file.write(os.path.join(op_folder, filename), filename)

        return result_path
