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
.. moduleauthor:: Adrian Dordea <adrian.dordea@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
import json
import os
import shutil
from cgi import FieldStorage
from datetime import datetime

from cherrypy._cpreqbody import Part
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm.attributes import manager_of_class
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.config import VIEW_MODEL2ADAPTER, TVB_IMPORTER_MODULE, TVB_IMPORTER_CLASS
from tvb.config.algorithm_categories import UploadAlgorithmCategoryConfig, DEFAULTDATASTATE_INTERMEDIATE
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.file.exceptions import FileStructureException, MissingDataSetException
from tvb.core.entities.file.exceptions import IncompatibleFileManagerException
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.files_update_manager import FilesUpdateManager
from tvb.core.entities.file.simulator.burst_configuration_h5 import BurstConfigurationH5
from tvb.core.entities.file.xml_metadata_handlers import XMLReader
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.model.model_datatype import DataTypeGroup
from tvb.core.entities.model.model_operation import ResultFigure, Operation, STATUS_FINISHED, STATUS_ERROR, \
    OperationGroup
from tvb.core.entities.model.model_project import Project
from tvb.core.entities.storage import dao, transactional
from tvb.core.neocom import h5
from tvb.core.neotraits.db import HasTraitsIndex
from tvb.core.neotraits.h5 import H5File, ViewModelH5
from tvb.core.project_versions.project_update_manager import ProjectUpdateManager
from tvb.core.services.algorithm_service import AlgorithmService
from tvb.core.services.exceptions import ImportException, ServicesBaseException, MissingReferenceException

OPERATION_XML = "Operation.xml"


class Operation2ImportData(object):
    # Plain Object for transporting operation related data before import
    def __init__(self, operation, operation_folder, main_view_model=None,
                 dt_paths=None, all_view_model_files=None, is_fake=False, info_from_xml=None):
        self.operation = operation
        self.operation_folder = operation_folder
        self.main_view_model = main_view_model
        self.dt_paths = dt_paths
        self.all_view_model_files = all_view_model_files
        self.is_self_generated = is_fake
        self.info_from_xml = info_from_xml

    @property
    def is_old_form(self):
        return (self.operation is not None and hasattr(self.operation, "import_file") and
                self.operation.import_file is not None and self.main_view_model is None)

    @property
    def order_field(self):
        return self.operation.create_date if (self.operation is not None) else datetime.now()


class ImportService(object):
    """
    Service for importing TVB entities into system.
    It supports TVB exported H5 files as input, but it should also handle H5 files
    generated outside of TVB, as long as they respect the same structure.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.user_id = None
        self.files_helper = FilesHelper()
        self.created_projects = []

    def _download_and_unpack_project_zip(self, uploaded, uq_file_name, temp_folder):

        if isinstance(uploaded, FieldStorage) or isinstance(uploaded, Part):
            if not uploaded.file:
                raise ImportException("Please select the archive which contains the project structure.")
            with open(uq_file_name, 'wb') as file_obj:
                self.files_helper.copy_file(uploaded.file, file_obj)
        else:
            shutil.copy2(uploaded, uq_file_name)

        try:
            self.files_helper.unpack_zip(uq_file_name, temp_folder)
        except FileStructureException as excep:
            self.logger.exception(excep)
            raise ImportException("Bad ZIP archive provided. A TVB exported project is expected!")

    @staticmethod
    def _compute_unpack_path():
        """
        :return: the name of the folder where to expand uploaded zip
        """
        now = datetime.now()
        date_str = "%d-%d-%d_%d-%d-%d_%d" % (now.year, now.month, now.day, now.hour,
                                             now.minute, now.second, now.microsecond)
        uq_name = "%s-ImportProject" % date_str
        return os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, uq_name)

    @transactional
    def import_project_structure(self, uploaded, user_id):
        """
        Execute import operations:

        1. check if ZIP or folder
        2. find all project nodes
        3. for each project node:
            - create project
            - create all operations and groups
            - import all images
            - create all dataTypes
        """

        self.user_id = user_id
        self.created_projects = []

        # Now compute the name of the folder where to explode uploaded ZIP file
        temp_folder = self._compute_unpack_path()
        uq_file_name = temp_folder + ".zip"

        try:
            self._download_and_unpack_project_zip(uploaded, uq_file_name, temp_folder)
            self._import_projects_from_folder(temp_folder)

        except Exception as excep:
            self.logger.exception("Error encountered during import. Deleting projects created during this operation.")
            # Remove project folders created so far.
            # Note that using the project service to remove the projects will not work,
            # because we do not have support for nested transaction.
            # Removing from DB is not necessary because in transactional env a simple exception throw
            # will erase everything to be inserted.
            for project in self.created_projects:
                project_path = os.path.join(TvbProfile.current.TVB_STORAGE, FilesHelper.PROJECTS_FOLDER, project.name)
                shutil.rmtree(project_path)
            raise ImportException(str(excep))

        finally:
            # Now delete uploaded file
            if os.path.exists(uq_file_name):
                os.remove(uq_file_name)
            # Now delete temporary folder where uploaded ZIP was exploded.
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)

    def _import_projects_from_folder(self, temp_folder):
        """
        Process each project from the uploaded pack, to extract names.
        """
        project_roots = []
        for root, _, files in os.walk(temp_folder):
            if FilesHelper.TVB_PROJECT_FILE in files:
                project_roots.append(root)

        for temp_project_path in project_roots:
            update_manager = ProjectUpdateManager(temp_project_path)
            update_manager.run_all_updates()
            project = self.__populate_project(temp_project_path)
            # Populate the internal list of create projects so far, for cleaning up folders, in case of failure
            self.created_projects.append(project)
            # Ensure project final folder exists on disk
            project_path = self.files_helper.get_project_folder(project)
            shutil.move(os.path.join(temp_project_path, FilesHelper.TVB_PROJECT_FILE), project_path)
            # Now import project operations with their results
            self.import_project_operations(project, temp_project_path)
            # Import images and move them from temp into target
            self._store_imported_images(project, temp_project_path, project.name)

    def _load_datatypes_from_operation_folder(self, src_op_path, operation_entity, datatype_group):
        """
        Loads datatypes from operation folder
        :returns: Datatype entities as dict {original_path: Dt instance}
        """
        all_datatypes = {}
        for file_name in os.listdir(src_op_path):
            if file_name.endswith(FilesHelper.TVB_STORAGE_FILE_EXTENSION):
                h5_file = os.path.join(src_op_path, file_name)
                try:
                    file_update_manager = FilesUpdateManager()
                    file_update_manager.upgrade_file(h5_file)
                    datatype = self.load_datatype_from_file(h5_file, operation_entity.id,
                                                            datatype_group, operation_entity.fk_launched_in)
                    all_datatypes[h5_file] = datatype

                except IncompatibleFileManagerException:
                    os.remove(h5_file)
                    self.logger.warning("Incompatible H5 file will be ignored: %s" % h5_file)
                    self.logger.exception("Incompatibility details ...")
        return all_datatypes

    def _store_imported_datatypes_in_db(self, project, all_datatypes):
        # type: (Project, dict) -> int
        sorted_dts = sorted(all_datatypes.items(),
                            key=lambda dt_item: dt_item[1].create_date or datetime.now())

        count = 0
        for dt_path, datatype in sorted_dts:
            datatype_already_in_tvb = dao.get_datatype_by_gid(datatype.gid)
            if not datatype_already_in_tvb:
                self.store_datatype(datatype, dt_path)
                count += 1
            else:
                AlgorithmService.create_link([datatype_already_in_tvb.id], project.id)

            file_path = h5.h5_file_for_index(datatype).path
            h5_class = H5File.h5_class_from_file(file_path)
            reference_list = h5_class(file_path).gather_references()

            for _, reference_gid in reference_list:
                if not reference_gid:
                    continue

                ref_index = dao.get_datatype_by_gid(reference_gid.hex)
                if ref_index is None:
                    os.remove(file_path)
                    dao.remove_entity(datatype.__class__, datatype.id)
                    raise MissingReferenceException(
                        'Imported file depends on datatypes that do not exist. Please upload '
                        'those first!')

        return count

    def _store_imported_images(self, project, temp_project_path, project_name):
        """
        Import all images from project
        """
        images_root = os.path.join(temp_project_path, FilesHelper.IMAGES_FOLDER)
        target_images_path = self.files_helper.get_images_folder(project_name)
        for root, _, files in os.walk(images_root):
            for metadata_file in files:
                if metadata_file.endswith(FilesHelper.TVB_FILE_EXTENSION):
                    self._import_image(root, metadata_file, project.id, target_images_path)

    def _retrieve_operations_in_order(self, project, import_path):
        # type: (Project, str) -> list[Operation2ImportData]
        retrieved_operations = []

        for root, _, files in os.walk(import_path):
            if OPERATION_XML in files:
                # Previous Operation format for uploading previous versions of projects
                operation_file_path = os.path.join(root, OPERATION_XML)
                operation, operation_xml_parameters = self.__build_operation_from_file(project, operation_file_path)
                operation.import_file = operation_file_path
                self.logger.debug("Found operation in old XML format: " + str(operation))
                retrieved_operations.append(
                    Operation2ImportData(operation, root, info_from_xml=operation_xml_parameters))

            else:
                # We strive for the new format with ViewModelH5
                main_view_model = None
                dt_paths = []
                all_view_model_files = []
                for file in files:
                    if file.endswith(FilesHelper.TVB_STORAGE_FILE_EXTENSION):
                        h5_file = os.path.join(root, file)
                        try:
                            h5_class = H5File.h5_class_from_file(h5_file)
                            if h5_class is ViewModelH5:
                                all_view_model_files.append(h5_file)
                                if not main_view_model:
                                    view_model = h5.load_view_model_from_file(h5_file)
                                    if type(view_model) in VIEW_MODEL2ADAPTER.keys():
                                        main_view_model = view_model
                            else:
                                file_update_manager = FilesUpdateManager()
                                file_update_manager.upgrade_file(h5_file)
                                dt_paths.append(h5_file)
                        except Exception:
                            self.logger.warning("Unreadable H5 file will be ignored: %s" % h5_file)

                if main_view_model is not None:
                    alg = VIEW_MODEL2ADAPTER[type(main_view_model)]
                    op_group_id = None
                    if main_view_model.operation_group_gid:
                        op_group = dao.get_operationgroup_by_gid(main_view_model.operation_group_gid.hex)
                        if not op_group:
                            op_group = OperationGroup(project.id, ranges=json.loads(main_view_model.ranges),
                                                      gid=main_view_model.operation_group_gid.hex)
                            op_group = dao.store_entity(op_group)
                        op_group_id = op_group.id
                    operation = Operation(main_view_model.gid.hex, project.fk_admin, project.id, alg.id,
                                          status=STATUS_FINISHED,
                                          user_group=main_view_model.generic_attributes.operation_tag,
                                          start_date=datetime.now(), completion_date=datetime.now(),
                                          op_group_id=op_group_id, range_values=main_view_model.range_values)
                    operation.create_date = main_view_model.create_date
                    operation.visible = main_view_model.generic_attributes.visible
                    self.logger.debug("Found main ViewModel to create operation for it: " + str(operation))

                    retrieved_operations.append(
                        Operation2ImportData(operation, root, main_view_model, dt_paths, all_view_model_files))

                elif len(dt_paths) > 0:
                    alg = dao.get_algorithm_by_module(TVB_IMPORTER_MODULE, TVB_IMPORTER_CLASS)
                    default_adapter = ABCAdapter.build_adapter(alg)
                    view_model = default_adapter.get_view_model_class()()
                    view_model.data_file = dt_paths[0]
                    vm_path = h5.store_view_model(view_model, root)
                    all_view_model_files.append(vm_path)
                    operation = Operation(view_model.gid.hex, project.fk_admin, project.id, alg.id,
                                          status=STATUS_FINISHED,
                                          start_date=datetime.now(), completion_date=datetime.now())
                    self.logger.debug("Found no ViewModel in folder, so we default to " + str(operation))

                    retrieved_operations.append(
                        Operation2ImportData(operation, root, view_model, dt_paths, all_view_model_files, True))

        return sorted(retrieved_operations, key=lambda op_data: op_data.order_field)

    def import_project_operations(self, project, import_path):
        """
        This method scans provided folder and identify all operations that needs to be imported
        """
        imported_operations = []
        ordered_operations = self._retrieve_operations_in_order(project, import_path)

        for operation_data in ordered_operations:

            if operation_data.is_old_form:
                operation_entity, datatype_group = self.__import_operation(operation_data.operation)
                new_op_folder = self.files_helper.get_project_folder(project, str(operation_entity.id))

                try:
                    operation_datatypes = self._load_datatypes_from_operation_folder(operation_data.operation_folder,
                                                                                     operation_entity, datatype_group)
                    # Create and store view_model from operation
                    view_model = self._get_new_form_view_model(operation_entity, operation_data.info_from_xml)
                    h5.store_view_model(view_model, new_op_folder)
                    view_model_disk_size = FilesHelper.compute_recursive_h5_disk_usage(new_op_folder)
                    operation_entity.view_model_disk_size = view_model_disk_size
                    operation_entity.view_model_gid = view_model.gid.hex
                    dao.store_entity(operation_entity)

                    self._store_imported_datatypes_in_db(project, operation_datatypes)
                    imported_operations.append(operation_entity)
                except MissingReferenceException:
                    operation_entity.status = STATUS_ERROR
                    dao.store_entity(operation_entity)

            elif operation_data.main_view_model is not None:
                operation_entity = dao.store_entity(operation_data.operation)
                dt_group = None
                op_group = dao.get_operationgroup_by_id(operation_entity.fk_operation_group)
                if op_group:
                    dt_group = dao.get_datatypegroup_by_op_group_id(op_group.id)
                    if not dt_group:
                        first_op = dao.get_operations_in_group(op_group.id, only_first_operation=True)
                        dt_group = DataTypeGroup(op_group, operation_id=first_op.id,
                                                 state=DEFAULTDATASTATE_INTERMEDIATE)
                        dt_group = dao.store_entity(dt_group)
                # Store the DataTypes in db
                dts = {}
                for dt_path in operation_data.dt_paths:
                    dt = self.load_datatype_from_file(dt_path, operation_entity.id, dt_group, project.id)
                    if isinstance(dt, BurstConfiguration):
                        if op_group:
                            dt.fk_operation_group = op_group.id
                        dao.store_entity(dt)
                        self.store_datatype(dt)
                    else:
                        dts[dt_path] = dt
                        if op_group:
                            op_group.fill_operationgroup_name(dt.type)
                            dao.store_entity(op_group)
                try:
                    stored_dts_count = self._store_imported_datatypes_in_db(project, dts)

                    if operation_data.main_view_model.is_metric_operation:
                        self._update_burst_metric(operation_entity)

                    if stored_dts_count > 0 or not operation_data.is_self_generated:
                        imported_operations.append(operation_entity)
                        new_op_folder = self.files_helper.get_project_folder(project, str(operation_entity.id))
                        for h5_file in operation_data.all_view_model_files:
                            shutil.move(h5_file, new_op_folder)
                    else:
                        # In case all Dts under the current operation were Links and the ViewModel is dummy,
                        # don't keep the Operation empty in DB
                        dao.remove_entity(Operation, operation_entity.id)
                except MissingReferenceException:
                    operation_entity.status = STATUS_ERROR
                    dao.store_entity(operation_entity)

            else:
                self.logger.warning("Folder %s will be ignored, as we could not find a serialized "
                                    "operation or DTs inside!" % operation_data.operation_folder)

        self._update_dt_groups(project.id)
        self._update_burst_configurations(project.id)
        return imported_operations

    @staticmethod
    def _get_new_form_view_model(operation, xml_parameters):
        # type (Operation) -> ViewModel
        ad = ABCAdapter.build_adapter(operation.algorithm)
        view_model = ad.get_view_model_class()()

        if xml_parameters:
            params = json.loads(xml_parameters)
            declarative_attrs = type(view_model).declarative_attrs

            for param in params:
                new_param_name = param
                if param[0] == "_":
                    new_param_name = param[1:]
                new_param_name = new_param_name.lower()
                if new_param_name in declarative_attrs:
                    setattr(view_model, new_param_name, params[param])
        return view_model

    def _import_image(self, src_folder, metadata_file, project_id, target_images_path):
        """
        Create and store a image entity.
        """
        figure_dict = XMLReader(os.path.join(src_folder, metadata_file)).read_metadata()
        actual_figure = os.path.join(src_folder, os.path.split(figure_dict['file_path'])[1])
        if not os.path.exists(actual_figure):
            self.logger.warning("Expected to find image path %s .Skipping" % actual_figure)
            return
        figure_dict['fk_user_id'] = self.user_id
        figure_dict['fk_project_id'] = project_id
        figure_entity = manager_of_class(ResultFigure).new_instance()
        figure_entity = figure_entity.from_dict(figure_dict)
        stored_entity = dao.store_entity(figure_entity)

        # Update image meta-data with the new details after import
        figure = dao.load_figure(stored_entity.id)
        shutil.move(actual_figure, target_images_path)
        self.logger.debug("Store imported figure")
        self.files_helper.write_image_metadata(figure)

    def load_datatype_from_file(self, current_file, op_id, datatype_group=None, current_project_id=None):
        # type: (str, int, DataTypeGroup, int) -> HasTraitsIndex
        """
        Creates an instance of datatype from storage / H5 file 
        :returns: DatatypeIndex
        """
        self.logger.debug("Loading DataType from file: %s" % current_file)
        h5_class = H5File.h5_class_from_file(current_file)

        if h5_class is BurstConfigurationH5:
            if current_project_id is None:
                op_entity = dao.get_operationgroup_by_id(op_id)
                current_project_id = op_entity.fk_launched_in
            h5_file = BurstConfigurationH5(current_file)
            burst = BurstConfiguration(current_project_id)
            burst.fk_simulation = op_id
            h5_file.load_into(burst)
            result = burst
        else:
            datatype, generic_attributes = h5.load_with_links(current_file)
            index_class = h5.REGISTRY.get_index_for_datatype(datatype.__class__)
            datatype_index = index_class()
            datatype_index.fill_from_has_traits(datatype)
            datatype_index.fill_from_generic_attributes(generic_attributes)

            # Add all the required attributes
            if datatype_group:
                datatype_index.fk_datatype_group = datatype_group.id
            datatype_index.fk_from_operation = op_id

            associated_file = h5.path_for_stored_index(datatype_index)
            if os.path.exists(associated_file):
                datatype_index.disk_size = FilesHelper.compute_size_on_disk(associated_file)
            result = datatype_index

        return result

    def store_datatype(self, datatype, current_file=None):
        """This method stores data type into DB"""
        try:
            self.logger.debug("Store datatype: %s with Gid: %s" % (datatype.__class__.__name__, datatype.gid))
            # Now move storage file into correct folder if necessary
            if current_file is not None:
                final_path = h5.path_for_stored_index(datatype)
                if final_path != current_file:
                    shutil.move(current_file, final_path)

            return dao.store_entity(datatype)
        except MissingDataSetException as e:
            self.logger.exception(e)
            error_msg = "Datatype %s has missing data and could not be imported properly." % (datatype,)
            raise ImportException(error_msg)
        except IntegrityError as excep:
            self.logger.exception(excep)
            error_msg = "Could not import data with gid: %s. There is already a one with " \
                        "the same name or gid." % datatype.gid
            raise ImportException(error_msg)

    def __populate_project(self, project_path):
        """
        Create and store a Project entity.
        """
        self.logger.debug("Creating project from path: %s" % project_path)
        project_dict = self.files_helper.read_project_metadata(project_path)

        project_entity = manager_of_class(Project).new_instance()
        project_entity = project_entity.from_dict(project_dict, self.user_id)

        try:
            self.logger.debug("Storing imported project")
            return dao.store_entity(project_entity)
        except IntegrityError as excep:
            self.logger.exception(excep)
            error_msg = ("Could not import project: %s with gid: %s. There is already a "
                         "project with the same name or gid.") % (project_entity.name, project_entity.gid)
            raise ImportException(error_msg)

    def __build_operation_from_file(self, project, operation_file):
        """
        Create Operation entity from metadata file.
        """
        operation_dict = XMLReader(operation_file).read_metadata()
        operation_entity = manager_of_class(Operation).new_instance()
        return operation_entity.from_dict(operation_dict, dao, self.user_id, project.gid)

    @staticmethod
    def __import_operation(operation_entity):
        """
        Store a Operation entity.
        """
        operation_entity = dao.store_entity(operation_entity)
        operation_group_id = operation_entity.fk_operation_group
        datatype_group = None

        if operation_group_id is not None:
            try:
                datatype_group = dao.get_datatypegroup_by_op_group_id(operation_group_id)
            except SQLAlchemyError:
                # If no dataType group present for current op. group, create it.
                operation_group = dao.get_operationgroup_by_id(operation_group_id)
                datatype_group = DataTypeGroup(operation_group, operation_id=operation_entity.id)
                datatype_group.state = UploadAlgorithmCategoryConfig.defaultdatastate
                datatype_group = dao.store_entity(datatype_group)

        return operation_entity, datatype_group

    def import_simulator_configuration_zip(self, zip_file):
        # Now compute the name of the folder where to explode uploaded ZIP file
        temp_folder = self._compute_unpack_path()
        uq_file_name = temp_folder + ".zip"

        if isinstance(zip_file, FieldStorage) or isinstance(zip_file, Part):
            if not zip_file.file:
                raise ServicesBaseException("Could not process the given ZIP file...")

            with open(uq_file_name, 'wb') as file_obj:
                self.files_helper.copy_file(zip_file.file, file_obj)
        else:
            shutil.copy2(zip_file, uq_file_name)

        try:
            self.files_helper.unpack_zip(uq_file_name, temp_folder)
            return temp_folder
        except FileStructureException as excep:
            raise ServicesBaseException("Could not process the given ZIP file..." + str(excep))

    def _update_burst_metric(self, operation_entity):
        burst_config = dao.get_burst_for_operation_id(operation_entity.id)
        if burst_config.ranges:
            if burst_config.fk_metric_operation_group is None:
                burst_config.fk_metric_operation_group = operation_entity.fk_operation_group
        dao.store_entity(burst_config)

    def _update_dt_groups(self, project_id):
        dt_groups = dao.get_datatypegroup_for_project(project_id)
        for dt_group in dt_groups:
            dt_group.count_results = dao.count_datatypes_in_group(dt_group.id)
            dts_in_group = dao.get_datatypes_from_datatype_group(dt_group.id)
            if dts_in_group:
                dt_group.fk_parent_burst = dts_in_group[0].fk_parent_burst
            dao.store_entity(dt_group)

    def _update_burst_configurations(self, project_id):
        burst_configs = dao.get_bursts_for_project(project_id)
        for burst_config in burst_configs:
            burst_config.datatypes_number = dao.count_datatypes_in_burst(burst_config.gid)
            dao.store_entity(burst_config)
