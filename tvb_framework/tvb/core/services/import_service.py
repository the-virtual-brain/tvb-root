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
.. moduleauthor:: Adrian Dordea <adrian.dordea@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import json
import os
import shutil
import uuid
from cgi import FieldStorage
from datetime import datetime
from cherrypy._cpreqbody import Part
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.attributes import manager_of_class
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.ex import TraitTypeError, TraitAttributeError
from tvb.basic.profile import TvbProfile
from tvb.config import VIEW_MODEL2ADAPTER, TVB_IMPORTER_MODULE, TVB_IMPORTER_CLASS
from tvb.config.algorithm_categories import UploadAlgorithmCategoryConfig, DEFAULTDATASTATE_INTERMEDIATE
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.exceptions import IntrospectionException
from tvb.core.entities import load
from tvb.core.entities.file.files_update_manager import FilesUpdateManager
from tvb.core.entities.file.simulator.burst_configuration_h5 import BurstConfigurationH5
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
from tvb.core.services.exceptions import ImportException, ServicesBaseException, MissingReferenceException, \
    DatatypeGroupImportException
from tvb.storage.h5.file.exceptions import MissingDataSetException, FileStructureException, \
    IncompatibleFileManagerException
from tvb.storage.storage_interface import StorageInterface

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
        self.storage_interface = StorageInterface()
        self.created_projects = []
        self.view_model2adapter = self._populate_view_model2adapter()

    def _download_and_unpack_project_zip(self, uploaded, uq_file_name, temp_folder):

        if isinstance(uploaded, (FieldStorage, Part)):
            if not uploaded.file:
                raise ImportException("Please select the archive which contains the project structure.")
            with open(uq_file_name, 'wb') as file_obj:
                self.storage_interface.copy_file(uploaded.file, file_obj)
        else:
            shutil.copy2(uploaded, uq_file_name)

        try:
            self.storage_interface.unpack_zip(uq_file_name, temp_folder)
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
            self._import_project_from_folder(temp_folder)

        except Exception as excep:
            self.logger.exception("Error encountered during import. Deleting projects created during this operation.")
            # Remove project folders created so far.
            # Note that using the project service to remove the projects will not work,
            # because we do not have support for nested transaction.
            # Removing from DB is not necessary because in transactional env a simple exception throw
            # will erase everything to be inserted.
            for project in self.created_projects:
                self.storage_interface.remove_project(project)
            raise ImportException(str(excep))

        finally:
            # Now delete uploaded file and temporary folder where uploaded ZIP was exploded.
            self.storage_interface.remove_files([uq_file_name, temp_folder])

    def _import_project_from_folder(self, temp_folder):
        """
        Process each project from the uploaded pack, to extract names.
        """
        temp_project_path = None
        for root, _, files in os.walk(temp_folder):
            if StorageInterface.TVB_PROJECT_FILE in files:
                temp_project_path = root
                break

        if temp_project_path is not None:
            update_manager = ProjectUpdateManager(temp_project_path)

            if update_manager.checked_version < 3:
                raise ImportException('Importing projects with versions older than 3 is not supported in TVB 2! '
                                      'Please import the project in TVB 1.5.8 and then launch the current version of '
                                      'TVB in order to upgrade this project!')

            update_manager.run_all_updates()
            project = self.__populate_project(temp_project_path)
            # Populate the internal list of create projects so far, for cleaning up folders, in case of failure
            self.created_projects.append(project)
            # Ensure project final folder exists on disk
            project_path = self.storage_interface.get_project_folder(project.name)
            shutil.move(os.path.join(temp_project_path, StorageInterface.TVB_PROJECT_FILE), project_path)
            # Now import project operations with their results
            self.import_list_of_operations(project, temp_project_path)
            # Import images and move them from temp into target
            self._store_imported_images(project, temp_project_path, project.name)
            if StorageInterface.encryption_enabled():
                project_folder = self.storage_interface.get_project_folder(project.name)
                self.storage_interface.sync_folders(project_folder)
                self.storage_interface.remove_folder(project_folder)

    def _load_datatypes_from_operation_folder(self, src_op_path, operation_entity, datatype_group):
        """
        Loads datatypes from operation folder
        :returns: Datatype entities as dict {original_path: Dt instance}
        """
        all_datatypes = {}
        for file_name in os.listdir(src_op_path):
            if self.storage_interface.ends_with_tvb_storage_file_extension(file_name):
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

    @staticmethod
    def check_import_references(file_path, datatype):
        h5_class = H5File.h5_class_from_file(file_path)
        reference_list = h5_class(file_path).gather_references()

        for _, reference_gid in reference_list:
            if not reference_gid:
                continue

            ref_index = load.load_entity_by_gid(reference_gid)
            if ref_index is None:
                os.remove(file_path)
                dao.remove_entity(datatype.__class__, datatype.id)
                raise MissingReferenceException(
                    'Imported file depends on datatypes that do not exist. Please upload '
                    'those first!')

    def _store_or_link_burst_config(self, burst_config, bc_path):
        bc_already_in_tvb = dao.get_generic_entity(BurstConfiguration, burst_config.gid, 'gid')
        if len(bc_already_in_tvb) == 0:
            self.store_datatype(burst_config, bc_path)
            return 1
        return 0

    def store_or_link_datatype(self, datatype, dt_path, project_id):
        self.check_import_references(dt_path, datatype)
        stored_dt_count = 0
        datatype_already_in_tvb = load.load_entity_by_gid(datatype.gid)
        if not datatype_already_in_tvb:
            self.store_datatype(datatype, dt_path)
            stored_dt_count = 1
        elif datatype_already_in_tvb.parent_operation.project.id != project_id:
            AlgorithmService.create_link(datatype_already_in_tvb.id, project_id)
            if datatype_already_in_tvb.fk_datatype_group:
                AlgorithmService.create_link(datatype_already_in_tvb.fk_datatype_group, project_id)
        return stored_dt_count

    def _store_imported_datatypes_in_db(self, project, all_datatypes):
        # type: (Project, dict) -> int
        sorted_dts = sorted(all_datatypes.items(),
                            key=lambda dt_item: dt_item[1].create_date or datetime.now())
        count = 0
        for dt_path, datatype in sorted_dts:
            count += self.store_or_link_datatype(datatype, dt_path, project.id)
        return count

    def _store_imported_images(self, project, temp_project_path, project_name):
        """
        Import all images from project
        """
        images_root = os.path.join(temp_project_path, StorageInterface.IMAGES_FOLDER)
        target_images_path = self.storage_interface.get_images_folder(project_name)
        for root, _, files in os.walk(images_root):
            for metadata_file in files:
                if self.storage_interface.ends_with_tvb_file_extension(metadata_file):
                    self._import_image(root, metadata_file, project.id, target_images_path)

    def _populate_view_model2adapter(self):
        if len(VIEW_MODEL2ADAPTER) > 0:
            return VIEW_MODEL2ADAPTER
        view_model2adapter = {}
        algos = dao.get_all_algorithms()
        for algo in algos:
            try:
                adapter = ABCAdapter.build_adapter(algo)
                view_model_class = adapter.get_view_model_class()
                view_model2adapter[view_model_class] = algo
            except IntrospectionException:
                self.logger.exception("Could not load %s" % algo)

        return view_model2adapter

    def _retrieve_operations_in_order(self, project, import_path, importer_operation_id=None):
        # type: (Project, str, int) -> list[Operation2ImportData]
        retrieved_operations = []

        for root, _, files in os.walk(import_path):
            if OPERATION_XML in files:
                # Previous Operation format for uploading previous versions of projects
                operation_file_path = os.path.join(root, OPERATION_XML)
                operation, operation_xml_parameters, _ = self.build_operation_from_file(project, operation_file_path)
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
                    if self.storage_interface.ends_with_tvb_storage_file_extension(file):
                        h5_file = os.path.join(root, file)
                        try:
                            h5_class = H5File.h5_class_from_file(h5_file)
                            if h5_class is ViewModelH5:
                                all_view_model_files.append(h5_file)
                                if not main_view_model:
                                    view_model = h5.load_view_model_from_file(h5_file)
                                    if type(view_model) in self.view_model2adapter.keys():
                                        main_view_model = view_model
                            else:
                                file_update_manager = FilesUpdateManager()
                                file_update_manager.upgrade_file(h5_file)
                                dt_paths.append(h5_file)
                        except Exception:
                            self.logger.warning("Unreadable H5 file will be ignored: %s" % h5_file)

                if main_view_model is not None:
                    alg = self.view_model2adapter[type(main_view_model)]
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
                    operation.create_date = datetime.min
                    self.logger.debug("Found no ViewModel in folder, so we default to " + str(operation))

                    if importer_operation_id:
                        operation.id = importer_operation_id

                    retrieved_operations.append(
                        Operation2ImportData(operation, root, view_model, dt_paths, all_view_model_files, True))

        return sorted(retrieved_operations, key=lambda op_data: op_data.order_field)

    def create_view_model(self, operation_entity, operation_data, new_op_folder, generic_attributes=None,
                          add_params=None):
        view_model = self._get_new_form_view_model(operation_entity, operation_data.info_from_xml)
        if add_params is not None:
            for element in add_params:
                key_attr = getattr(view_model, element[0])
                setattr(key_attr, element[1], element[2])

        view_model.range_values = operation_entity.range_values
        op_group = dao.get_operationgroup_by_id(operation_entity.fk_operation_group)
        if op_group:
            view_model.operation_group_gid = uuid.UUID(op_group.gid)
            view_model.ranges = json.dumps(op_group.range_references)
            view_model.is_metric_operation = 'DatatypeMeasure' in op_group.name

        if generic_attributes is not None:
            view_model.generic_attributes = generic_attributes
        view_model.generic_attributes.operation_tag = operation_entity.user_group

        h5.store_view_model(view_model, new_op_folder)
        view_model_disk_size = StorageInterface.compute_recursive_h5_disk_usage(new_op_folder)
        operation_entity.view_model_disk_size = view_model_disk_size
        operation_entity.view_model_gid = view_model.gid.hex
        dao.store_entity(operation_entity)
        return view_model

    def import_list_of_operations(self, project, import_path, is_group=False, importer_operation_id=None):
        """
        This method scans provided folder and identify all operations that needs to be imported
        """
        all_dts_count = 0
        all_stored_dts_count = 0
        imported_operations = []
        ordered_operations = self._retrieve_operations_in_order(project, import_path,
                                                                None if is_group else importer_operation_id)

        if is_group and len(ordered_operations) > 0:
            first_op = dao.get_operation_by_id(importer_operation_id)
            vm_path = h5.determine_filepath(first_op.view_model_gid, os.path.dirname(import_path))
            os.remove(vm_path)

            ordered_operations[0].operation.id = importer_operation_id

        for operation_data in ordered_operations:
            if operation_data.is_old_form:
                operation_entity, datatype_group = self.import_operation(operation_data.operation)
                new_op_folder = self.storage_interface.get_project_folder(project.name, str(operation_entity.id))

                try:
                    operation_datatypes = self._load_datatypes_from_operation_folder(operation_data.operation_folder,
                                                                                     operation_entity, datatype_group)
                    # Create and store view_model from operation
                    self.create_view_model(operation_entity, operation_data, new_op_folder)

                    self._store_imported_datatypes_in_db(project, operation_datatypes)
                    imported_operations.append(operation_entity)
                except MissingReferenceException:
                    operation_entity.status = STATUS_ERROR
                    dao.store_entity(operation_entity)

            elif operation_data.main_view_model is not None:
                operation_data.operation.create_date = datetime.now()
                operation_data.operation.start_date = datetime.now()
                operation_data.operation.completion_date = datetime.now()

                do_merge = False
                if importer_operation_id:
                    do_merge = True
                operation_entity = dao.store_entity(operation_data.operation, merge=do_merge)
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
                all_dts_count += len(operation_data.dt_paths)
                for dt_path in operation_data.dt_paths:
                    dt = self.load_datatype_from_file(dt_path, operation_entity.id, dt_group, project.id)
                    if isinstance(dt, BurstConfiguration):
                        if op_group:
                            dt.fk_operation_group = op_group.id
                        all_stored_dts_count += self._store_or_link_burst_config(dt, dt_path)
                    else:
                        dts[dt_path] = dt
                        if op_group:
                            op_group.fill_operationgroup_name(dt.type)
                            dao.store_entity(op_group)
                try:
                    stored_dts_count = self._store_imported_datatypes_in_db(project, dts)
                    all_stored_dts_count += stored_dts_count

                    if operation_data.main_view_model.is_metric_operation:
                        self._update_burst_metric(operation_entity)

                    imported_operations.append(operation_entity)
                    new_op_folder = self.storage_interface.get_project_folder(project.name,
                                                                              str(operation_entity.id))
                    view_model_disk_size = 0
                    for h5_file in operation_data.all_view_model_files:
                        view_model_disk_size += StorageInterface.compute_size_on_disk(h5_file)
                        shutil.move(h5_file, new_op_folder)
                    operation_entity.view_model_disk_size = view_model_disk_size
                    dao.store_entity(operation_entity)
                except MissingReferenceException as excep:
                    self.storage_interface.remove_operation_data(project.name, operation_entity.id)
                    operation_entity.fk_operation_group = None
                    dao.store_entity(operation_entity)
                    dao.remove_entity(DataTypeGroup, dt_group.id)
                    raise excep
            else:
                self.logger.warning("Folder %s will be ignored, as we could not find a serialized "
                                    "operation or DTs inside!" % operation_data.operation_folder)

            # We want importer_operation_id to be kept just for the first operation (the first iteration)
            if is_group:
                importer_operation_id = None

        self._update_dt_groups(project.id)
        self._update_burst_configurations(project.id)
        return imported_operations, all_dts_count, all_stored_dts_count

    @staticmethod
    def _get_new_form_view_model(operation, xml_parameters):
        # type (Operation) -> ViewModel
        algo = dao.get_algorithm_by_id(operation.fk_from_algo)
        ad = ABCAdapter.build_adapter(algo)
        view_model = ad.get_view_model_class()()

        if xml_parameters:
            declarative_attrs = type(view_model).declarative_attrs

            if isinstance(xml_parameters, str):
                xml_parameters = json.loads(xml_parameters)
            for param in xml_parameters:
                new_param_name = param
                if param != '' and param[0] == "_":
                    new_param_name = param[1:]
                new_param_name = new_param_name.lower()
                if new_param_name in declarative_attrs:
                    try:
                        setattr(view_model, new_param_name, xml_parameters[param])
                    except (TraitTypeError, TraitAttributeError):
                        pass
        return view_model

    def _import_image(self, src_folder, metadata_file, project_id, target_images_path):
        """
        Create and store a image entity.
        """
        figure_dict = StorageInterface().read_metadata_from_xml(os.path.join(src_folder, metadata_file))
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
        _, meta_data = figure.to_dict()
        self.storage_interface.write_image_metadata(figure, meta_data)

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

            already_existing_datatype = h5.load_entity_by_gid(datatype.gid)
            if datatype_group is not None and already_existing_datatype is not None:
                raise DatatypeGroupImportException("The datatype group that you are trying to import"
                                                   " already exists!")
            index_class = h5.REGISTRY.get_index_for_datatype(datatype.__class__)
            datatype_index = index_class()
            datatype_index.fill_from_has_traits(datatype)
            datatype_index.fill_from_generic_attributes(generic_attributes)

            if datatype_group is not None and hasattr(datatype_index, 'fk_source_gid') and \
                    datatype_index.fk_source_gid is not None:
                ts = h5.load_entity_by_gid(datatype_index.fk_source_gid)

                if ts is None:
                    op = dao.get_operations_in_group(datatype_group.fk_operation_group, only_first_operation=True)
                    op.fk_operation_group = None
                    dao.store_entity(op)
                    dao.remove_entity(OperationGroup, datatype_group.fk_operation_group)
                    dao.remove_entity(DataTypeGroup, datatype_group.id)
                    raise DatatypeGroupImportException("Please import the time series group before importing the"
                                                       " datatype measure group!")

            # Add all the required attributes
            if datatype_group:
                datatype_index.fk_datatype_group = datatype_group.id
                if len(datatype_group.subject) == 0:
                    datatype_group.subject = datatype_index.subject
                    dao.store_entity(datatype_group)
            datatype_index.fk_from_operation = op_id

            associated_file = h5.path_for_stored_index(datatype_index)
            if os.path.exists(associated_file):
                datatype_index.disk_size = StorageInterface.compute_size_on_disk(associated_file)
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
            stored_entry = load.load_entity_by_gid(datatype.gid)
            if not stored_entry:
                stored_entry = dao.store_entity(datatype)

            return stored_entry
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
        project_dict = self.storage_interface.read_project_metadata(project_path)

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

    def build_operation_from_file(self, project, operation_file):
        """
        Create Operation entity from metadata file.
        """
        operation_dict = StorageInterface().read_metadata_from_xml(operation_file)
        operation_entity = manager_of_class(Operation).new_instance()
        return operation_entity.from_dict(operation_dict, dao, self.user_id, project.gid)

    @staticmethod
    def import_operation(operation_entity, migration=False):
        """
        Store a Operation entity.
        """
        do_merge = False
        if operation_entity.id:
            do_merge = True
        operation_entity = dao.store_entity(operation_entity, merge=do_merge)
        operation_group_id = operation_entity.fk_operation_group
        datatype_group = None

        if operation_group_id is not None:
            datatype_group = dao.get_datatypegroup_by_op_group_id(operation_group_id)

            if datatype_group is None and migration is False:
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

        if isinstance(zip_file, (FieldStorage, Part)):
            if not zip_file.file:
                raise ServicesBaseException("Could not process the given ZIP file...")

            with open(uq_file_name, 'wb') as file_obj:
                self.storage_interface.copy_file(zip_file.file, file_obj)
        else:
            shutil.copy2(zip_file, uq_file_name)

        try:
            self.storage_interface.unpack_zip(uq_file_name, temp_folder)
            return temp_folder
        except FileStructureException as excep:
            raise ServicesBaseException("Could not process the given ZIP file..." + str(excep))

    @staticmethod
    def _update_burst_metric(operation_entity):
        burst_config = dao.get_burst_for_operation_id(operation_entity.id)
        if burst_config and burst_config.ranges:
            if burst_config.fk_metric_operation_group is None:
                burst_config.fk_metric_operation_group = operation_entity.fk_operation_group
            dao.store_entity(burst_config)

    @staticmethod
    def _update_dt_groups(project_id):
        dt_groups = dao.get_datatypegroup_for_project(project_id)
        for dt_group in dt_groups:
            dt_group.count_results = dao.count_datatypes_in_group(dt_group.id)
            dts_in_group = dao.get_datatypes_from_datatype_group(dt_group.id)
            if dts_in_group:
                dt_group.fk_parent_burst = dts_in_group[0].fk_parent_burst
            dao.store_entity(dt_group)

    @staticmethod
    def _update_burst_configurations(project_id):
        burst_configs = dao.get_bursts_for_project(project_id)
        for burst_config in burst_configs:
            burst_config.datatypes_number = dao.count_datatypes_in_burst(burst_config.gid)
            dao.store_entity(burst_config)
