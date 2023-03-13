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
Service Layer for the Project entity.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import formencode
import os
import uuid

from tvb.basic.logger.builder import get_logger
from tvb.config import DATATYPE_MEASURE_INDEX_MODULE, DATATYPE_MEASURE_INDEX_CLASS
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.inputs_processor import review_operation_inputs_from_adapter
from tvb.core.entities.file.simulator.burst_configuration_h5 import BurstConfigurationH5
from tvb.core.entities.load import get_class_by_name, load_entity_by_gid
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.model.model_datatype import Links, DataType, DataTypeGroup
from tvb.core.entities.model.model_operation import Operation, OperationGroup
from tvb.core.entities.model.model_project import Project
from tvb.core.entities.storage import dao, transactional
from tvb.core.entities.transient.context_overlay import CommonDetails, DataTypeOverlayDetails, OperationOverlayDetails
from tvb.core.entities.transient.structure_entities import StructureNode, DataTypeMetaData
from tvb.core.neocom import h5
from tvb.core.neotraits.h5 import H5File, ViewModelH5
from tvb.core.removers_factory import get_remover
from tvb.core.services.algorithm_service import AlgorithmService
from tvb.core.services.exceptions import RemoveDataTypeException
from tvb.core.services.exceptions import StructureException, ProjectServiceException
from tvb.core.services.user_service import UserService, MEMBERS_PAGE_SIZE
from tvb.core.utils import format_timedelta, format_bytes_human
from tvb.core.utils import string2date, date2string
from tvb.storage.h5.file.exceptions import FileStructureException
from tvb.storage.storage_interface import StorageInterface


def initialize_storage():
    """
    Create Projects storage root folder in case it does not exist.
    """
    try:
        StorageInterface().check_created()
    except FileStructureException:
        # Do nothing, because we do not have any UI to display exception
        logger = get_logger("tvb.core.services.initialize_storage")
        logger.exception("Could not make sure the root folder exists!")


OPERATIONS_PAGE_SIZE = 20
PROJECTS_PAGE_SIZE = 20

MONTH_YEAR_FORMAT = "%B %Y"
DAY_MONTH_YEAR_FORMAT = "%d %B %Y"


class ProjectService:
    """
    Services layer for Project entities.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.storage_interface = StorageInterface()

    def store_project(self, current_user, is_create, selected_id, **data):
        """
        We want to create/update a project entity.
        """
        # Validate Unique Name
        new_name = data["name"]
        if len(new_name) < 1:
            raise ProjectServiceException("Invalid project name!")
        projects_no = dao.count_projects_for_name(new_name, selected_id)
        if projects_no > 0:
            err = {'name': 'Please choose another name, this one is used!'}
            raise formencode.Invalid("Duplicate Name Error", {}, None, error_dict=err)
        started_operations = dao.get_operation_numbers(selected_id)[1]
        if started_operations > 0:
            raise ProjectServiceException("A project can not be renamed while operations are still running!")
        if is_create:
            current_proj = Project(new_name, current_user.id, data["max_operation_size"], data["description"],
                                   data["disable_imports"])
            self.storage_interface.get_project_folder(current_proj.name)
        else:
            try:
                current_proj = dao.get_project_by_id(selected_id)
            except Exception as excep:
                self.logger.exception("An error has occurred!")
                raise ProjectServiceException(str(excep))
            if current_proj.name != new_name:
                self.storage_interface.rename_project(current_proj.name, new_name)
            current_proj.name = new_name
            current_proj.description = data["description"]
            current_proj.disable_imports = data["disable_imports"]
            current_proj.max_operation_size = data['max_operation_size']
        # Commit to make sure we have a valid ID
        current_proj.refresh_update_date()
        _, metadata_proj = current_proj.to_dict()
        self.storage_interface.write_project_metadata(metadata_proj)
        current_proj = dao.store_entity(current_proj)

        # Retrieve, to initialize lazy attributes
        current_proj = dao.get_project_by_id(current_proj.id)
        # Update share settings on current Project entity
        visited_pages = []
        prj_admin = current_proj.administrator.username
        if 'visited_pages' in data and data['visited_pages']:
            visited_pages = data['visited_pages'].split(',')
        for page in visited_pages:
            members = UserService.retrieve_users_except([prj_admin], int(page), MEMBERS_PAGE_SIZE)[0]
            members = [m.id for m in members]
            dao.delete_members_for_project(current_proj.id, members)

        selected_user_ids = data["users"]
        if is_create and current_user.id not in selected_user_ids:
            # Make the project admin also member of the current project
            selected_user_ids.append(current_user.id)
        dao.add_members_to_project(current_proj.id, selected_user_ids)
        # Finish operation
        self.logger.debug("Edit/Save OK for project:" + str(current_proj.id) + ' by user:' + current_user.username)
        return current_proj

    def remove_member_from_project(self, proj_id, user_id):
        """
        remove a user from the list of members of that project
        """
        try:
            dao.delete_members_for_project(proj_id, [user_id])
        except Exception as error:
            self.logger.exception(f'An error has occurred while trying to leave project {proj_id}')
            raise ProjectServiceException(str(error))

    def find_project(self, project_id):
        """
        Simply retrieve Project entity from Database.
        """
        try:
            return dao.get_project_by_id(project_id)
        except Exception as excep:
            self.logger.exception("Given Project ID was not found in DB!")
            raise ProjectServiceException(str(excep))

    def find_project_lazy_by_gid(self, project_gid):
        """
        Simply retrieve Project entity from Database by gid.
        """
        try:
            return dao.get_project_lazy_by_gid(project_gid)
        except Exception as excep:
            self.logger.exception("Given Project GID was not found in DB!")
            raise ProjectServiceException(str(excep))

    @staticmethod
    def count_filtered_operations(project_id, filters=None):
        """Pass to DAO counters for filtered operations"""
        return dao.get_filtered_operations(project_id, filters, is_count=True)

    def retrieve_project_full(self, project_id, applied_filters=None, current_page=1):
        """
        Return a Tuple with Project entity and Operations for current Project.
        :param project_id: Current Project Identifier
        :param applied_filters: Filters to apply on Operations
        :param current_page: Number for current page in operations
        """
        selected_project = self.find_project(project_id)
        total_filtered = self.count_filtered_operations(project_id, applied_filters)
        pages_no = total_filtered // OPERATIONS_PAGE_SIZE + (1 if total_filtered % OPERATIONS_PAGE_SIZE else 0)
        total_ops_nr = self.count_filtered_operations(project_id)

        start_idx = OPERATIONS_PAGE_SIZE * (current_page - 1)
        current_ops = dao.get_filtered_operations(project_id, applied_filters, start_idx, OPERATIONS_PAGE_SIZE)
        if current_ops is None:
            return selected_project, 0, [], 0

        operations = []
        for one_op in current_ops:
            try:
                result = {}
                if one_op[0] != one_op[1]:
                    result["id"] = str(one_op[0]) + "-" + str(one_op[1])
                else:
                    result["id"] = str(one_op[0])
                burst = dao.get_burst_for_operation_id(one_op[0])
                result["burst_name"] = burst.name if burst else '-'
                result["count"] = one_op[2]
                result["gid"] = one_op[13]
                operation_group_id = one_op[3]
                if operation_group_id is not None and operation_group_id:
                    try:
                        operation_group = dao.get_generic_entity(OperationGroup, operation_group_id)[0]
                        result["group"] = operation_group.name
                        result["group"] = result["group"].replace("_", " ")
                        result["operation_group_id"] = operation_group.id
                        datatype_group = dao.get_datatypegroup_by_op_group_id(operation_group_id)
                        result["datatype_group_gid"] = datatype_group.gid if datatype_group is not None else None
                        result["gid"] = operation_group.gid
                        # Filter only viewers for current DataTypeGroup entity:

                        if datatype_group is None:
                            view_groups = None
                        else:
                            view_groups = AlgorithmService().get_visualizers_for_group(datatype_group.gid)
                        result["view_groups"] = view_groups
                    except Exception:
                        self.logger.exception("We will ignore group on entity:" + str(one_op))
                        result["datatype_group_gid"] = None
                else:
                    result['group'] = None
                    result['datatype_group_gid'] = None
                result["algorithm"] = dao.get_algorithm_by_id(one_op[4])
                result["user"] = dao.get_user_by_id(one_op[5])
                if type(one_op[6]) is str:
                    result["create"] = string2date(str(one_op[6]))
                else:
                    result["create"] = one_op[6]
                if type(one_op[7]) is str:
                    result["start"] = string2date(str(one_op[7]))
                else:
                    result["start"] = one_op[7]
                if type(one_op[8]) is str:
                    result["complete"] = string2date(str(one_op[8]))
                else:
                    result["complete"] = one_op[8]

                if result["complete"] is not None and result["start"] is not None:
                    result["duration"] = format_timedelta(result["complete"] - result["start"])
                result["status"] = one_op[9]
                result["additional"] = one_op[10]
                result["visible"] = True if one_op[11] > 0 else False
                result['operation_tag'] = one_op[12]
                if not result['group']:
                    result['results'] = dao.get_results_for_operation(result['id'])
                else:
                    result['results'] = None
                operations.append(result)
            except Exception:
                # We got an exception when processing one Operation Row. We will continue with the rest of the rows.
                self.logger.exception("Could not prepare operation for display:" + str(one_op))
        return selected_project, total_ops_nr, operations, pages_no

    def retrieve_projects_for_user(self, user_id, current_page=1):
        """
        Return a list with all Projects visible for current user.
        """
        start_idx = PROJECTS_PAGE_SIZE * (current_page - 1)
        total = dao.get_projects_for_user(user_id, is_count=True)
        available_projects = dao.get_projects_for_user(user_id, start_idx, PROJECTS_PAGE_SIZE)
        pages_no = total // PROJECTS_PAGE_SIZE + (1 if total % PROJECTS_PAGE_SIZE else 0)
        for prj in available_projects:
            fns, sta, err, canceled, pending = dao.get_operation_numbers(prj.id)
            prj.operations_finished = fns
            prj.operations_started = sta
            prj.operations_error = err
            prj.operations_canceled = canceled
            prj.operations_pending = pending
            prj.disk_size = dao.get_project_disk_size(prj.id)
            prj.disk_size_human = format_bytes_human(prj.disk_size)
        self.logger.debug("Displaying " + str(len(available_projects)) + " projects in UI for user " + str(user_id))
        return available_projects, pages_no

    @staticmethod
    def retrieve_all_user_projects(user_id, page_start=0, page_size=PROJECTS_PAGE_SIZE):
        """
        Return a list with all projects visible for current user, without pagination.
        """
        return dao.get_projects_for_user(user_id, page_start=page_start, page_size=page_size)

    @staticmethod
    def get_linkable_projects_for_user(user_id, data_id):
        """
        Find projects with are visible for current user, and in which current datatype hasn't been linked yet.
        """
        return dao.get_linkable_projects_for_user(user_id, data_id)

    @transactional
    def remove_project(self, project_id):
        """
        Remove Project from DB and File Storage.
        """
        try:
            project2delete = dao.get_project_by_id(project_id)

            self.logger.debug("Deleting project: id=" + str(project_id) + ' name=' + project2delete.name)
            project_datatypes = dao.get_datatypes_in_project(project_id)

            # Delete datatypes one by one in the reversed order of their creation date
            project_datatypes.sort(key=lambda dt: dt.create_date, reverse=True)
            links = []
            for one_data in project_datatypes:
                new_links = self.remove_datatype(project_id, one_data.gid, True, links)
                if new_links is not None:
                    # Keep track of links so we don't create the same link more than once
                    links.extend(new_links)

            self.storage_interface.remove_project(project2delete)
            dao.delete_project(project_id)
            self.logger.debug("Deleted project: id=" + str(project_id) + ' name=' + project2delete.name)

        except RemoveDataTypeException as excep:
            self.logger.exception("Could not execute operation Node Remove!")
            raise ProjectServiceException(str(excep))
        except FileStructureException as excep:
            self.logger.exception("Could not delete because of rights!")
            raise ProjectServiceException(str(excep))
        except Exception as excep:
            self.logger.exception(str(excep))
            raise ProjectServiceException(str(excep))

    # ----------------- Methods for populating Data-Structure Page ---------------

    @staticmethod
    def get_datatype_in_group(group):
        """
        Return all dataTypes that are the result of the same DTgroup.
        """
        return dao.get_datatype_in_group(datatype_group_id=group)

    @staticmethod
    def get_datatypes_from_datatype_group(datatype_group_id):
        """
        Retrieve all dataType which are part from the given dataType group.
        """
        return dao.get_datatypes_from_datatype_group(datatype_group_id)

    @staticmethod
    def load_operation_by_gid(operation_gid):
        """ Retrieve loaded Operation from DB"""
        return dao.get_operation_by_gid(operation_gid)

    @staticmethod
    def load_operation_lazy_by_gid(operation_gid):
        """ Retrieve lazy Operation from DB"""
        return dao.get_operation_lazy_by_gid(operation_gid)

    @staticmethod
    def get_operation_group_by_id(operation_group_id):
        """ Loads OperationGroup from DB"""
        return dao.get_operationgroup_by_id(operation_group_id)

    @staticmethod
    def get_operation_group_by_gid(operation_group_gid):
        """ Loads OperationGroup from DB"""
        return dao.get_operationgroup_by_gid(operation_group_gid)

    @staticmethod
    def get_operations_in_group(operation_group):
        """ Return all the operations from an operation group. """
        return dao.get_operations_in_group(operation_group.id)

    @staticmethod
    def is_upload_operation(operation_gid):
        """ Returns True only if the operation with the given GID is an upload operation. """
        return dao.is_upload_operation(operation_gid)

    @staticmethod
    def get_all_operations_for_uploaders(project_id):
        """ Returns all finished upload operations. """
        return dao.get_all_operations_for_uploaders(project_id)

    def set_operation_and_group_visibility(self, entity_gid, is_visible, is_operation_group=False):
        """
        Sets the operation visibility.

        If 'is_operation_group' is True than this method will change the visibility for all
        the operation from the OperationGroup with the GID field equal to 'entity_gid'.
        """

        def set_visibility(op):
            # workaround:
            # 'reload' the operation so that it has the project property set.
            # get_operations_in_group does not eager load it and now we're out of a sqlalchemy session
            # write_operation_metadata requires that property
            op = dao.get_operation_by_id(op.id)
            # end hack
            op.visible = is_visible
            dao.store_entity(op)

        def set_group_descendants_visibility(operation_group_id):
            ops_in_group = dao.get_operations_in_group(operation_group_id)
            for group_op in ops_in_group:
                set_visibility(group_op)

        if is_operation_group:
            op_group_id = dao.get_operationgroup_by_gid(entity_gid).id
            set_group_descendants_visibility(op_group_id)
        else:
            operation = dao.get_operation_by_gid(entity_gid)
            # we assure that if the operation belongs to a group than the visibility will be changed for the entire group
            if operation.fk_operation_group is not None:
                set_group_descendants_visibility(operation.fk_operation_group)
            else:
                set_visibility(operation)

    def get_operation_details(self, operation_gid, is_group):
        """
        :returns: an entity OperationOverlayDetails filled with all information for current operation details.
        """

        if is_group:
            operation_group = self.get_operation_group_by_gid(operation_gid)
            operation = dao.get_operations_in_group(operation_group.id, False, True)
            # Reload, to make sure all attributes lazy are populated as well.
            operation = dao.get_operation_by_gid(operation.gid)
            no_of_op_in_group = dao.get_operations_in_group(operation_group.id, is_count=True)
            datatype_group = self.get_datatypegroup_by_op_group_id(operation_group.id)
            count_result = dao.count_datatypes_in_group(datatype_group.id)

        else:
            operation = dao.get_operation_by_gid(operation_gid)
            if operation is None:
                return None
            no_of_op_in_group = 1
            count_result = dao.count_resulted_datatypes(operation.id)

        user_display_name = dao.get_user_by_id(operation.fk_launched_by).display_name
        burst = dao.get_burst_for_operation_id(operation.id)
        datatypes_param, all_special_params = self._review_operation_inputs(operation.gid)

        op_pid = dao.get_operation_process_for_operation(operation.id)
        op_details = OperationOverlayDetails(operation, user_display_name, len(datatypes_param),
                                             count_result, burst, no_of_op_in_group, op_pid)

        # Add all parameter which are set differently by the user on this Operation.
        if all_special_params is not None:
            op_details.add_scientific_fields(all_special_params)
        return op_details

    @staticmethod
    def get_filterable_meta():
        """
        Contains all the attributes by which
        the user can structure the tree of DataTypes
        """
        return DataTypeMetaData.get_filterable_meta()

    def get_project_structure(self, project, visibility_filter, first_level, second_level, filter_value):
        """
        Find all DataTypes (including the linked ones and the groups) relevant for the current project.
        In case of a problem, will return an empty list.
        """
        metadata_list = []
        dt_list = dao.get_data_in_project(project.id, visibility_filter, filter_value)

        for dt in dt_list:
            # Prepare the DT results from DB, for usage in controller, by converting into DataTypeMetaData objects
            data = {}
            is_group = False
            group_op = None

            #  Filter by dt.type, otherwise Links to individual DT inside a group will be mistaken
            if dt.type == "DataTypeGroup" and dt.parent_operation.operation_group is not None:
                is_group = True
                group_op = dt.parent_operation.operation_group

            # All these fields are necessary here for dynamic Tree levels.
            data[DataTypeMetaData.KEY_DATATYPE_ID] = dt.id
            data[DataTypeMetaData.KEY_GID] = dt.gid
            data[DataTypeMetaData.KEY_NODE_TYPE] = dt.display_type
            data[DataTypeMetaData.KEY_STATE] = dt.state
            data[DataTypeMetaData.KEY_SUBJECT] = str(dt.subject)
            data[DataTypeMetaData.KEY_TITLE] = dt.display_name
            data[DataTypeMetaData.KEY_RELEVANCY] = dt.visible
            data[DataTypeMetaData.KEY_LINK] = dt.parent_operation.fk_launched_in != project.id

            data[DataTypeMetaData.KEY_TAG_1] = dt.user_tag_1 if dt.user_tag_1 else ''
            data[DataTypeMetaData.KEY_TAG_2] = dt.user_tag_2 if dt.user_tag_2 else ''
            data[DataTypeMetaData.KEY_TAG_3] = dt.user_tag_3 if dt.user_tag_3 else ''
            data[DataTypeMetaData.KEY_TAG_4] = dt.user_tag_4 if dt.user_tag_4 else ''
            data[DataTypeMetaData.KEY_TAG_5] = dt.user_tag_5 if dt.user_tag_5 else ''

            # Operation related fields:
            operation_name = CommonDetails.compute_operation_name(
                dt.parent_operation.algorithm.algorithm_category.displayname,
                dt.parent_operation.algorithm.displayname)
            data[DataTypeMetaData.KEY_OPERATION_TYPE] = operation_name
            data[DataTypeMetaData.KEY_OPERATION_ALGORITHM] = dt.parent_operation.algorithm.displayname
            data[DataTypeMetaData.KEY_AUTHOR] = dt.parent_operation.user.username
            data[DataTypeMetaData.KEY_OPERATION_TAG] = group_op.name if is_group else dt.parent_operation.user_group
            data[DataTypeMetaData.KEY_OP_GROUP_ID] = group_op.id if is_group else None

            completion_date = dt.parent_operation.completion_date
            string_year = completion_date.strftime(MONTH_YEAR_FORMAT) if completion_date is not None else ""
            string_month = completion_date.strftime(DAY_MONTH_YEAR_FORMAT) if completion_date is not None else ""
            data[DataTypeMetaData.KEY_DATE] = date2string(completion_date) if (completion_date is not None) else ''
            data[DataTypeMetaData.KEY_CREATE_DATA_MONTH] = string_year
            data[DataTypeMetaData.KEY_CREATE_DATA_DAY] = string_month

            data[DataTypeMetaData.KEY_BURST] = dt._parent_burst.name if dt._parent_burst is not None else '-None-'

            metadata_list.append(DataTypeMetaData(data, dt.invalid))

        return StructureNode.metadata2tree(metadata_list, first_level, second_level, project.id, project.name)

    @staticmethod
    def get_datatype_details(datatype_gid):
        """
        :returns: an array. First entry in array is an instance of DataTypeOverlayDetails\
            The second one contains all the possible states for the specified dataType.
        """
        meta_atts = DataTypeOverlayDetails()
        states = DataTypeMetaData.STATES
        try:
            datatype_result = dao.get_datatype_details(datatype_gid)
            meta_atts.fill_from_datatype(datatype_result, datatype_result._parent_burst)
            return meta_atts, states, datatype_result
        except Exception:
            # We ignore exception here (it was logged above, and we want to return no details).
            return meta_atts, states, None

    @staticmethod
    def _add_links_for_datatype_references(datatype, fk_to_project, link_to_delete, existing_dt_links):
        # If we found a datatype that has links, we need to link those as well to the linked project
        # so they can be also copied

        linked_datatype_paths = []
        h5_file = h5.h5_file_for_index(datatype)
        h5.gather_all_references_by_index(h5_file, linked_datatype_paths)

        for h5_path in linked_datatype_paths:
            if existing_dt_links is not None and h5_path in existing_dt_links:
                continue

            gid = H5File.get_metadata_param(h5_path, 'gid')
            dt_index = h5.load_entity_by_gid(uuid.UUID(gid))
            new_link = Links(dt_index.id, fk_to_project)
            dao.store_entity(new_link)

        dao.remove_entity(Links, link_to_delete)
        return linked_datatype_paths

    def _remove_project_node_files(self, project_id, gid, links, skip_validation=False):
        """
        Delegate removal of a node in the structure of the project.
        In case of a problem will THROW StructureException.
        """
        try:
            project = self.find_project(project_id)
            datatype = dao.get_datatype_by_gid(gid)

            if links:
                op = dao.get_operation_by_id(datatype.fk_from_operation)
                # Instead of deleting, we copy the datatype to the linked project
                # We also clone the operation
                new_operation = self.__copy_linked_datatype_before_delete(op, datatype, project,
                                                                          links[0].fk_to_project)

                # If there is a  datatype group and operation group and they were not moved yet to the linked project,
                # then do it
                if datatype.fk_datatype_group is not None:
                    dt_group_op = dao.get_operation_by_id(datatype.fk_from_operation)
                    op_group = dao.get_operationgroup_by_id(dt_group_op.fk_operation_group)
                    op_group.fk_launched_in = links[0].fk_to_project
                    dao.store_entity(op_group)

                    burst = dao.get_burst_for_operation_id(op.id)
                    if burst is not None:
                        burst.fk_project = links[0].fk_to_project
                        dao.store_entity(burst)

                    dt_group = dao.get_datatypegroup_by_op_group_id(op_group.id)
                    dt_group.parent_operation = new_operation
                    dt_group.fk_from_operation = new_operation.id
                    dao.store_entity(dt_group)

            else:
                # There is no link for this datatype so it has to be deleted
                specific_remover = get_remover(datatype.type)(datatype)
                specific_remover.remove_datatype(skip_validation)

                # Remove burst if dt has one and it still exists
                if datatype.fk_parent_burst is not None and datatype.is_ts:
                    burst = dao.get_burst_for_operation_id(datatype.fk_from_operation)

                    if burst is not None:
                        dao.remove_entity(BurstConfiguration, burst.id)

        except RemoveDataTypeException:
            self.logger.exception("Could not execute operation Node Remove!")
            raise
        except FileStructureException:
            self.logger.exception("Remove operation failed")
            raise StructureException("Remove operation failed for unknown reasons.Please contact system administrator.")

    def __copy_linked_datatype_before_delete(self, op, datatype, project, fk_to_project):
        new_op = Operation(op.view_model_gid,
                           dao.get_system_user().id,
                           fk_to_project,
                           datatype.parent_operation.fk_from_algo,
                           datatype.parent_operation.status,
                           datatype.parent_operation.start_date,
                           datatype.parent_operation.completion_date,
                           datatype.parent_operation.fk_operation_group,
                           datatype.parent_operation.additional_info,
                           datatype.parent_operation.user_group,
                           datatype.parent_operation.range_values)
        new_op.visible = datatype.parent_operation.visible
        new_op = dao.store_entity(new_op)
        to_project = self.find_project(fk_to_project)
        to_project_path = self.storage_interface.get_project_folder(to_project.name)

        full_path = h5.path_for_stored_index(datatype)
        old_folder = self.storage_interface.get_project_folder(project.name, str(op.id))
        file_paths = h5.gather_references_of_view_model(op.view_model_gid, old_folder, only_view_models=True)[0]
        file_paths.append(full_path)

        # The BurstConfiguration h5 file has to be moved only when we handle the time series which has the operation
        # folder containing the file
        if datatype.is_ts and datatype.fk_parent_burst is not None:
            bc_path = h5.path_for(datatype.parent_operation.id, BurstConfigurationH5, datatype.fk_parent_burst,
                                  project.name)
            if os.path.exists(bc_path):
                file_paths.append(bc_path)

                bc = dao.get_burst_for_operation_id(op.id)
                bc.fk_simulation = new_op.id
                dao.store_entity(bc)

        # Move all files to the new operation folder
        self.storage_interface.move_datatype_with_sync(to_project, to_project_path, new_op.id, file_paths)

        datatype.fk_from_operation = new_op.id
        datatype.parent_operation = new_op
        dao.store_entity(datatype)

        return new_op

    def remove_operation(self, operation_id):
        """
        Remove a given operation
        """
        operation = dao.try_get_operation_by_id(operation_id)
        if operation is not None:
            self.logger.debug("Deleting operation %s " % operation)
            datatypes_for_op = dao.get_results_for_operation(operation_id)
            for dt in reversed(datatypes_for_op):
                self.remove_datatype(operation.project.id, dt.gid, False)
            # Here the Operation is mot probably already removed - in case DTs were found inside
            # but we still remove it for the case when no DTs exist
            dao.remove_entity(Operation, operation.id)
            self.storage_interface.remove_operation_data(operation.project.name, operation_id)
            self.storage_interface.push_folder_to_sync(operation.project.name)
            self.logger.debug("Finished deleting operation %s " % operation)
        else:
            self.logger.warning("Attempt to delete operation with id=%s which no longer exists." % operation_id)

    def remove_datatype(self, project_id, datatype_gid, skip_validation=False, existing_dt_links=None):
        """
        Method used for removing a dataType. If the given dataType is a DatatypeGroup
        or a dataType from a DataTypeGroup than this method will remove the entire group.
        The operation(s) used for creating the dataType(s) will also be removed.
        """
        datatype = dao.get_datatype_by_gid(datatype_gid)
        if datatype is None:
            self.logger.warning("Attempt to delete DT[%s] which no longer exists." % datatype_gid)
            return

        if datatype.parent_operation.fk_launched_in != int(project_id):
            self.logger.warning("Datatype with GUID [%s] has been moved to another project and does "
                                "not need to be deleted anymore." % datatype_gid)
            return

        is_datatype_group = False
        datatype_group = None
        new_dt_links = []

        # Datatype Groups were already handled when the first DatatypeMeasureIndex has been found
        if dao.is_datatype_group(datatype_gid):
            is_datatype_group = True
            datatype_group = datatype
        # Found the first DatatypeMeasureIndex from a group
        elif datatype.fk_datatype_group is not None:
            is_datatype_group = True
            # We load it this way to make sure we have the 'fk_operation_group' in every case
            datatype_group_gid = dao.get_datatype_by_id(datatype.fk_datatype_group).gid
            datatype_group = h5.load_entity_by_gid(datatype_group_gid)

        operations_set = [datatype.fk_from_operation]
        correct = True

        if is_datatype_group:
            operations_set = [datatype_group.fk_from_operation]
            self.logger.debug("Removing datatype group %s" % datatype_group)

            datatypes = self.get_all_datatypes_from_data(datatype_group)
            first_datatype = datatypes[0]

            if hasattr(first_datatype, 'fk_source_gid'):
                ts = h5.load_entity_by_gid(first_datatype.fk_source_gid)
                ts_group = dao.get_datatypegroup_by_op_group_id(ts.parent_operation.fk_operation_group)
                dm_group = datatype_group
            else:
                dt_measure_index = get_class_by_name("{}.{}".format(DATATYPE_MEASURE_INDEX_MODULE,
                                                                    DATATYPE_MEASURE_INDEX_CLASS))
                dm_group = dao.get_datatype_measure_group_from_ts_from_pse(first_datatype.gid, dt_measure_index)
                ts_group = datatype_group

            links = []

            if ts_group:
                links.extend(dao.get_links_for_datatype(ts_group.id))
                correct = correct and self._remove_operation_group(ts_group.fk_operation_group, project_id,
                                                                   skip_validation, operations_set, links)

            if dm_group:
                links.extend(dao.get_links_for_datatype(dm_group.id))
                correct = correct and self._remove_operation_group(dm_group.fk_operation_group, project_id,
                                                                   skip_validation, operations_set, links)

            if len(links) > 0:
                # We want to get the links for the first TSIndex directly
                # This code works for all cases
                datatypes = dao.get_datatype_in_group(ts_group.id)
                ts = datatypes[0]

                new_dt_links = self._add_links_for_datatype_references(ts, links[0].fk_to_project, links[0].id,
                                                                       existing_dt_links)

        else:
            self.logger.debug("Removing datatype %s" % datatype)
            links = dao.get_links_for_datatype(datatype.id)

            if len(links) > 0:
                new_dt_links = self._add_links_for_datatype_references(datatype, links[0].fk_to_project, links[0].id,
                                                                       existing_dt_links)

            self._remove_project_node_files(project_id, datatype.gid, links, skip_validation)

        # Remove Operation entity in case no other DataType needs them.
        project = dao.get_project_by_id(project_id)
        for operation_id in operations_set:
            dependent_dt = dao.get_generic_entity(DataType, operation_id, "fk_from_operation")
            if len(dependent_dt) > 0:
                # Do not remove Operation in case DataType still exist referring it.
                continue
            correct = correct and dao.remove_entity(Operation, operation_id)
            # Make sure Operation folder is removed
            self.storage_interface.remove_operation_data(project.name, operation_id)

        self.storage_interface.push_folder_to_sync(project.name)
        if not correct:
            raise RemoveDataTypeException("Could not remove DataType " + str(datatype_gid))
        return new_dt_links

    def _remove_operation_group(self, operation_group_id, project_id, skip_validation, operations_set, links):
        metrics_groups = dao.get_generic_entity(DataTypeGroup, operation_group_id,
                                                'fk_operation_group')
        if len(metrics_groups) > 0:
            metric_datatype_group_id = metrics_groups[0].id
            self._remove_datatype_group_dts(project_id, metric_datatype_group_id, skip_validation,
                                            operations_set, links)
            datatypes = dao.get_datatype_in_group(metric_datatype_group_id)

            # If this condition is false, then there is a link for the datatype group and we don't need to delete it
            if len(datatypes) == 0:
                dao.remove_entity(DataTypeGroup, metric_datatype_group_id)
                return dao.remove_entity(OperationGroup, operation_group_id)
            else:
                return True

    def _remove_datatype_group_dts(self, project_id, dt_group_id, skip_validation, operations_set, links):
        data_list = dao.get_datatypes_from_datatype_group(dt_group_id)
        for adata in data_list:
            self._remove_project_node_files(project_id, adata.gid, links, skip_validation)
            if adata.fk_from_operation not in operations_set:
                operations_set.append(adata.fk_from_operation)

    def update_metadata(self, submit_data):
        """
        Update DataType/ DataTypeGroup metadata
        THROW StructureException when input data is invalid.
        """
        new_data = dict()
        for key in DataTypeOverlayDetails().meta_attributes_list:
            if key in submit_data:
                value = submit_data[key]
                if value == "None":
                    value = None
                if value == "" and key in [CommonDetails.CODE_OPERATION_TAG, CommonDetails.CODE_OPERATION_GROUP_ID]:
                    value = None
                new_data[key] = value

        try:
            if (CommonDetails.CODE_OPERATION_GROUP_ID in new_data
                    and new_data[CommonDetails.CODE_OPERATION_GROUP_ID]):
                # We need to edit a group
                all_data_in_group = dao.get_datatype_in_group(operation_group_id=
                                                              new_data[CommonDetails.CODE_OPERATION_GROUP_ID])
                if len(all_data_in_group) < 1:
                    raise StructureException("Inconsistent group, can not be updated!")
                # datatype_group = dao.get_generic_entity(DataTypeGroup, all_data_in_group[0].fk_datatype_group)[0]
                # all_data_in_group.append(datatype_group)
                for datatype in all_data_in_group:
                    self._edit_data(datatype, new_data, True)
            else:
                # Get the required DataType and operation from DB to store changes that will be done in XML.
                gid = new_data[CommonDetails.CODE_GID]
                datatype = dao.get_datatype_by_gid(gid)
                self._edit_data(datatype, new_data)
        except Exception as excep:
            self.logger.exception(excep)
            raise StructureException(str(excep))

    def _edit_data(self, datatype, new_data, from_group=False):
        # type: (DataType, dict, bool) -> None
        """
        Private method, used for editing a meta-data XML file and a DataType row
        for a given custom DataType entity with new dictionary of data from UI.
        """
        # 1. First update Operation fields:
        #    Update group field if possible
        new_group_name = new_data[CommonDetails.CODE_OPERATION_TAG]
        empty_group_value = (new_group_name is None or new_group_name == "")
        if from_group:
            if empty_group_value:
                raise StructureException("Empty group is not allowed!")

            group = dao.get_generic_entity(OperationGroup, new_data[CommonDetails.CODE_OPERATION_GROUP_ID])
            if group and len(group) > 0 and new_group_name != group[0].name:
                group = group[0]
                exists_group = dao.get_generic_entity(OperationGroup, new_group_name, 'name')
                if exists_group:
                    raise StructureException("Group '" + new_group_name + "' already exists.")
                group.name = new_group_name
                dao.store_entity(group)
        else:
            operation = dao.get_operation_by_id(datatype.fk_from_operation)
            operation.user_group = new_group_name
            dao.store_entity(operation)
            op_folder = self.storage_interface.get_project_folder(operation.project.name, str(operation.id))
            vm_gid = operation.view_model_gid
            view_model_file = h5.determine_filepath(vm_gid, op_folder)
            if view_model_file:
                view_model_class = H5File.determine_type(view_model_file)
                view_model = view_model_class()
                with ViewModelH5(view_model_file, view_model) as f:
                    ga = f.load_generic_attributes()
                    ga.operation_tag = new_group_name
                    f.store_generic_attributes(ga, False)
            else:
                self.logger.warning("Could not find ViewModel H5 file for op: {}".format(operation))

        # 2. Update GenericAttributes in the associated H5 files:
        h5_path = h5.path_for_stored_index(datatype)
        with H5File.from_file(h5_path) as f:
            ga = f.load_generic_attributes()

            ga.subject = new_data[DataTypeOverlayDetails.DATA_SUBJECT]
            ga.state = new_data[DataTypeOverlayDetails.DATA_STATE]
            ga.operation_tag = new_group_name
            if DataTypeOverlayDetails.DATA_TAG_1 in new_data:
                ga.user_tag_1 = new_data[DataTypeOverlayDetails.DATA_TAG_1]
            if DataTypeOverlayDetails.DATA_TAG_2 in new_data:
                ga.user_tag_2 = new_data[DataTypeOverlayDetails.DATA_TAG_2]
            if DataTypeOverlayDetails.DATA_TAG_3 in new_data:
                ga.user_tag_3 = new_data[DataTypeOverlayDetails.DATA_TAG_3]
            if DataTypeOverlayDetails.DATA_TAG_4 in new_data:
                ga.user_tag_4 = new_data[DataTypeOverlayDetails.DATA_TAG_4]
            if DataTypeOverlayDetails.DATA_TAG_5 in new_data:
                ga.user_tag_5 = new_data[DataTypeOverlayDetails.DATA_TAG_5]

            f.store_generic_attributes(ga, False)

        # 3. Update MetaData in DT Index DB as well.
        datatype.fill_from_generic_attributes(ga)
        dao.store_entity(datatype)

    def _review_operation_inputs(self, operation_gid):
        """
        :returns: A list of DataTypes that are used as input parameters for the specified operation.
                 And a dictionary will all operation parameters different then the default ones.
        """
        operation = dao.get_operation_by_gid(operation_gid)
        try:
            adapter = ABCAdapter.build_adapter(operation.algorithm)
            return review_operation_inputs_from_adapter(adapter, operation)

        except Exception:
            self.logger.exception("Could not load details for operation %s" % operation_gid)
            if operation.view_model_gid:
                changed_parameters = dict(Warning="Algorithm changed dramatically. We can not offer more details")
            else:
                changed_parameters = dict(Warning="GID parameter is missing. Old implementation of the operation.")
            return [], changed_parameters

    @staticmethod
    def get_results_for_operation(operation_id):
        """
        Retrieve the DataTypes entities resulted after the execution of the given operation.
        """
        return dao.get_results_for_operation(operation_id)

    @staticmethod
    def get_datatype_by_id(datatype_id):
        """Retrieve a DataType DB reference by its id."""
        return dao.get_datatype_by_id(datatype_id)

    @staticmethod
    def get_datatypegroup_by_gid(datatypegroup_gid):
        """ Returns the DataTypeGroup with the specified gid. """
        return dao.get_datatype_group_by_gid(datatypegroup_gid)

    @staticmethod
    def get_datatypegroup_by_op_group_id(operation_group_id):
        """ Returns the DataTypeGroup with the specified id. """
        return dao.get_datatypegroup_by_op_group_id(operation_group_id)

    @staticmethod
    def get_datatypes_in_project(project_id):
        return dao.get_data_in_project(project_id)

    @staticmethod
    def set_datatype_visibility(datatype_gid, is_visible):
        """
        Sets the dataType visibility. If the given dataType is a dataType group or it is part of a
        dataType group than this method will set the visibility for each dataType from this group.
        """

        def set_visibility(dt):
            """ set visibility flag, persist in db and h5"""
            dt.visible = is_visible
            dt = dao.store_entity(dt)

            h5_path = h5.path_for_stored_index(dt)
            with H5File.from_file(h5_path) as f:
                f.visible.store(is_visible)

        def set_group_descendants_visibility(datatype_group_id):
            datatypes_in_group = dao.get_datatypes_from_datatype_group(datatype_group_id)
            for group_dt in datatypes_in_group:
                set_visibility(group_dt)

        datatype = dao.get_datatype_by_gid(datatype_gid)

        if isinstance(datatype, DataTypeGroup):  # datatype is a group
            set_group_descendants_visibility(datatype.id)
            datatype.visible = is_visible
            dao.store_entity(datatype)
        elif datatype.fk_datatype_group is not None:  # datatype is member of a group
            set_group_descendants_visibility(datatype.fk_datatype_group)
            # the datatype to be updated is the parent datatype group
            parent = dao.get_datatype_by_id(datatype.fk_datatype_group)
            parent.visible = is_visible
            dao.store_entity(parent)
        else:
            # update the single datatype.
            set_visibility(datatype)

    @staticmethod
    def is_datatype_group(datatype_gid):
        """ Used to check if the dataType with the specified GID is a DataTypeGroup. """
        return dao.is_datatype_group(datatype_gid)

    def get_linked_datatypes_storage_path(self, project):
        """
        :return: the file paths to the datatypes that are linked in `project`
        """
        paths = []
        for lnk_dt in dao.get_linked_datatypes_in_project(project.id):
            # get datatype as a mapped type
            path = h5.path_for_stored_index(lnk_dt)
            if path is not None:
                paths.append(path)
            else:
                self.logger.warning("Problem when trying to retrieve path on %s:%s!" % (lnk_dt.type, lnk_dt.gid))
        return paths

    @staticmethod
    def get_all_datatypes_from_data(data):
        """
        This method builds an array with all data types to be processed later.
        - If current data is a simple data type is added to an array.
        - If it is an data type group all its children are loaded and added to array.
        """
        # first check if current data is a DataTypeGroup
        if DataTypeGroup.is_data_a_group(data):
            data_types = ProjectService.get_datatypes_from_datatype_group(data.id)

            result = []
            if data_types is not None and len(data_types) > 0:
                for data_type in data_types:
                    entity = load_entity_by_gid(data_type.gid)
                    result.append(entity)

            return result

        else:
            return [data]
