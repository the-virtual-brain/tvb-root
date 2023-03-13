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
DAO operations related to generic DataTypes are defined here.
 
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
import importlib

from sqlalchemy import func, or_, and_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import aliased
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.sql import text
from sqlalchemy.sql.expression import desc, cast
from sqlalchemy.types import Text
from tvb.core.entities.model.model_datatype import *
from tvb.core.entities.model.model_operation import Operation, AlgorithmCategory, Algorithm, OperationGroup
from tvb.core.entities.storage.root_dao import RootDAO, DEFAULT_PAGE_SIZE
from tvb.core.neotraits.db import Base


class DatatypeDAO(RootDAO):
    """
    DATATYPE and DATA_TYPES_GROUPS RELATED METHODS
    """

    def get_datatypegroup_for_project(self, project_id):
        """
        Returns all DataTypeGroup entities from a project.
        """
        query = self.session.query(DataTypeGroup
                                   ).join(OperationGroup, OperationGroup.id == DataTypeGroup.fk_operation_group
                                          ).filter(OperationGroup.fk_launched_in == project_id).order_by(DataTypeGroup.id)
        return query.all()

    def get_datatypegroup_by_op_group_id(self, operation_group_id):
        """
        Returns the DataTypeGroup corresponding to a certain OperationGroup.
        """
        try:
            result = self.session.query(DataTypeGroup).filter_by(fk_operation_group=operation_group_id).one()
            return result
        except NoResultFound:
            self.logger.debug("No operation group was found for operation group id = {}".format(operation_group_id))
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return None

    def get_datatype_group_by_gid(self, datatype_group_gid):
        """
        Returns the DataTypeGroup with the specified gid.
        """
        try:
            result = self.session.query(DataTypeGroup).filter_by(gid=datatype_group_gid).one()
            result.parent_operation_group
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def is_datatype_group(self, datatype_gid):
        """
        Used to check if the DataType with the specified GID is a DataTypeGroup.
        """
        try:
            result = self.session.query(DataType
                                        ).filter(DataType.gid == datatype_gid
                                                 ).filter(DataType.id == DataTypeGroup.id).count()
        except SQLAlchemyError:
            return False
        return result > 0


    def count_datatypes_in_group(self, datatype_group_id):
        """
        Returns the number of DataTypes from the specified DataTypeGroup ID.
        """
        try:
            result = self.session.query(DataType
                                        ).filter(DataType.fk_datatype_group == datatype_group_id
                                                 ).filter(DataType.type != self.EXCEPTION_DATATYPE_SIMULATION
                                                          ).count()
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def count_datatypes_in_burst(self, burst_gid):
        """
        Returns the number of DataTypes from the specified BurstConfiguration ID.
        """
        try:
            result = self.session.query(DataType
                                        ).filter(DataType.fk_parent_burst == burst_gid
                                                 ).filter(DataType.type != self.EXCEPTION_DATATYPE_SIMULATION
                                                          ).count()
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None

    def get_number_of_bursts(self, project_id):
        try:
            bursts = self.session.query(BurstConfiguration).filter_by(fk_project=project_id)
            return bursts.count()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None

    def get_disk_size_for_operation(self, operation_id):
        """
        Return the disk size for the operation by summing over the disk space of the resulting DataTypes.
        """
        try:
            disk_size = self.session.query(func.sum(DataType.disk_size)
                                           ).filter(DataType.fk_from_operation == operation_id).scalar() or 0
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            disk_size = 0
        return disk_size


    def get_summary_for_group(self, datatype_group_id):
        """
        :return (disk_size SUM, subject)
        """
        result = 0, ""
        try:
            result = self.session.query(func.sum(DataType.disk_size), func.max(DataType.subject)
                                        ).filter(DataType.fk_datatype_group == datatype_group_id).all()[0] or result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)

        return result


    def compute_bursts_disk_size(self, burst_gids):
        """
        SUM the disk_size of all data types generated by each requested burst
        Do not count DataType Groups as those already include the size of the entities inside the group.
        :returns a map from burst id to disk size
        """
        # do not execute a query that will return []
        if not burst_gids:
            return {}
        # The query might return less results than burst_gids.
        # This happens if a burst entity has not been persisted yet.
        # For those bursts the size will be zero
        ret = {b_gid: 0 for b_gid in burst_gids}
        try:
            query = self.session.query(DataType.fk_parent_burst, func.sum(DataType.disk_size)
                        ).group_by(DataType.fk_parent_burst
                        ).filter(DataType.type != "DataTypeGroup"
                        ).filter(DataType.fk_parent_burst.in_(burst_gids))
            for b_gid, size in query.all():
                ret[b_gid] = size or 0
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return ret

    #
    # DATA_TYPE RELATED METHODS
    #

    def get_datatypes_from_datatype_group(self, datatype_group_id):
        """Retrieve all datatype which are part from the given datatype group."""
        try:
            result = self.session.query(DataType).filter_by(
                fk_datatype_group=datatype_group_id).order_by(DataType.id).all()
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def set_datatype_visibility(self, datatype_gid, is_visible):
        """
        Sets the dataType visibility. If the given dataType is a dataTypeGroup or it is part of a
        dataType group than this method will set the visibility for each dataType from this group.
        """
        datatype = self.get_datatype_by_gid(datatype_gid)
        try:
            self.session.query(DataType).filter(or_(DataType.fk_datatype_group == datatype.id,
                                                          DataType.gid == datatype_gid)
                                                      ).update({"visible": is_visible})
            self.session.commit()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)


    def count_all_datatypes(self):
        """
        Gives you the count of all the datatypes currently stored by TVB. Is used by 
        the file storage update manager to upgrade from version to the next.
        """
        try:
            count = self.session.query(DataType).count()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            count = 0
        return count


    def get_all_datatypes(self, page_start=0, page_size=DEFAULT_PAGE_SIZE):
        """
        Return a list with all of the datatypes currently available in TVB. Is used by 
        the file storage update manager to upgrade from version to the next.
        
        :param page_start: the index from which to start adding datatypes to the result list
        :param page_size: maximum number of entities to retrieve
        """
        resulted_data = []
        try:
            resulted_data = self.session.query(DataType).order_by(DataType.id).offset(
                max(page_start, 0)).limit(max(page_size, 0)).all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return resulted_data


    def get_linked_datatypes_in_project(self, project_id):
        """
        Return a list of datatypes linked into this project
        :param project_id: the id of the project
        """
        datatypes = []
        try:
            query = self.session.query(DataType).join(Links).filter(Links.fk_to_project == project_id)
            for dt in query.all():
                if dt.type == DataTypeGroup.__name__:
                    datatypes.extend(self.get_datatypes_from_datatype_group(dt.id))
                else:
                    datatypes.append(dt)
        except SQLAlchemyError as excep:
            self.logger.exception(excep)

        return datatypes


    def get_datatypes_in_project(self, project_id, only_visible=False):
        """
        Get all the DataTypes for a given project with no other filter apart from the projectId
        """
        query = self.session.query(DataType
                    ).join(Operation, Operation.id == DataType.fk_from_operation
                    ).filter(Operation.fk_launched_in == project_id).order_by(DataType.id)

        if only_visible:
            query = query.filter(DataType.visible == True)

        return query.all()


    def get_data_in_project(self, project_id, visibility_filter=None, filter_value=None):
        """
        Get all the DataTypes for a given project, including Linked Entities and DataType Groups.

        :param project_id: ID for the current project to filter for
        :param visibility_filter: when not None, will filter by DataTye fields
        :param filter_value: when not None, will filter with ilike multiple DataType string attributes
        """
        resulted_data = []
        try:
            # First Query DT, DT_gr, Lk_DT and Lk_DT_gr
            datatype_alias = aliased(DataType, flat=True)
            query = self.session.query(datatype_alias
                        ).join(Operation, Operation.id == datatype_alias.fk_from_operation
                        ).join(Algorithm).join(AlgorithmCategory
                        ).outerjoin(Links, and_(Links.fk_from_datatype == datatype_alias.id,
                                                Links.fk_to_project == project_id)
                        ).outerjoin(BurstConfiguration, datatype_alias.fk_parent_burst == BurstConfiguration.gid
                                    ).filter(datatype_alias.fk_datatype_group == None
                        ).filter(or_(Operation.fk_launched_in == project_id,
                                     Links.fk_to_project == project_id))

            if visibility_filter:
                filter_str = visibility_filter.get_sql_filter_equivalent('datatype_alias')
                if filter_str is not None:
                    query = query.filter(eval(filter_str))
            if filter_value is not None:
                query = query.filter(self._compose_filter_datatype_ilike(filter_value))

            resulted_data = query.all()

            # Now query what it was not covered before:
            # Links of DT which are part of a group, but the entire group is not linked
            links = aliased(Links)
            query2 = self.session.query(datatype_alias
                        ).join(Operation, Operation.id == datatype_alias.fk_from_operation
                        ).join(Algorithm).join(AlgorithmCategory
                        ).join(Links, and_(Links.fk_from_datatype == datatype_alias.id,
                                           Links.fk_to_project == project_id)
                        ).outerjoin(links, and_(links.fk_from_datatype == datatype_alias.fk_datatype_group,
                                                links.fk_to_project == project_id)
                        ).outerjoin(BurstConfiguration, datatype_alias.fk_parent_burst == BurstConfiguration.gid
                                    ).filter(datatype_alias.fk_datatype_group != None
                        ).filter(links.id == None)

            if visibility_filter:
                filter_str = visibility_filter.get_sql_filter_equivalent('datatype_alias')
                if filter_str is not None:
                    query2 = query2.filter(eval(filter_str))
            if filter_value is not None:
                query2 = query2.filter(self._compose_filter_datatype_ilike(filter_value))

            resulted_data.extend(query2.all())

            # Load lazy fields for future usage
            for dt in resulted_data:
                dt._parent_burst
                dt.parent_operation.algorithm
                dt.parent_operation.algorithm.algorithm_category
                dt.parent_operation.project
                dt.parent_operation.operation_group
                dt.parent_operation.user
                dt.display_name

        except Exception as excep:
            self.logger.exception(excep)

        return resulted_data


    def _compose_filter_datatype_ilike(self, filter_string):
        """
        :param filter_string: String to be search for with ilike.
        :return: SqlAlchemy filtering clause
        """
        return or_(cast(DataType.id, Text).like('%' + filter_string + '%'),
                   DataType.gid.like('%' + filter_string + '%'),
                   DataType.type.ilike('%' + filter_string + '%'),
                   DataType.subject.ilike('%' + filter_string + '%'),
                   DataType.state.ilike('%' + filter_string + '%'),
                   DataType.user_tag_1.ilike('%' + filter_string + '%'),
                   DataType.user_tag_2.ilike('%' + filter_string + '%'),
                   DataType.user_tag_3.ilike('%' + filter_string + '%'),
                   DataType.user_tag_4.ilike('%' + filter_string + '%'),
                   DataType.user_tag_5.ilike('%' + filter_string + '%'),
                   Operation.user_group.ilike('%' + filter_string + '%'),
                   AlgorithmCategory.displayname.ilike('%' + filter_string + '%'),
                   Algorithm.displayname.ilike('%' + filter_string + '%'),
                   BurstConfiguration.name.ilike('%' + filter_string + '%'))


    def get_datatype_details(self, datatype_gid):
        """
        Returns the details for the dataType with the given GID.
        """
        result_dt = self.get_datatype_by_gid(datatype_gid)

        if isinstance(result_dt, DataTypeGroup) and result_dt.count_results is None:
            result_dt.count_results = self.count_datatypes_in_group(result_dt.id)
            self.store_entity(result_dt)
            result_dt = self.get_datatype_by_gid(datatype_gid)

        return result_dt


    def get_datatype_extra_info(self, datatype_gid):
        datatype = self.get_datatype_by_gid(datatype_gid, False, True)
        return datatype.get_extra_info()

    def get_datatype_by_gid(self, gid, load_lazy=True, load_lazy_extra_info=False):
        """
        Retrieve a DataType DB reference by a global identifier.
        """
        datatype_instance = None
        try:
            datatype_instance = self.session.query(DataType).filter_by(gid=gid).one()
            classname = datatype_instance.type
            module = importlib.import_module(datatype_instance.module)
            data_type = getattr(module, classname)
            result_dt = self.session.query(data_type).filter_by(gid=gid).one()

            result_dt.parent_operation.project
            if load_lazy:
                result_dt.parent_operation.user
                result_dt.parent_operation.algorithm.algorithm_category
                result_dt.parent_operation.operation_group
                result_dt._parent_burst
            if load_lazy_extra_info:
                result_dt.get_extra_info()

            return result_dt
        except NoResultFound as excep:
            self.logger.debug("No results found for gid=%s" % (gid,))
        except Exception as excep:
            self.logger.warning(datatype_instance)
            self.logger.exception(excep)
        return None


    def get_links_for_datatype(self, data_id):
        """Get the links to a specific datatype"""
        try:
            links = self.session.query(Links).join(DataType).filter(DataType.id == data_id).all()
            return links
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def get_datatype_in_group(self, datatype_group_id=None, operation_group_id=None):
        """
        Return a list of id-s of the DataTypes in the given dt group.
        """
        try:
            resulted_data = []
            result = self.session.query(DataType).join(Operation
                                        ).filter(DataType.type != self.EXCEPTION_DATATYPE_GROUP)
            if datatype_group_id is not None:
                result = result.filter(DataType.fk_datatype_group == datatype_group_id)
            if operation_group_id is not None:
                result = result.filter(Operation.fk_operation_group == operation_group_id)
            result = result.all()
            [data.parent_operation.project for data in result]
            for row in result:
                resulted_data.append(row)
            return resulted_data
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def get_last_data_with_uid(self, uid, datatype_class=DataType):
        """Retrieve the last dataType ID  witch has UDI field as 
        the passed parameter, or None if nothing found."""
        try:
            resulted_data = None
            result = self.session.query(datatype_class.gid
                                        ).filter(datatype_class.user_tag_1 == uid
                                                 ).order_by(desc(datatype_class.id)).all()
            if result is not None and len(result) > 0:
                resulted_data = result[0][0]
            return resulted_data
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def count_datatypes(self, project_id, datatype_class):

        query = self.session.query(datatype_class
                    ).join(Operation, datatype_class.fk_from_operation == Operation.id
                    ).outerjoin(Links
                    ).filter(or_(Operation.fk_launched_in == project_id,
                                 Links.fk_to_project == project_id))
        return query.count()


    def try_load_last_entity_of_type(self, project_id, datatype_class):

        query = self.session.query(datatype_class
                    ).join(Operation, datatype_class.fk_from_operation == Operation.id
                    ).outerjoin(Links
                    ).filter(or_(Operation.fk_launched_in == project_id,
                                 Links.fk_to_project == project_id))
        query = query.order_by(desc(datatype_class.id)).limit(1)
        result = query.all()

        if result is not None and len(result):
            return result[0]
        return None

    def get_values_of_datatype(self, project_id, datatype_class, filters=None, page_size=50):
        """
        Retrieve a list of dataTypes matching a filter inside a project.
        :returns: (results, total_count) maximum page_end rows are returned, to avoid endless time when loading a page
        """
        result = []
        count = 0

        if not issubclass(datatype_class, Base):
            self.logger.warning("Trying to filter not DB class:" + str(datatype_class))
            return result, count

        try:
            # Prepare generic query:
            datatype_alias = aliased(datatype_class, flat=True)
            query = self.session.query(datatype_alias.id,
                                       func.max(datatype_alias.type),
                                       func.max(datatype_alias.gid),
                                       func.max(datatype_alias.subject),
                                       func.max(Operation.completion_date),
                                       func.max(Operation.user_group),
                                       func.max(OperationGroup.name),
                                       func.max(datatype_alias.user_tag_1)
                        ).join(Operation, datatype_alias.fk_from_operation == Operation.id
                        ).outerjoin(BurstConfiguration, datatype_alias.fk_parent_burst == BurstConfiguration.gid
                        ).outerjoin(Links
                        ).outerjoin(OperationGroup, Operation.fk_operation_group == OperationGroup.id
                        ).filter(datatype_alias.invalid == False
                        ).filter(or_(Operation.fk_launched_in == project_id,
                                     Links.fk_to_project == project_id))

            if filters:
                filter_str = filters.get_sql_filter_equivalent(datatype_to_check='datatype_alias')
                if filter_str is not None:
                    query = query.filter(eval(filter_str))

            # Retrieve the results
            query = query.group_by(datatype_alias.id).order_by(desc(datatype_alias.id))

            result = query.limit(max(page_size, 0)).all()
            count = query.count()
        except Exception as excep:
            self.logger.exception(excep)

        return result, count


    def get_datatypes_for_range(self, op_group_id, range_json):
        """Retrieve from DB, DataTypes resulted after executing a specific range operation."""
        data = self.session.query(DataType).join(Operation
                                  ).filter(Operation.fk_operation_group == op_group_id
                                  ).filter(Operation.range_values == range_json
                                  ).filter(DataType.invalid == False
                                  ).order_by(DataType.id).all()
        return data


    def get_datatype_group_disk_size(self, dt_group_id):
        """
        Return the size of all the DataTypes from this datatype group.
        """
        try:
            hdd_size = self.session.query(func.sum(DataType.disk_size)
                                          ).filter(DataType.fk_datatype_group == dt_group_id).scalar() or 0
        except SQLAlchemyError as ex:
            self.logger.exception(ex)
            hdd_size = 0
        return hdd_size

    def get_datatype_measure_group_from_ts_from_pse(self, ts_gid, datatype_measure_class):
        """
        Having a Time Series that was part of a pse, this method should return the datatype group of a datatype measure
        that was part of the same pse
        """
        try:
            datatype_measures = self.session.query(datatype_measure_class).filter_by(fk_source_gid=ts_gid).all()
            datatype_group = None
            for dt_measure in datatype_measures:
                if dt_measure.fk_datatype_group is not None:
                    datatype_group = self.get_datatypegroup_by_op_group_id(
                        dt_measure.parent_operation.fk_operation_group)
                    break

        except SQLAlchemyError as ex:
            self.logger.exception(ex)
            datatype_group = None
        return datatype_group


    ##########################################################################
    ############ Below are specifics for MeasurePoint selections #############
    ##########################################################################

    def get_selections_for_project(self, project_id, datatype_gid, filter_ui_name=None):
        """
        Get available selections for a given project and data type.
        """
        try:
            query = self.session.query(MeasurePointsSelection
                                       ).filter_by(fk_in_project=project_id).filter_by(fk_datatype_gid=datatype_gid)
            if filter_ui_name is not None:
                query = query.filter_by(ui_name=filter_ui_name)
            return query.all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None

    ##########################################################################
    ##########    Bellow are PSE Filters specific methods   ##################
    ##########################################################################

    def get_stored_pse_filters(self, datatype_group_gid, filter_ui_name=None):
        """
        :return: Stored PSE filters for a given DatTypeGroup or None
        """
        try:
            query = self.session.query(StoredPSEFilter).filter_by(fk_datatype_gid=datatype_group_gid)
            if filter_ui_name is not None:
                query = query.filter_by(ui_name=filter_ui_name)
            return query.all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None
