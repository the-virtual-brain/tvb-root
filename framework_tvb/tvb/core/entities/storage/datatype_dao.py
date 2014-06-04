# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
DAO operations related to generic DataTypes are defined here.
 
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from sqlalchemy import func as func
from sqlalchemy import or_, not_, and_, Integer
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text
from sqlalchemy.sql.expression import desc, cast
from sqlalchemy.sql.expression import case as case_
from sqlalchemy.sql.expression import literal_column as literal_
from sqlalchemy.types import Text
from sqlalchemy.orm.exc import NoResultFound
from tvb.core.entities import model
from tvb.core.entities.storage.root_dao import RootDAO



class DatatypeDAO(RootDAO):
    """
    DATATYPE and DATA_TYPES_GROUPS RELATED METHODS
    """


    def get_datatypegroup_by_op_group_id(self, operation_group_id):
        """
        Returns the DataTypeGroup corresponding to a certain OperationGroup.
        """
        result = self.session.query(model.DataTypeGroup).filter_by(fk_operation_group=operation_group_id).one()
        return result


    def get_datatype_group_by_gid(self, datatype_group_gid):
        """
        Returns the DataTypeGroup with the specified gid.
        """
        try:
            result = self.session.query(model.DataTypeGroup).filter_by(gid=datatype_group_gid).one()
            return result
        except SQLAlchemyError, excep:
            self.logger.exception(excep)
            return None


    def get_group_by_op_group_id(self, op_group_id):
        """
        Returns the DataTypeGroup with the specified operation group id.
        """
        try:
            result = self.session.query(model.DataTypeGroup).filter_by(fk_operation_group=op_group_id).one()
            return result
        except SQLAlchemyError, excep:
            self.logger.exception(excep)
            return None


    def is_datatype_group(self, datatype_gid):
        """
        Used to check if the DataType with the specified GID is a DataTypeGroup.
        """
        try:
            result = self.session.query(model.DataType
                                        ).filter(model.DataType.gid == datatype_gid
                                                 ).filter(model.DataType.id == model.DataTypeGroup.id).count()
        except SQLAlchemyError:
            return False
        return result > 0


    def count_datatypes_in_group(self, datatype_group_id):
        """
        Returns the number of DataTypes from the specified DataTypeGroup ID.
        """
        try:
            result = self.session.query(model.DataType
                                        ).filter(model.DataType.fk_datatype_group == datatype_group_id
                                                 ).filter(model.DataType.type != self.EXCEPTION_DATATYPE_SIMULATION
                                                          ).count()
            return result
        except SQLAlchemyError, excep:
            self.logger.exception(excep)
            return None


    def get_disk_size_for_operation(self, operation_id):
        """
        Return the disk size for the operation by summing over the disk space of the resulting DataTypes.
        """
        try:
            disk_size = self.session.query(func.sum(model.DataType.disk_size)
                                           ).filter(model.DataType.fk_from_operation == operation_id).scalar() or 0
        except SQLAlchemyError, excep:
            self.logger.exception(excep)
            disk_size = 0
        return disk_size


    #
    # DATA_TYPE RELATED METHODS
    #

    def get_datatypes_from_datatype_group(self, datatype_group_id):
        """Retrieve all datatype which are part from the given datatype group."""
        try:
            result = self.session.query(model.DataType).filter_by(
                fk_datatype_group=datatype_group_id).order_by(model.DataType.id).all()
            return result
        except SQLAlchemyError, excep:
            self.logger.exception(excep)
            return None


    def set_datatype_visibility(self, datatype_gid, is_visible):
        """
        Sets the dataType visibility. If the given dataType is a dataTypeGroup or it is part of a
        dataType group than this method will set the visibility for each dataType from this group.
        """
        datatype = self.get_datatype_by_gid(datatype_gid)
        try:
            self.session.query(model.DataType).filter(or_(model.DataType.fk_datatype_group == datatype.id,
                                                          model.DataType.gid == datatype_gid)
                                                      ).update({"visible": is_visible})
            self.session.commit()
        except SQLAlchemyError, excep:
            self.logger.exception(excep)


    def count_all_datatypes(self):
        """
        Gives you the count of all the datatypes currently stored by TVB. Is used by 
        the file storage update manager to upgrade from version to the next.
        """
        try:
            count = self.session.query(model.DataType).count()
        except SQLAlchemyError, excep:
            self.logger.exception(excep)
            count = 0
        return count


    def get_all_datatypes(self, page_start=0, page_end=20):
        """
        Return a list with all of the datatypes currently available in TVB. Is used by 
        the file storage update manager to upgrade from version to the next.
        
        :param page_start: the index from which to start adding datatypes to the result list
        :param page_end: the index up until which you add datatypes to the result list
        """
        resulted_data = []
        try:
            resulted_data = self.session.query(model.DataType).offset(max(page_start, 0)).limit(max(page_end, 0)).all()
        except SQLAlchemyError, excep:
            self.logger.exception(excep)
        return resulted_data
    
    
    def count_datatypes_generated_from(self, datatype_gid):
        """
        Returns a count of all the datatypes that were generated by an operation
        having as input the datatype ginen by 'datatype_gid'
        """
        count = 0
        try:
            count = self.session.query(model.DataType).join(model.Operation
                                        ).filter(model.Operation.parameters.ilike('%' + datatype_gid + '%')).count()
        except SQLAlchemyError, excep:
            self.logger.exception(excep)
        return count


    def get_datatypes_for_project(self, project_id, page_start=0, page_end=20, count=False, only_visible=False):
        """
        Return a list of datatypes for this project, paginated between page_start and start_end.
        :param project_id: the id of the project for which you want the datatypes count
        :param page_start: the index from which to start adding datatypes to the result list
        :param page_end: the index up until which you add datatypes to the result list
        :param count: boolean. When true, return the count
        :param only_visible: when true, filter and return or count only visible datatypes in project
        """
        resulted_data = []
        try:
            query = self.session.query(model.DataType).join(model.Operation).join(model.Project
                                        ).filter(model.Project.id == project_id)
            if only_visible:
                query = query.filter(model.DataType.visible == True)
            if count:
                return query.count()
            resulted_data = query.order_by(model.DataType.fk_from_operation
                                           ).offset(max(page_start, 0)).limit(max(page_end, 0)).all()
        except SQLAlchemyError, excep:
            self.logger.exception(excep)

        return resulted_data

    def get_linked_datatypes_for_project(self, project_id):
        """
        Return a list of datatypes linked into this project
        :param project_id: the id of the project
        """
        datatypes = {}
        try:
            query = self.session.query(model.DataType).join(model.Links).join(model.Project
                                        ).filter(model.Project.id == project_id)
            for dt in query.all():
                if dt.type == model.DataTypeGroup.__name__:
                    for subdt in self.get_datatypes_from_datatype_group(dt.id):
                        datatypes[subdt.gid] = subdt
                else:
                    datatypes[dt.gid] = dt
        except SQLAlchemyError, excep:
            self.logger.exception(excep)

        return datatypes.values()


    def get_datatypes_info_for_project(self, project_id, visibility_filter=None, filter_value=None):
        """
        Get all the dataTypes for a given project, including linked data.
    
        If filter_value is not None then it will be returned only the
        dataTypes which contains in the $filter_value value in one of the
        following fields: model.DataType.id, model.DataType.type,
        model.DataType.subject,model.DataType.state, model.DataType.gid
        """
        resulted_data = []
        try:
            query = self.session.query(func.max(model.DataType.type),
                                       func.max(model.DataType.state),
                                       func.max(model.DataType.subject),
                                       func.max(model.AlgorithmCategory.displayname),
                                       func.max(model.AlgorithmGroup.displayname),
                                       func.max(model.Algorithm.name),
                                       func.max(model.User.username),
                                       func.max(model.Operation.fk_operation_group),
                                       func.max(model.Operation.user_group),
                                       func.max(model.DataType.gid),
                                       func.max(model.Operation.completion_date),
                                       func.max(model.DataType.id),
                                       func.sum(case_([(model.Links.fk_to_project > 0, literal_('1', Integer))],
                                                      else_=literal_('0', Integer))),
                                       func.max(case_([(model.DataType.invalid, literal_('1', Integer))],
                                                      else_=literal_('0', Integer))),
                                       func.max(model.DataType.fk_datatype_group),
                                       func.max(model.BurstConfiguration.name),
                                       func.max(model.DataType.user_tag_1), func.max(model.DataType.user_tag_2),
                                       func.max(model.DataType.user_tag_3), func.max(model.DataType.user_tag_4),
                                       func.max(model.DataType.user_tag_5),
                                       func.max(case_([(model.DataType.visible, 1)], else_=0))
                        ).join((model.Operation, model.Operation.id == model.DataType.fk_from_operation)
                        ).join((model.User, model.Operation.fk_launched_by == model.User.id)
                        ).join(model.Algorithm).join(model.AlgorithmGroup).join(model.AlgorithmCategory
                        ).outerjoin((model.Links, and_(model.Links.fk_from_datatype == model.DataType.id,
                                                       model.Links.fk_to_project == project_id))
                        ).outerjoin(model.BurstConfiguration,
                                    model.DataType.fk_parent_burst == model.BurstConfiguration.id
                        ).outerjoin(model.DataTypeGroup,
                                    model.DataType.fk_datatype_group == model.DataTypeGroup.id
                        ).filter(or_(model.Operation.fk_launched_in == project_id,
                                     model.Links.fk_to_project == project_id))
            if visibility_filter:
                filter_str = visibility_filter.get_sql_filter_equivalent()
                if filter_str is not None:
                    query = query.filter(eval(filter_str))
            if filter_value is not None:
                query = query.filter(or_(cast(model.DataType.id, Text).like('%' + filter_value + '%'),
                                         model.DataType.type.ilike('%' + filter_value + '%'),
                                         model.DataType.subject.ilike('%' + filter_value + '%'),
                                         model.DataType.state.ilike('%' + filter_value + '%'),
                                         model.DataType.user_tag_1.ilike('%' + filter_value + '%'),
                                         model.DataType.user_tag_2.ilike('%' + filter_value + '%'),
                                         model.DataType.user_tag_3.ilike('%' + filter_value + '%'),
                                         model.DataType.user_tag_4.ilike('%' + filter_value + '%'),
                                         model.DataType.user_tag_5.ilike('%' + filter_value + '%'),
                                         model.Operation.user_group.ilike('%' + filter_value + '%'),
                                         model.AlgorithmCategory.displayname.ilike('%' + filter_value + '%'),
                                         model.AlgorithmGroup.displayname.ilike('%' + filter_value + '%'),
                                         model.Algorithm.name.ilike('%' + filter_value + '%'),
                                         model.BurstConfiguration.name.ilike('%' + filter_value + '%'),
                                         model.DataType.gid.like('%' + filter_value + '%')))
            #            resulted_data = query.group_by(case_([(model.Operation.fk_operation_group > 0,
            #                                                   - model.Operation.fk_operation_group)],
            #                                               else_=model.DataType.id)).all()
            resulted_data = query.group_by(model.DataType.id).all()
        except Exception, excep:
            self.logger.exception(excep)

        return resulted_data


    def get_datatype_details(self, datatype_gid):
        """
        Returns the details for the dataType with the given GID.
        """
        datatype_instance = self.session.query(model.DataType).filter_by(gid=datatype_gid).one()
        classname = datatype_instance.type
        data_class = __import__(datatype_instance.module, globals(), locals(), [classname])
        data_class = eval("data_class." + classname)
        data_type = data_class
        result_dt = self.session.query(data_type).filter_by(gid=datatype_gid).one()

        if isinstance(result_dt, model.DataTypeGroup) and result_dt.count_results is None:
            result_dt.count_results = self.count_datatypes_in_group(result_dt.id)
            self.session.add(result_dt)
            self.session.commit()
            result_dt = self.session.query(data_type).filter_by(gid=datatype_gid).one()

        result_dt.parent_operation.user
        result_dt.parent_operation.algorithm.algo_group.group_category
        result_dt.parent_operation.operation_group

        parent_burst = None
        if result_dt.fk_parent_burst is not None:
            parent_burst = self.get_generic_entity(model.BurstConfiguration, result_dt.fk_parent_burst, "id")[0]

        return result_dt, parent_burst


    def get_datatype_by_gid(self, gid):
        """Retrieve a DataType DB reference by a global identifier."""
        try:
            datatype_instance = self.session.query(model.DataType).filter_by(gid=gid).one()
            classname = datatype_instance.type
            data_class = __import__(datatype_instance.module, globals(), locals(), [classname])
            data_class = eval("data_class." + classname)
            data_type = data_class
            result = self.session.query(data_type).filter_by(gid=gid).one()
            result.parent_operation.project
            return result
        except NoResultFound, excep:
            self.logger.debug("No results found for gid=%s" % (gid,))
        except Exception, excep:
            self.logger.exception(excep)
        return None


    def get_links_for_datatype(self, data_id):
        """Get the links to a specific datatype"""
        try:
            links = self.session.query(model.Links).join(model.DataType).filter(model.DataType.id == data_id).all()
            return links
        except SQLAlchemyError, excep:
            self.logger.exception(excep)
            return None


    def get_datatype_in_group(self, op_group_id):
        """
        Return a list of id-s of the DataTypes in the given operation group.
        """
        try:
            resulted_data = []
            result = self.session.query(model.DataType).join(model.Operation
                                        ).filter(model.Operation.fk_operation_group == op_group_id
                                        ).filter(model.DataType.type != self.EXCEPTION_DATATYPE_GROUP).all()
            [data.parent_operation.project for data in result]
            for row in result:
                resulted_data.append(row)
            return resulted_data
        except SQLAlchemyError, excep:
            self.logger.exception(excep)
            return None


    def get_last_data_with_uid(self, uid, datatype_class=model.DataType):
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
        except SQLAlchemyError, excep:
            self.logger.exception(excep)
            return None


    def get_values_of_datatype(self, project_id, datatype_class, filters=None, page_end=50):
        """
        Retrieve a list of dataTypes matching a filter inside a project.
        :returns: (results, total_count) maximum page_end rows are returned, to avoid endless time when loading a page
        """
        result = []
        count = 0

        if not issubclass(datatype_class, model.Base):
            self.logger.warning("Trying to filter not DB class:" + str(datatype_class))
            return result, count

        try:
            #Prepare generic query:
            query = self.session.query(datatype_class.id,
                                       func.max(datatype_class.type),
                                       func.max(datatype_class.gid),
                                       func.max(datatype_class.subject),
                                       func.max(model.Operation.completion_date),
                                       func.max(model.Operation.user_group),
                                       func.max(text('"OPERATION_GROUPS_1".name')),
                                       func.max(model.DataType.user_tag_1)
                        ).join((model.Operation, datatype_class.fk_from_operation == model.Operation.id)
                        ).outerjoin(model.Links
                        ).outerjoin((model.OperationGroup, model.Operation.fk_operation_group ==
                                     model.OperationGroup.id), aliased=True
                        ).filter(model.DataType.invalid == False
                        ).filter(or_(model.Operation.fk_launched_in == project_id,
                                     model.Links.fk_to_project == project_id))
            if filters:
                filter_str = filters.get_sql_filter_equivalent(datatype_to_check='datatype_class')
                if filter_str is not None:
                    query = query.filter(eval(filter_str))

            #Retrieve the results
            query = query.group_by(datatype_class.id).order_by(desc(datatype_class.id))

            result = query.limit(max(page_end, 0)).all()
            count = query.count()
        except Exception, excep:
            self.logger.exception(excep)

        return result, count


    def get_datatypes_for_range(self, op_group_id, range_json):
        """Retrieve from DB, DataTypes resulted after executing a specific range operation."""
        data = self.session.query(model.DataType).join(model.Operation
                                  ).filter(model.Operation.fk_operation_group == op_group_id
                                  ).filter(model.Operation.range_values == range_json
                                  ).filter(model.DataType.invalid == False
                                  ).order_by(model.DataType.id).all()
        return data


    def get_datatype_group_disk_size(self, dt_group_id):
        """
        Return the size of all the DataTypes from this datatype group.
        """
        try:
            hdd_size = self.session.query(func.sum(model.DataType.disk_size)
                                          ).filter(model.DataType.fk_datatype_group == dt_group_id).scalar() or 0
        except SQLAlchemyError, ex:
            self.logger.exception(ex)
            hdd_size = 0
        return hdd_size


    def get_burst_disk_size(self, burst_id):
        """
        Return the size of all the DataTypes that resulted from this burst.
        """
        try:
            hdd_size = self.session.query(func.sum(model.DataType.disk_size)
                                          ).filter(model.DataType.fk_parent_burst == burst_id).scalar() or 0
        except SQLAlchemyError, ex:
            self.logger.exception(ex)
            hdd_size = 0
        return hdd_size


    ##########################################################################
    ############ Below are specifics for MeasurePoint selections #############
    ##########################################################################

    def get_selections_for_project(self, project_id, datatype_gid, filter_ui_name=None):
        """
        Get available selections for a given project and data type.
        """
        try:
            query = self.session.query(model.MeasurePointsSelection
                                       ).filter_by(fk_in_project=project_id).filter_by(fk_datatype_gid=datatype_gid)
            if filter_ui_name is not None:
                query = query.filter_by(ui_name=filter_ui_name)
            return query.all()
        except SQLAlchemyError, excep:
            self.logger.exception(excep)
            return None
