# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
DAO operations related to Algorithms and User Operations are defined here.
 
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from sqlalchemy import or_, and_
from sqlalchemy import func as func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.sql.expression import case as case_, desc
from tvb.core.entities import model
from tvb.core.entities.storage.root_dao import RootDAO


class OperationDAO(RootDAO):
    """
    OPERATION RELATED METHODS
    """


    def get_operation_by_id(self, operation_id):
        """Retrieve OPERATION entity for a given Identifier."""

        operation = self.session.query(model.Operation).filter_by(id=operation_id).one()
        # Load lazy fields:
        operation.user
        operation.project
        operation.operation_group
        operation.algorithm

        return operation


    def try_get_operation_by_id(self, operation_id):
        """
        Try to call self.get_operation_by_id, but when operation was not found, instead of failing, return None.
        This could be called from situations like: stopping & removing op.
        A check for None is compulsory after this call!
        """

        try:
            return self.get_operation_by_id(operation_id)
        except SQLAlchemyError:
            self.logger.exception("Operation not found for ID %s, we will return None" % operation_id)
            return None


    def get_operation_by_gid(self, operation_gid):
        """Retrieve OPERATION entity for a given gid."""
        try:
            operation = self.session.query(model.Operation).filter_by(gid=operation_gid).one()
            operation.user
            operation.project
            operation.operation_group
            operation.algorithm.algorithm_category
            return operation
        except SQLAlchemyError:
            self.logger.exception("When fetching gid %s" % operation_gid)
            return None


    def get_all_operations_for_uploaders(self, project_id):
        """
        Returns all finished upload operations.
        """
        try:
            result = self.session.query(model.Operation).join(model.Algorithm).join(model.AlgorithmCategory).filter(
                                        model.AlgorithmCategory.rawinput == True).filter(
                                        model.Operation.fk_launched_in == project_id).filter(
                                        model.Operation.status == model.STATUS_FINISHED).all()
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def is_upload_operation(self, operation_gid):
        """
        Returns True only if the operation with the given gid is an upload operation.
        """
        try:
            result = self.session.query(model.Operation).join(model.Algorithm).join(model.AlgorithmCategory
                                        ).filter(model.AlgorithmCategory.rawinput == True
                                                 ).filter(model.Operation.gid == operation_gid).count()
            return result > 0
        except SQLAlchemyError:
            return False


    def count_resulted_datatypes(self, operation_id):
        """
        Returns the number of resulted datatypes from the specified operation.
        """
        try:
            result = self.session.query(model.DataType).filter_by(fk_from_operation=operation_id).count()
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def get_operation_process_for_operation(self, operation_id):
        """
        Get the OperationProcessIdentifier for this operation id.
        """
        try:
            result = self.session.query(model.OperationProcessIdentifier
                                        ).filter(model.OperationProcessIdentifier.fk_from_operation == operation_id
                                                 ).one()
        except NoResultFound:
            self.logger.debug("No operation process found for operation id=%s." % (str(operation_id),))
            result = None
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            result = None
        return result


    def get_operations_in_group(self, operation_group_id, is_count=False,
                                only_first_operation=False, only_gids=False):
        """
        Retrieve OPERATION entities for a given group.
        """
        result = None
        try:
            query = self.session.query(model.Operation)
            if only_gids:
                query = self.session.query(model.Operation.gid)
            query = query.filter_by(fk_operation_group=operation_group_id)
            if is_count:
                result = query.count()
            elif only_first_operation:
                result = query.first()
            else:
                result = query.all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return result


    def compute_disk_size_for_started_ops(self, user_id):
        """ Get all the disk space that should be reserved for the started operations of this user. """
        try:
            expected_hdd_size = self.session.query(func.sum(model.Operation.estimated_disk_size)
                                                   ).filter(model.Operation.fk_launched_by == user_id
                                                   ).filter(model.Operation.status == model.STATUS_STARTED).scalar()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            expected_hdd_size = 0
        return expected_hdd_size or 0


    def get_filtered_operations(self, project_id, filter_chain, page_start=0, page_size=20, is_count=False):
        """
        :param project_id: current project ID
        :param filter_chain: instance of FilterChain
        :param is_count: when True, return a number, otherwise the list of operation entities

        :return a list of filtered operation in current project, page by page, or the total count for them.
        """
        try:
            select_clause = self.session.query(func.min(model.Operation.id))
            if not is_count:
                # Do not add select columns in case of COUNT, as they will be ignored anyway
                select_clause = self.session.query(func.min(model.Operation.id), func.max(model.Operation.id),
                                                   func.count(model.Operation.id),
                                                   func.max(model.Operation.fk_operation_group),
                                                   func.min(model.Operation.fk_from_algo),
                                                   func.max(model.Operation.fk_launched_by),
                                                   func.min(model.Operation.create_date),
                                                   func.min(model.Operation.start_date),
                                                   func.max(model.Operation.completion_date),
                                                   func.min(model.Operation.status),
                                                   func.max(model.Operation.additional_info),
                                                   func.min(case_([(model.Operation.visible, 1)], else_=0)),
                                                   func.min(model.Operation.user_group),
                                                   func.min(model.Operation.gid))

            query = select_clause.join(model.Algorithm).join(
                model.AlgorithmCategory).filter(model.Operation.fk_launched_in == project_id)

            if filter_chain is not None:
                filter_string = filter_chain.get_sql_filter_equivalent()
                query = query.filter(eval(filter_string))
            query = query.group_by(case_([(model.Operation.fk_operation_group > 0,
                                           - model.Operation.fk_operation_group)], else_=model.Operation.id))

            if is_count:
                return query.count()

            return query.order_by(desc(func.max(model.Operation.id))).offset(page_start).limit(page_size).all()

        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return 0 if is_count else None


    def get_results_for_operation(self, operation_id, filters=None):
        """
        Retrieve DataTypes entities, resulted after executing an operation.
        """
        try:
            query = self.session.query(model.DataType
                                       ).filter_by(fk_from_operation=operation_id
                                       ).filter(and_(model.DataType.type != self.EXCEPTION_DATATYPE_GROUP,
                                                     model.DataType.type != self.EXCEPTION_DATATYPE_SIMULATION))
            if filters:
                filter_str = filters.get_sql_filter_equivalent()
                if filter_str is not None:
                    query = query.filter(eval(filter_str))
            query = query.order_by(model.DataType.id)
            result = query.all()

            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def get_operations_for_datatype(self, datatype_gid, only_relevant=True, only_in_groups=False):
        """
        Returns all the operations which uses as an input parameter
        the dataType with the specified GID.
        If the flag only_relevant is True than only the relevant operations will be returned.
    
        If only_in_groups is True than this method will return only the operations that are part
        from an operation group, otherwise it will return only the operations that are NOT part of an operation group.
        """
        try:
            query = self.session.query(model.Operation).filter(
                                model.Operation.parameters.like('%' + datatype_gid + '%')).join(
                                model.Algorithm).join(model.AlgorithmCategory).filter(
                                model.AlgorithmCategory.display == False)
            query = self._apply_visibility_and_group_filters(query, only_relevant, only_in_groups)
            result = query.all()
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def get_operations_for_datatype_group(self, datatype_group_id, only_relevant=True, only_in_groups=False):
        """
        Returns all the operations which uses as an input parameter a datatype from the given DataTypeGroup.
        If the flag only_relevant is True than only the relevant operations will be returned.
    
        If only_in_groups is True than this method will return only the operations that are
        part from an operation group, otherwise it will return only the operations that
        are NOT part of an operation group.
        """
        try:
            query = self.session.query(model.Operation).filter(
                model.DataType.fk_datatype_group == datatype_group_id).filter(
                model.Operation.parameters.like('%' + model.DataType.gid + '%')).join(
                model.Algorithm).join(model.AlgorithmCategory).filter(
                model.AlgorithmCategory.display == False)
            query = self._apply_visibility_and_group_filters(query, only_relevant, only_in_groups)
            result = query.all()
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def get_operations_in_burst(self, burst_id, is_count=False):
        """
        Return a list with all the operations generated by a given burst. 
        These need to be removed when the burst is deleted.
        :param: is_count When True, a counter of the filtered operations is returned.
        """
        try:
            result = self.session.query(model.Operation
                                    ).join(model.WorkflowStep, model.WorkflowStep.fk_operation == model.Operation.id
                                    ).join(model.Workflow, model.Workflow.id == model.WorkflowStep.fk_workflow
                                    ).filter(model.Workflow.fk_burst == burst_id)
            if is_count:
                result = result.count()
            else:
                result = result.all()
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            if is_count:
                return 0
            return []


    @staticmethod
    def _apply_visibility_and_group_filters(query, only_relevant, only_in_groups):
        """
        Used for applying filters on the given query.
        """
        if only_relevant:
            query = query.filter(model.Operation.visible == True)
        if only_in_groups:
            query = query.filter(model.Operation.fk_operation_group != None)
        else:
            query = query.filter(model.Operation.fk_operation_group == None)
        return query


    def set_operation_and_group_visibility(self, entity_gid, is_visible, is_operation_group=False):
        """
        Sets the operation visibility.
    
        If 'is_operation_group' is True than this method will change the visibility for all
        the operation from the OperationGroup with the GID field equal to 'entity_gid'.
        """
        try:
            query = self.session.query(model.Operation)
            if is_operation_group:
                group = self.get_operationgroup_by_gid(entity_gid)
                query = query.filter(model.Operation.fk_operation_group == group.id)
            else:
                query = query.filter(model.Operation.gid == entity_gid)
            query.update({"visible": is_visible})
            self.session.commit()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)


    def get_operationgroup_by_gid(self, gid):
        """Retrieve by GID"""
        try:
            result = self.session.query(model.OperationGroup).filter_by(gid=gid).one()
            return result
        except SQLAlchemyError:
            return None


    def get_operationgroup_by_id(self, op_group_id):
        """Retrieve by ID"""
        try:
            result = self.session.query(model.OperationGroup).filter_by(id=op_group_id).one()
            return result
        except SQLAlchemyError:
            return None


    def get_operation_numbers(self, proj_id):
        """
        Count total number of operations started for current project.
        """
        stats = self.session.query(model.Operation.status, func.count(model.Operation.id)
                                    ).filter_by(fk_launched_in=proj_id
                                    ).group_by(model.Operation.status).all()
        stats = dict(stats)
        finished = stats.get(model.STATUS_FINISHED, 0)
        started = stats.get(model.STATUS_STARTED, 0)
        failed = stats.get(model.STATUS_ERROR, 0)
        canceled = stats.get(model.STATUS_CANCELED, 0)
        pending = stats.get(model.STATUS_PENDING, 0)

        return finished, started, failed, canceled, pending


    #
    # CATEGORY RELATED METHODS
    #

    def get_algorithm_categories(self):
        """Retrieve all existent categories of Algorithms."""
        try:
            categories = self.session.query(model.AlgorithmCategory).distinct().order_by(
                model.AlgorithmCategory.displayname).all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            categories = []
        return categories


    def get_uploader_categories(self):
        """Retrieve categories with raw_input = true"""
        try:
            result = self.session.query(model.AlgorithmCategory).filter_by(rawinput=True).all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            result = []
        return result


    def get_raw_categories(self):
        """Retrieve categories with raw_input = true"""
        try:
            result = self.session.query(model.AlgorithmCategory).filter_by(defaultdatastate='RAW_DATA').all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            result = []
        return result


    def get_visualisers_categories(self):
        """Retrieve categories with display = true"""
        try:
            result = self.session.query(model.AlgorithmCategory).filter_by(display=True).all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            result = []
        return result


    def get_launchable_categories(self, elimin_viewers=False):
        """Retrieve algorithm categories which can be launched on right-click (optionally filter visualizers)"""
        try:
            result = self.session.query(model.AlgorithmCategory).filter_by(launchable=True)
            if elimin_viewers:
                result = result.filter_by(display=False)
            result = result.all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            result = []
        return result


    def get_category_by_id(self, categ_id):
        """Retrieve category with given id"""
        try:
            result = self.session.query(model.AlgorithmCategory).filter_by(id=categ_id).one()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            result = None
        return result


    def filter_category(self, displayname, rawinput, display, launchable, order_nr):
        """Retrieve category with given id"""
        try:
            result = self.session.query(model.AlgorithmCategory
                                        ).filter_by(displayname=displayname
                                        ).filter_by(rawinput=rawinput).filter_by(display=display
                                        ).filter_by(launchable=launchable).filter_by(order_nr=order_nr).one()
            return result
        except NoResultFound:
            return None


    #
    # ALGORITHM RELATED METHODS
    #

    def get_algorithm_by_id(self, algorithm_id):
        try:
            result = self.session.query(model.Algorithm).filter_by(id=algorithm_id).one()
            result.algorithm_category
            return result
        except SQLAlchemyError as ex:
            self.logger.exception(ex)
            return None


    def get_algorithm_by_module(self, module_name, class_name):
        try:
            result = self.session.query(model.Algorithm).filter_by(module=module_name, classname=class_name).one()
            result.algorithm_category
            return result
        except SQLAlchemyError:
            return None


    def get_applicable_adapters(self, compatible_class_names, launch_categ):
        """
        Retrieve a list of algorithms in a given list of categories with a given dataType classes as required input.
        """
        try:
            return self.session.query(model.Algorithm
                                      ).filter_by(removed=False
                                      ).filter(model.Algorithm.fk_category.in_(launch_categ)
                                      ).filter(model.Algorithm.required_datatype.in_(compatible_class_names)
                                      ).order_by(model.Algorithm.fk_category
                                      ).order_by(model.Algorithm.group_name).all()
        except SQLAlchemyError:
            self.logger.exception("Could not retrieve applicable Adapters ...")
            return []


    def get_adapters_from_categories(self, categories):
        """
        Retrieve a list of stored adapters in the given categories.
        """
        try:
            return self.session.query(model.Algorithm
                                      ).filter_by(removed=False
                                      ).filter(model.Algorithm.fk_category.in_(categories)
                                      ).order_by(model.Algorithm.group_name
                                      ).order_by(model.Algorithm.displayname).all()
        except SQLAlchemyError:
            self.logger.exception("Could not retrieve Adapters ...")
            return []


    #
    # RESULT FIGURE RELATED CODE
    #

    def load_figure(self, figure_id):
        """ Load a figure with all it's lazy load fields to have all required 
        info available. """
        try:
            figure = self.session.query(model.ResultFigure).filter_by(id=figure_id).one()
            figure.project
            figure.operation
            return figure
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def get_previews(self, project_id, user_id=None, selected_session_name='all_sessions'):
        """
        This method returns a tuple of 2 elements. The first element represents a dictionary
        of form {'$session_name': [list_of_figures]}. This dictionary contains data only for the selected self.session.
        If the selected session is 'all_sessions' than it will contain data for all the sessions.
        The second element of the returned tuple is a dictionary of form
        {'$session_name': $no_of_figures_in_this_session, ...}.
        This dictionary contains information about all the sessions.
    
        selected_session_name - represents the name of the session for which you
                                want to obtain the stored figures.
        """
        try:
            previews_info = self._get_previews_info(project_id, user_id)
            if selected_session_name == 'all_sessions':
                session_names = previews_info.keys()
                session_names.sort()
            else:
                session_names = [selected_session_name]

            result = {}
            for session_name in session_names:
                figures_list = self.session.query(model.ResultFigure
                                                  ).filter_by(fk_in_project=project_id
                                                              ).filter_by(session_name=session_name)
                if user_id is not None:
                    figures_list = figures_list.filter_by(fk_for_user=user_id)

                figures_list = figures_list.order_by(desc(model.ResultFigure.id)).all()

                # Force loading of project and operation - needed to compute image path
                for figure in figures_list:
                    figure.project
                    figure.operation
                result[session_name] = figures_list
            return result, previews_info
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return {}, {}


    def _get_previews_info(self, project_id, user_id):
        """
        Returns a dictionary of form: {$session_name: $no_of_images_in_this_session, ...}.
        """
        try:
            result = {}
            session_items = self.session.query(model.ResultFigure.session_name,
                                               func.count(model.ResultFigure.session_name)
                                               ).filter_by(fk_in_project=project_id)
            if user_id is not None:
                session_items = session_items.filter_by(fk_for_user=user_id)

            session_items = session_items.group_by(model.ResultFigure.session_name
                                                   ).order_by(model.ResultFigure.session_name).all()

            for item in session_items:
                result[item[0]] = item[1]
            return result

        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return {}


    def get_figure_count(self, project_id, user_id):
        """
        Used to generate sequential image names.
        """
        try:
            session_items = self.session.query(model.ResultFigure).filter_by(fk_in_project=project_id)
            if user_id is not None:
                session_items = session_items.filter_by(fk_for_user=user_id)

            return session_items.count()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return {}


    def get_figures_for_operation(self, operation_id):
        """Retrieve Figure entities, resulted after executing an operation."""
        try:
            result = self.session.query(model.ResultFigure).filter_by(fk_from_operation=operation_id).all()
            for figure in result:
                figure.project
                figure.operation
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None
