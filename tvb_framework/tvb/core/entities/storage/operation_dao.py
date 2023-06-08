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
DAO operations related to Algorithms and User Operations are defined here.
 
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from sqlalchemy import and_
from sqlalchemy import func as func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.sql.expression import case as case_, desc
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.entities.model.model_operation import *
from tvb.core.entities.model.model_operation import STATUS_ERROR
from tvb.core.entities.storage.root_dao import RootDAO, DEFAULT_PAGE_SIZE


class OperationDAO(RootDAO):
    """
    OPERATION RELATED METHODS
    """


    def get_operation_by_id(self, operation_id):
        """Retrieve OPERATION entity for a given Identifier."""

        operation = self.session.query(Operation).filter_by(id=operation_id).one()
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
            operation = self.session.query(Operation).filter_by(gid=operation_gid).one()
            operation.user
            operation.project
            operation.operation_group
            operation.algorithm.algorithm_category
            return operation
        except SQLAlchemyError:
            self.logger.exception("When fetching gid %s" % operation_gid)
            return None

    def get_operation_lazy_by_gid(self, operation_gid):
        """Retrieve OPERATION entity for a given gid."""
        try:
            return self.session.query(Operation).filter_by(gid=operation_gid).one()
        except SQLAlchemyError:
            self.logger.exception("When fetching gid %s" % operation_gid)
            return None


    def get_all_operations_for_uploaders(self, project_id):
        """
        Returns all finished upload operations.
        """
        try:
            result = self.session.query(Operation).join(Algorithm).join(AlgorithmCategory).filter(
                                        AlgorithmCategory.rawinput == True).filter(
                                        Operation.fk_launched_in == project_id).filter(
                                        Operation.status == STATUS_FINISHED).all()
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None

    def get_operations(self, status=None, algorithm_classname="SimulatorAdapter"):
        if status is None:
            status = [STATUS_PENDING, STATUS_STARTED]
        try:
            result = self.session.query(Operation).join(Algorithm) \
                .filter(Algorithm.classname == algorithm_classname) \
                .filter(Operation.status.in_(status)).all()
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None

    def get_operations_for_hpc_job(self):
        status = [STATUS_PENDING, STATUS_STARTED]
        algorithm_classname = "SimulatorAdapter"
        queue_full = False
        try:
            result = self.session.query(Operation).join(Algorithm) \
                .filter(Algorithm.classname == algorithm_classname) \
                .filter(Operation.status.in_(status)) \
                .filter(Operation.queue_full == queue_full).all()
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def is_upload_operation(self, operation_gid):
        """
        Returns True only if the operation with the given gid is an upload operation.
        """
        try:
            result = self.session.query(Operation).join(Algorithm).join(AlgorithmCategory
                                        ).filter(AlgorithmCategory.rawinput == True
                                                 ).filter(Operation.gid == operation_gid).count()
            return result > 0
        except SQLAlchemyError:
            return False


    def count_resulted_datatypes(self, operation_id):
        """
        Returns the number of resulted datatypes from the specified operation.
        """
        try:
            result = self.session.query(DataType).filter_by(fk_from_operation=operation_id).count()
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def get_operation_process_for_operation(self, operation_id):
        """
        Get the OperationProcessIdentifier for this operation id.
        """
        try:
            result = self.session.query(OperationProcessIdentifier
                                        ).filter(OperationProcessIdentifier.fk_from_operation == operation_id
                                                 ).one()
        except NoResultFound:
            self.logger.debug("No operation process found for operation id=%s." % (str(operation_id),))
            result = None
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            result = None
        return result


    def get_operations_with_error_in_project(self, project_id):
        """
        Retrieve OPERATION with errors entities for a given project.
        """
        result = None
        try:
            query = self.session.query(Operation).filter_by(fk_launched_in=project_id).filter(
                Operation.status == STATUS_ERROR)
            result = query.all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return result

    def get_operations_in_group(self, operation_group_id, is_count=False,
                                only_first_operation=False, only_gids=False):
        """
        Retrieve OPERATION entities for a given group.
        """
        result = None
        try:
            query = self.session.query(Operation)
            if only_gids:
                query = self.session.query(Operation.gid)
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
            expected_hdd_size = self.session.query(func.sum(Operation.estimated_disk_size)
                                                   ).filter(Operation.fk_launched_by == user_id
                                                   ).filter(Operation.status == STATUS_STARTED).scalar()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            expected_hdd_size = 0
        return expected_hdd_size or 0


    def get_filtered_operations(self, project_id, filter_chain, page_start=0,
                                page_size=DEFAULT_PAGE_SIZE, is_count=False):
        """
        :param project_id: current project ID
        :param filter_chain: instance of FilterChain
        :param is_count: when True, return a number, otherwise the list of operation entities

        :return a list of filtered operation in current project, page by page, or the total count for them.
        """
        try:
            select_clause = self.session.query(func.min(Operation.id))
            if not is_count:
                # Do not add select columns in case of COUNT, as they will be ignored anyway
                select_clause = self.session.query(func.min(Operation.id), func.max(Operation.id),
                                                   func.count(Operation.id),
                                                   func.max(Operation.fk_operation_group),
                                                   func.min(Operation.fk_from_algo),
                                                   func.max(Operation.fk_launched_by),
                                                   func.min(Operation.create_date),
                                                   func.min(Operation.start_date),
                                                   func.max(Operation.completion_date),
                                                   func.min(Operation.status),
                                                   func.max(Operation.additional_info),
                                                   func.min(case_((Operation.visible, 1), else_=0)),
                                                   func.min(Operation.user_group),
                                                   func.min(Operation.gid))

            query = select_clause.join(Algorithm).join(
                AlgorithmCategory).filter(Operation.fk_launched_in == project_id)

            if filter_chain is not None:
                filter_string = filter_chain.get_sql_filter_equivalent()
                query = query.filter(eval(filter_string))
            query = query.group_by(case_((Operation.fk_operation_group > 0,
                                           - Operation.fk_operation_group), else_=Operation.id))

            if is_count:
                return query.count()

            return query.order_by(desc(func.max(Operation.id))).offset(page_start).limit(page_size).all()

        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return 0 if is_count else None


    def get_results_for_operation(self, operation_id):
        """
        Retrieve DataTypes entities, resulted after executing an operation.
        """
        try:
            query = self.session.query(DataType
                                       ).filter_by(fk_from_operation=operation_id
                                       ).filter(and_(DataType.type != self.EXCEPTION_DATATYPE_GROUP,
                                                     DataType.type != self.EXCEPTION_DATATYPE_SIMULATION))
            query = query.order_by(DataType.id)
            result = query.all()
            for dt in result:
                dt.display_name

            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def set_operation_and_group_visibility(self, entity_gid, is_visible, is_operation_group=False):
        """
        Sets the operation visibility.
    
        If 'is_operation_group' is True than this method will change the visibility for all
        the operation from the OperationGroup with the GID field equal to 'entity_gid'.
        """
        try:
            query = self.session.query(Operation)
            if is_operation_group:
                group = self.get_operationgroup_by_gid(entity_gid)
                query = query.filter(Operation.fk_operation_group == group.id)
            else:
                query = query.filter(Operation.gid == entity_gid)
            query.update({"visible": is_visible})
            self.session.commit()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)


    def get_operationgroup_by_gid(self, gid):
        """Retrieve by GID"""
        try:
            result = self.session.query(OperationGroup).filter_by(gid=gid).one()
            return result
        except SQLAlchemyError:
            return None


    def get_operationgroup_by_id(self, op_group_id):
        """Retrieve by ID"""
        try:
            result = self.session.query(OperationGroup).filter_by(id=op_group_id).one()
            return result
        except SQLAlchemyError:
            return None


    def get_operation_numbers(self, proj_id):
        """
        Count total number of operations started for current project.
        """
        stats = self.session.query(Operation.status, func.count(Operation.id)
                                    ).filter_by(fk_launched_in=proj_id
                                    ).group_by(Operation.status).all()
        stats = dict(stats)
        finished = stats.get(STATUS_FINISHED, 0)
        started = stats.get(STATUS_STARTED, 0)
        failed = stats.get(STATUS_ERROR, 0)
        canceled = stats.get(STATUS_CANCELED, 0)
        pending = stats.get(STATUS_PENDING, 0)

        return finished, started, failed, canceled, pending


    #
    # CATEGORY RELATED METHODS
    #

    def get_algorithm_categories(self):
        """Retrieve all existent categories of Algorithms."""
        try:
            categories = self.session.query(AlgorithmCategory).distinct().order_by(
                AlgorithmCategory.displayname).all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            categories = []
        return categories


    def get_uploader_categories(self):
        """Retrieve categories with raw_input = true"""
        try:
            result = self.session.query(AlgorithmCategory).filter_by(rawinput=True).all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            result = []
        return result


    def get_raw_categories(self):
        """Retrieve categories with raw_input = true"""
        try:
            result = self.session.query(AlgorithmCategory).filter_by(defaultdatastate='RAW_DATA').all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            result = []
        return result


    def get_visualisers_categories(self):
        """Retrieve categories with display = true"""
        try:
            result = self.session.query(AlgorithmCategory).filter_by(display=True).all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            result = []
        return result


    def get_launchable_categories(self, elimin_viewers=False):
        """Retrieve algorithm categories which can be launched on right-click (optionally filter visualizers)"""
        try:
            result = self.session.query(AlgorithmCategory).filter_by(launchable=True)
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
            result = self.session.query(AlgorithmCategory).filter_by(id=categ_id).one()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            result = None
        return result


    def filter_category(self, displayname, rawinput, display, launchable, order_nr):
        """Retrieve category with given id"""
        try:
            result = self.session.query(AlgorithmCategory
                                        ).filter_by(displayname=displayname
                                        ).filter_by(rawinput=rawinput).filter_by(display=display
                                        ).filter_by(launchable=launchable).filter_by(order_nr=order_nr).one()
            return result
        except NoResultFound:
            return None


    #
    # ALGORITHM RELATED METHODS
    #

    def get_all_algorithms(self):
        try:
            result = self.session.query(Algorithm).distinct().all()
            return result
        except SQLAlchemyError as ex:
            self.logger.exception(ex)
            return None

    def get_algorithm_by_id(self, algorithm_id):
        try:
            result = self.session.query(Algorithm).filter_by(id=algorithm_id).one()
            result.algorithm_category
            return result
        except SQLAlchemyError as ex:
            self.logger.exception(ex)
            return None


    def get_algorithm_by_module(self, module_name, class_name):
        try:
            result = self.session.query(Algorithm).filter_by(module=module_name, classname=class_name).one()
            result.algorithm_category
            return result
        except SQLAlchemyError:
            return None


    def get_applicable_adapters(self, compatible_class_names, launch_categ):
        """
        Retrieve a list of algorithms in a given list of categories with a given dataType classes as required input.
        """
        try:
            return self.session.query(Algorithm
                                      ).filter_by(removed=False
                                      ).filter(Algorithm.fk_category.in_(launch_categ)
                                      ).filter(Algorithm.required_datatype.in_(compatible_class_names)
                                      ).order_by(Algorithm.fk_category
                                      ).order_by(Algorithm.group_name).all()
        except SQLAlchemyError:
            self.logger.exception("Could not retrieve applicable Adapters ...")
            return []


    def get_adapters_from_categories(self, categories):
        """
        Retrieve a list of stored adapters in the given categories.
        """
        try:
            return self.session.query(Algorithm
                                      ).filter_by(removed=False
                                      ).filter(Algorithm.fk_category.in_(categories)
                                      ).order_by(Algorithm.group_name
                                      ).order_by(Algorithm.displayname).all()
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
            figure = self.session.query(ResultFigure).filter_by(id=figure_id).one()
            figure.project
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
                session_names = list(previews_info)
                session_names.sort()
            else:
                session_names = [selected_session_name]

            result = {}
            for session_name in session_names:
                figures_list = self.session.query(ResultFigure
                                                  ).filter_by(fk_in_project=project_id
                                                              ).filter_by(session_name=session_name)
                if user_id is not None:
                    figures_list = figures_list.filter_by(fk_for_user=user_id)

                figures_list = figures_list.order_by(desc(ResultFigure.id)).all()

                # Force loading of project and operation - needed to compute image path
                for figure in figures_list:
                    figure.project
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
            session_items = self.session.query(ResultFigure.session_name,
                                               func.count(ResultFigure.session_name)
                                               ).filter_by(fk_in_project=project_id)
            if user_id is not None:
                session_items = session_items.filter_by(fk_for_user=user_id)

            session_items = session_items.group_by(ResultFigure.session_name
                                                   ).order_by(ResultFigure.session_name).all()

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
            session_items = self.session.query(ResultFigure).filter_by(fk_in_project=project_id)
            if user_id is not None:
                session_items = session_items.filter_by(fk_for_user=user_id)

            return session_items.count()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return {}
