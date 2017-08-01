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
DAO layer for WorkFlow and Burst entities.
"""

from sqlalchemy import func as func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.sql.expression import desc, not_, or_
from tvb.core.entities import model
from tvb.core.entities.storage.root_dao import RootDAO


class WorkflowDAO(RootDAO):
    """
    DAO layer for WorkFlow and Burst entities.
    """


    def get_non_validated_entities(self, reference_time):
        """
        Get a list of all categories, portlets and algorithm groups that were not found valid since the reference_time.
        Used in initializer on each start to filter out any entities that for some reason became invalid.
        :return tuple (list of entities to get invalidated) (list of entities to be removed)
        """
        try:
            stored_adapters = self.session.query(model.Algorithm
                                        ).filter(or_(model.Algorithm.last_introspection_check == None,
                                                     model.Algorithm.last_introspection_check < reference_time)).all()
            categories = self.session.query(model.AlgorithmCategory
                                        ).filter(model.AlgorithmCategory.last_introspection_check<reference_time).all()
            portlets = self.session.query(model.Portlet
                                        ).filter(model.Portlet.last_introspection_check < reference_time).all()
            result = stored_adapters + categories, portlets
        except SQLAlchemyError as ex:
            self.logger.exception(ex)
            result = [], []
        return result


    def get_bursts_for_project(self, project_id, page_start=0, page_size=None, count=False):
        """Get latest 50 BurstConfiguration entities for the current project"""
        try:
            bursts = self.session.query(model.BurstConfiguration
                                        ).filter_by(fk_project=project_id
                                                    ).order_by(desc(model.BurstConfiguration.start_time))
            if count:
                return bursts.count()
            if page_size is not None:
                bursts = bursts.offset(max(page_start, 0)).limit(page_size)

            bursts = bursts.all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            bursts = None
        return bursts


    def get_max_burst_id(self):
        """
        Return the maximum of the currently stored burst IDs to be used as the new burst name.
        This is not a thread-safe value, but we use it just for a label.
        """
        try:
            max_id = self.session.query(func.max(model.BurstConfiguration.id)).one()
            if max_id[0] is None:
                return 0
            return max_id[0]
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return 0


    def count_bursts_with_name(self, burst_name, project_id):
        """
        Return the number of burst already named 'custom_b%' and NOT 'custom_b%_%' in current project.
        """
        count = 0
        try:
            count = self.session.query(model.BurstConfiguration
                                       ).filter_by(fk_project=project_id
                                       ).filter(model.BurstConfiguration.name.like(burst_name + '%')
                                       ).filter(not_(model.BurstConfiguration.name.like(burst_name + '/_%/_%',
                                                                                        escape='/'))
                                       ).count()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return count


    def get_burst_by_id(self, burst_id):
        """Get the BurstConfiguration entity with the given id"""
        try:
            burst = self.session.query(model.BurstConfiguration).filter_by(id=burst_id).one()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            burst = None
        return burst


    def get_visualization_steps(self, workflow_id):
        """Retrieve all the visualization steps for a workflow."""
        try:
            result = self.session.query(model.WorkflowStepView
                                        ).filter(model.WorkflowStepView.fk_workflow == workflow_id
                                        ).order_by(model.WorkflowStepView.tab_index,
                                                   model.WorkflowStepView.index_in_tab).all()
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def get_workflow_steps(self, workflow_id):
        """Retrieve all the simulation/analyzers steps for a workflow."""
        try:
            # Also check that index is non-negative to preserve backwards
            # compatibility to versions < 1.0.2.
            result = self.session.query(model.WorkflowStep
                                        ).filter(model.WorkflowStep.fk_workflow == workflow_id
                                        ).filter(model.WorkflowStep.step_index > -1
                                        ).order_by(model.WorkflowStep.step_index).all()
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None


    def get_workflows_for_burst(self, burst_id, is_count=False):
        """Returns all the workflows that were launched for this burst id"""
        query = self.session.query(model.Workflow).filter_by(fk_burst=burst_id)

        if is_count:
            result = query.count()
        else:
            result = query.all()

        return result


    def get_workflow_by_id(self, workflow_id):
        """"Returns the workflow instance with the given id"""
        workflow = None
        try:
            workflow = self.session.query(model.Workflow).filter_by(id=workflow_id).one()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)

        return workflow


    def get_workflow_step_by_step_index(self, workflow_id, step_index):
        """
        :returns: WorkflowStep entity or None.
        """
        step = None
        try:
            step = self.session.query(model.WorkflowStep).filter_by(fk_workflow=workflow_id,
                                                                    step_index=step_index).one()
        except NoResultFound:
            self.logger.debug("No step found for workflow_id=%s and step_index=%s" % (workflow_id, step_index))
        except SQLAlchemyError as excep:
            self.logger.exception(excep)

        return step


    def get_workflow_steps_for_position(self, workflow_id, tab_index, index_in_tab):
        """
        Retrieve a list of analyzers corresponding to current cell in the portlets grid.
        Will be used for deciding the interface.
        """
        steps = []
        try:
            steps = self.session.query(model.WorkflowStep
                                       ).filter_by(fk_workflow=workflow_id,
                                                   tab_index=tab_index, index_in_tab=index_in_tab
                                       ).order_by(model.WorkflowStep.step_index).all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)

        return steps


    def get_configured_portlets_for_id(self, portlet_id):
        """
        Get the workflow steps that were generated from the portlet given by portlet_id.
        """
        wf_steps = []
        try:
            wf_steps = self.session.query(model.WorkflowStepView).filter_by(fk_portlet=portlet_id).all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)

        return wf_steps


    def get_workflow_step_for_operation(self, operation_id):
        """
        Returns the executed workflow step from which resulted
        the operation with the given id 'operation_id'.
        Returns None if there is no such executed workflow step.
        """
        step = None
        try:
            step = self.session.query(model.WorkflowStep).filter_by(fk_operation=operation_id).one()
        except NoResultFound:
            self.logger.debug("No step found for operation_id=%s" % operation_id)
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return step


    def get_available_portlets(self, ):
        """
        Get all the stored portlets form the db.
        """
        portlets = []
        try:
            portlets = self.session.query(model.Portlet).order_by(model.Portlet.name).all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)

        return portlets


    def get_portlet_by_identifier(self, portlet_identifier):
        """
        Given an identifer retieve the portlet that corresponds to it.
        """
        portlet = None
        try:
            portlet = self.session.query(model.Portlet).filter_by(algorithm_identifier=portlet_identifier).one()
        except NoResultFound:
            self.logger.debug("No portlet found with id=%s." % portlet_identifier)
        except SQLAlchemyError as excep:
            self.logger.exception(excep)

        return portlet


    def get_portlet_by_id(self, portlet_id):
        """
        Given an portlet id retieve the portlet entity.
        """
        portlet = None
        try:
            portlet = self.session.query(model.Portlet).filter_by(id=portlet_id).one()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)

        return portlet


    def get_workflow_for_operation_id(self, operation_id):
        """
        Get the workflow from which operation_id was generated.
        """
        workflow = None
        try:
            workflow = self.session.query(model.Workflow).join(model.WorkflowStep
                                          ).filter(model.WorkflowStep.fk_operation == operation_id).one()
        except NoResultFound:
            self.logger.warning("Operation with id=%s was not generated from any workflow." % operation_id)
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return workflow


    def get_burst_for_operation_id(self, operation_id):
        """
        Get the burst for which this operation was created.
        """
        burst = None
        try:
            burst = self.session.query(model.BurstConfiguration
                                       ).join(model.Workflow, model.Workflow.fk_burst==model.BurstConfiguration.id
                                       ).join(model.WorkflowStep
                                       ).filter(model.WorkflowStep.fk_operation==operation_id).one()
        except NoResultFound:
            self.logger.debug("No burst found for operation id = %s"%(operation_id,))
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return burst


    def get_all_datatypes_in_burst(self, burst_id):
        """
        Get all dataTypes in burst, order by their creation, desc.
        
        :param burst_id BurstConfiguration Identifier.
        :returns: list dataType GIDs or empty list.
        """
        try:
            groups = self.session.query(model.DataTypeGroup,
                           ).join(model.Operation, model.DataTypeGroup.fk_from_operation == model.Operation.id
                           ).join(model.WorkflowStep, model.Operation.id == model.WorkflowStep.fk_operation
                           ).join(model.Workflow).filter(model.Workflow.fk_burst == burst_id
                           ).order_by(desc(model.DataTypeGroup.id)).all()
            result = self.session.query(model.DataType
                                      ).filter(model.DataType.fk_parent_burst == burst_id
                                      ).filter(model.DataType.fk_datatype_group == None
                                      ).filter(model.DataType.type != self.EXCEPTION_DATATYPE_GROUP
                                      ).order_by(desc(model.DataType.id)).all()
            result.extend(groups)
        except SQLAlchemyError as exc:
            self.logger.exception(exc)
            result = []
        return result


    def get_dynamics_for_user(self, user_id):
        try:
            return self.session.query(model.Dynamic).filter(model.Dynamic.fk_user == user_id).all()
        except SQLAlchemyError as exc:
            self.logger.exception(exc)
            return []


    def get_dynamic(self, dyn_id):
        try:
            return self.session.query(model.Dynamic).filter(model.Dynamic.id == dyn_id).one()
        except SQLAlchemyError as exc:
            self.logger.exception(exc)

    def get_dynamic_by_name(self, name):
        try:
            return self.session.query(model.Dynamic).filter(model.Dynamic.name == name).all()
        except SQLAlchemyError as exc:
            self.logger.exception(exc)
