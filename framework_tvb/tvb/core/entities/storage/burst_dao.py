from operator import not_
from sqlalchemy import desc, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import NoResultFound
from tvb.core.entities.model.simulator.burst_configuration import BurstConfiguration2
from tvb.core.entities.model.simulator.simulator import SimulatorIndex
from tvb.core.entities.storage.root_dao import RootDAO


class BurstDAO(RootDAO):
    """
    DAO layer for Burst entities.
    """

    def get_bursts_for_project(self, project_id, page_start=0, page_size=None, count=False):
        """Get latest 50 BurstConfiguration entities for the current project"""
        try:
            bursts = self.session.query(BurstConfiguration2
                                        ).filter_by(project_id=project_id
                                                    ).order_by(desc(BurstConfiguration2.start_time))
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
            max_id = self.session.query(func.max(BurstConfiguration2.id)).one()
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
            count = self.session.query(BurstConfiguration2
                                       ).filter_by(project_id=project_id
                                       ).filter(BurstConfiguration2.name.like(burst_name + '%')
                                       ).filter(not_(BurstConfiguration2.name.like(burst_name + '/_%/_%', escape='/'))
                                       ).count()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return count

    def get_burst_by_id(self, burst_id):
        """Get the BurstConfiguration entity with the given id"""
        try:
            burst = self.session.query(BurstConfiguration2).filter_by(id=burst_id).one()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            burst = None
        return burst

    def get_burst_for_operation_id(self, operation_id):
        burst = None
        try:
            burst = self.session.query(BurstConfiguration2
                                       ).join(SimulatorIndex, SimulatorIndex.fk_parent_burst == BurstConfiguration2.id
                                              ).filter(SimulatorIndex.fk_from_operation == operation_id).one()
        except NoResultFound:
            self.logger.debug("No burst found for operation id = %s" % (operation_id,))
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return burst
