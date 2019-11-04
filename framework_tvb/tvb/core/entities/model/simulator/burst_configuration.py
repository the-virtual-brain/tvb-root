from sqlalchemy import Column, Integer, ForeignKey, String, DateTime
from sqlalchemy.orm import relationship, backref

from tvb.core.utils import format_timedelta
from tvb.core.entities.model.model_operation import OperationGroup
from tvb.core.entities.model.model_project import Project
from tvb.core.neotraits.db import HasTraitsIndex


# TODO: this used to extend Exportable
class BurstConfiguration2(HasTraitsIndex):
    BURST_RUNNING = 'running'
    BURST_ERROR = 'error'
    BURST_FINISHED = 'finished'
    BURST_CANCELED = 'canceled'

    # TODO: Fix attrs for portlets
    nr_of_tabs = 0
    selected_tab = -1
    is_group = False
    datatypes_number = Column(Integer)
    dynamic_ids = Column(String, default='[]', nullable=False)

    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    project_id = Column(Integer, ForeignKey('PROJECTS.id', ondelete='CASCADE'))
    project = relationship(Project, backref=backref('BurstConfiguration2', cascade='all,delete'))

    name = Column(String)
    status = Column(String)
    error_message = Column(String)

    start_time = Column(DateTime)
    finish_time = Column(DateTime)

    operation_group_id = Column(Integer, ForeignKey('OPERATION_GROUPS.id'), nullable=True)
    operation_group = relationship(OperationGroup, foreign_keys=operation_group_id,
                                   primaryjoin=OperationGroup.id == operation_group_id, cascade='none')

    metric_operation_group_id = Column(Integer, ForeignKey('OPERATION_GROUPS.id'), nullable=True)
    metric_operation_group = relationship(OperationGroup, foreign_keys=metric_operation_group_id,
                                   primaryjoin=OperationGroup.id == metric_operation_group_id, cascade='none')

    def __init__(self, project_id, simulator_id=None, status="running", name=None):
        self.project_id = project_id
        self.simulator_id = simulator_id
        self.name = name
        self.status = status
        self.dynamic_ids = '[]'

    def clone(self):
        new_burst = BurstConfiguration2(self.project_id)
        new_burst.name = self.name
        new_burst.status = self.BURST_RUNNING
        return new_burst

    @property
    def process_time(self):
        if self.finish_time is not None and self.start_time is not None:
            return format_timedelta(self.finish_time - self.start_time)
        return ''