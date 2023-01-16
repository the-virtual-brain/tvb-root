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
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""

from sqlalchemy import Column, Integer, ForeignKey, String, DateTime
from sqlalchemy.orm import relationship, backref
from tvb.core.entities.model.model_operation import OperationGroup
from tvb.core.entities.model.model_project import Project
from tvb.core.neotraits.db import Base, HasTraitsIndex
from tvb.core.utils import format_timedelta

PARAM_RANGE_PREFIX = 'range_'
RANGE_PARAMETER_1 = "range_1"
RANGE_PARAMETER_2 = "range_2"


class Dynamic(Base):
    __tablename__ = 'DYNAMIC'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    fk_user = Column(Integer, ForeignKey('USERS.id'))
    code_version = Column(Integer)

    model_class = Column(String)
    model_parameters = Column(String)
    integrator_class = Column(String)
    integrator_parameters = Column(String)

    def __init__(self, name, user_id, model_class, model_parameters, integrator_class, integrator_parameters):
        self.name = name
        self.fk_user = user_id
        self.model_class = model_class
        self.model_parameters = model_parameters
        self.integrator_class = integrator_class
        self.integrator_parameters = integrator_parameters

    def __repr__(self):
        return "<Dynamic(%s, %s, %s)" % (self.name, self.model_class, self.integrator_class)


class BurstConfiguration(HasTraitsIndex):
    BURST_RUNNING = 'running'
    BURST_ERROR = 'error'
    BURST_FINISHED = 'finished'
    BURST_CANCELED = 'canceled'

    selected_tab = -1
    is_group = False

    datatypes_number = Column(Integer)
    dynamic_ids = Column(String, default='[]', nullable=False)

    range1 = Column(String, nullable=True)
    range2 = Column(String, nullable=True)

    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    fk_project = Column(Integer, ForeignKey('PROJECTS.id', ondelete='CASCADE'))
    project = relationship(Project, backref=backref('BurstConfiguration', cascade='all,delete'))

    name = Column(String)
    status = Column(String)
    error_message = Column(String)

    start_time = Column(DateTime)
    finish_time = Column(DateTime)

    # This will store the first Simulation Operation, and First Simulator GID, in case of PSE
    simulator_gid = Column(String, nullable=True)
    fk_simulation = Column(Integer, ForeignKey('OPERATIONS.id', ondelete="SET NULL"), nullable=True)

    fk_operation_group = Column(Integer, ForeignKey('OPERATION_GROUPS.id'), nullable=True)
    operation_group = relationship(OperationGroup, foreign_keys=fk_operation_group,
                                   primaryjoin=OperationGroup.id == fk_operation_group, cascade='none')

    fk_metric_operation_group = Column(Integer, ForeignKey('OPERATION_GROUPS.id'), nullable=True)
    metric_operation_group = relationship(OperationGroup, foreign_keys=fk_metric_operation_group,
                                          primaryjoin=OperationGroup.id == fk_metric_operation_group, cascade='none')

    # Transient attribute, for when copying or branching
    parent_burst_object = None

    def __init__(self, project_id, status="running", name=None):
        super().__init__()
        self.fk_project = project_id
        self.name = name
        self.status = status
        self.dynamic_ids = '[]'

    def clone(self):
        new_burst = BurstConfiguration(self.fk_project)
        new_burst.name = self.name
        new_burst.range1 = self.range1
        new_burst.range2 = self.range2
        new_burst.status = self.BURST_RUNNING
        new_burst.parent_burst_object = self
        return new_burst

    @property
    def process_time(self):
        if self.finish_time is not None and self.start_time is not None:
            return format_timedelta(self.finish_time - self.start_time)
        return ''

    def is_pse_burst(self):
        return self.range1 is not None

    @property
    def ranges(self):
        if self.range2:
            return [self.range1, self.range2]
        if self.range1:
            return [self.range1]
        return None

    @property
    def is_finished(self):
        return self.status != self.BURST_RUNNING

    @property
    def operation_info_for_burst_removal(self):
        """
        Return operation id for whole burst removal and a flag specifying whether the current burst is a group.
        """
        if self.fk_operation_group is None:
            return self.fk_simulation, False
        return self.fk_operation_group, True
