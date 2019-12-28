# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

from sqlalchemy import Column, Integer, ForeignKey, String, DateTime
from sqlalchemy.orm import relationship, backref
from tvb.core.utils import format_timedelta
from tvb.core.entities.model.model_operation import OperationGroup
from tvb.core.entities.model.model_project import Project
from tvb.core.neotraits.db import HasTraitsIndex


# TODO: this used to extend Exportable
# TODO: re
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