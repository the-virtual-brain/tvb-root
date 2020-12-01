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

"""
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""

from sqlalchemy import Column, Integer, ForeignKey, String, DateTime
from sqlalchemy.orm import relationship, backref
from tvb.core.utils import format_timedelta
from tvb.core.entities.model.model_operation import OperationGroup
from tvb.core.entities.model.model_project import Project
from tvb.core.neotraits.db import Base, HasTraitsIndex

NUMBER_OF_PORTLETS_PER_TAB = 4

PARAM_RANGE_PREFIX = 'range_'
RANGE_PARAMETER_1 = "range_1"
RANGE_PARAMETER_2 = "range_2"

## TabConfiguration entity is not moved in the "transient" module, although it's not stored in DB,
## because it was directly referenced from the BurstConfiguration old class.
## In most of the case, we depend in "transient" module from classed in "model", and not vice-versa.


class TabConfiguration():
    """
    Helper entity to hold data that is currently being configured in a new
    burst page.
    """

    def __init__(self):
        self.portlets = [None for _ in range(NUMBER_OF_PORTLETS_PER_TAB)]

    def reset(self):
        """
        Set to None all portlets in current TAB.
        """
        for idx in range(len(self.portlets)):
            self.portlets[idx] = None

    def get_portlet(self, portlet_id):
        """
        :returns: a PortletConfiguration entity.
        """
        for portlet in self.portlets:
            if portlet is not None and str(portlet.portlet_id) == str(portlet_id):
                return portlet
        return None

    def clone(self):
        """
        Return an exact copy of the entity with the exception than none of it's
        sub-entities (portlets, workflow steps) are persisted in db.
        """
        new_config = TabConfiguration()
        for portlet_idx, portlet_entity in enumerate(self.portlets):
            if portlet_entity is not None:
                new_config.portlets[portlet_idx] = portlet_entity.clone()
            else:
                new_config.portlets[portlet_idx] = None
        return new_config

    def __repr__(self):
        repr_str = "Tab: "
        for portlet in self.portlets:
            repr_str += str(portlet) + '; '
        return repr_str


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

    # TODO: Fix attrs for portlets
    nr_of_tabs = 0
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
    fk_simulation = Column(Integer, ForeignKey('OPERATIONS.id'), nullable=True)

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
