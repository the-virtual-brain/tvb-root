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
Here we define entities related to workflows and portlets.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import json
import datetime
from sqlalchemy import Integer, String, Column, ForeignKey, DateTime
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declared_attr
from tvb.core.entities.model.model_base import Base
from tvb.core.entities.model.model_project import Project
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.model.model_operation import Algorithm
from tvb.core.entities.exportable import Exportable



class Portlet(Base):
    """
    Store the Portlet entities. 
    One entity will hold:
    - portlet id;
    - path to the XML file that declares it;
    - a name given in the XML
    - an unique identifier (also from XML)
    - last date of introspection.
    """
    __tablename__ = 'PORTLETS'

    id = Column(Integer, primary_key=True)
    algorithm_identifier = Column(String)
    xml_path = Column(String)
    name = Column(String)
    last_introspection_check = Column(DateTime)


    def __init__(self, algorithm_identifier, xml_path, name="TVB Portlet"):
        self.algorithm_identifier = algorithm_identifier
        self.xml_path = xml_path
        self.name = name
        self.last_introspection_check = datetime.datetime.now()



class Workflow(Base, Exportable):
    """
    Main Workflow class.
    One Burst can have multiple Workflow entities associated (when launching a range).
    """
    __tablename__ = 'WORKFLOWS'

    STATUS_STARTED = 'started'
    STATUS_FINISHED = 'finished'

    id = Column(Integer, primary_key=True)
    fk_project = Column(Integer, ForeignKey('PROJECTS.id', ondelete="CASCADE"))
    fk_burst = Column(Integer, ForeignKey('BURST_CONFIGURATIONS.id', ondelete="CASCADE"))
    status = Column(String)

    project = relationship(Project, backref=backref('WORKFLOWS', order_by=id, cascade="delete, all"))
    burst = relationship(BurstConfiguration, backref=backref('WORKFLOWS', order_by=id, cascade="delete, all"))


    def __init__(self, project_id, burst_id, status='started'):
        self.fk_project = project_id
        self.fk_burst = burst_id
        self.status = status


    def from_dict(self, info_dict):
        self.status = info_dict['status']



class _BaseWorkflowStep(Exportable):
    """
    Base class for a step inside a Workflow.
    This class will not be persisted, it is just used for grouping common behavior.
    
    - fk_workflow - the id of the parent workflow current step is part of;
    - fk_algorithm - the id of the algorithm-group that should be executed into this step;
    - tab_index - position on the burst-page - used for filling portlet input parameters;
    - index_in_tab - the index in the current tab of burst page;
    - static_param -  a dictionary which contains the static parameters that should be passed
                    to the executed algorithm. This parameters will be the same from one execution to another.
    - dynamic_param - a dictionary which contains the dynamic parameters that should be passed
                    to the executed algorithm. This parameters will be different from one execution to another.
                    A dynamic parameter represents a parameter resulted after the execution
                    of a 'WorkflowStep' from the current workflow.

                    The form of the 'dynamic_param' dict will be of form:
                    dynamic_param = {$param_name: $dict1, ...} 
                    where $dict1 will be a dict with two entries.

                    $dict1 = {step_index: $idx, datatype_idx: $dt_idx} where
                    $idx -  represents the index of the 'WorkflowStep', from the current workflow,
                            after which execution resulted the datatype
                    $dt_idx -   the index of the datatype resulted after the execution of the
                            'WorkflowStep' with index $idx from the current workflow
    - user_dynamic_param - dynamic parameters dictionary
    """
    tab_index = Column(Integer)
    index_in_tab = Column(Integer)

    _static_param = Column(String)
    _dynamic_param = Column(String)
    user_dynamic_param = Column(String)

    ### Transient fields:
    step_visible = True


    @declared_attr
    def fk_workflow(cls):
        """
        Column attributes with Foreign Keys can not be declared directly in SqlAlchemy
        """
        return Column(Integer, ForeignKey('WORKFLOWS.id', ondelete="CASCADE"))


    @declared_attr
    def fk_algorithm(cls):
        """
        Column attributes with Foreign Keys can not be declared directly in SqlAlchemy
        """
        return Column(Integer, ForeignKey('ALGORITHMS.id'))


    def __init__(self, workflow_id, algorithm_id, tab_index, index_in_tab,
                 static_param, dynamic_param, user_dynamic_param=None):
        """ Set default attributes in current step."""
        self.fk_workflow = workflow_id
        self.fk_algorithm = algorithm_id
        self.tab_index = tab_index
        self.index_in_tab = index_in_tab
        self.static_param = static_param
        self.dynamic_param = dynamic_param
        self.user_dynamic_param = user_dynamic_param


    def from_dict(self, data):
        self._static_param = data['_static_param']
        self._dynamic_param = data['_dynamic_param']
        self.user_dynamic_param = data['user_dynamic_param']
        self.tab_index = data['tab_index']
        self.index_in_tab = data['index_in_tab']


    def _set_dynamic_parameters(self, dynamic_params):
        """Set dynamic parameters into JSON"""
        if isinstance(dynamic_params, str):
            self._dynamic_param = dynamic_params
        else:
            self._dynamic_param = json.dumps(dynamic_params)


    def _get_dynamic_parameters(self):
        """Get dynamic parameters as a dictionary"""
        return json.loads(self._dynamic_param)


    dynamic_param = property(fget=_get_dynamic_parameters, fset=_set_dynamic_parameters)


    @property
    def dynamic_workflow_param_names(self):
        """Return list with strings, representing flatten names of the input dynamic parameters"""
        return self.dynamic_param.keys()


    def _set_static_parameters(self, static_params):
        """Set static parameters into JSON"""
        if isinstance(static_params, str):
            self._static_param = static_params
        else:
            self._static_param = json.dumps(static_params)


    def _get_static_parameters(self):
        """Get static parameters as a dictionary"""
        return json.loads(self._static_param)


    static_param = property(fget=_get_static_parameters, fset=_set_static_parameters)



class WorkflowStepView(Base, _BaseWorkflowStep):
    """
    View Step inside a workflow.
    
    - portlet_id - the id of the portlet (or -1 if simulator or result measure);
    - ui_name - custom title given by the user when selecting a portlet.
    
    """
    __tablename__ = 'WORKFLOW_VIEW_STEPS'
    id = Column(Integer, primary_key=True)

    fk_portlet = Column(Integer, ForeignKey('PORTLETS.id', ondelete="CASCADE"))
    ui_name = Column(String)

    workflow = relationship(Workflow, backref=backref('WORKFLOW_VIEW_STEPS', order_by=id, cascade="delete, all"))
    algorithm = relationship(Algorithm, backref=backref('WORKFLOW_VIEW_STEPS', order_by=id))
    portlet = relationship(Portlet, backref=backref('WORKFLOW_VIEW_STEPS', order_by=id, cascade="delete, all"))


    def __init__(self, algorithm_id, static_param=None, dynamic_param=None, user_dynamic_param=None,
                 workflow_id=None, portlet_id=-1, tab_index=None, index_in_tab=None, ui_name="Default"):
        """Fill current selected portlet parameters."""
        _BaseWorkflowStep.__init__(self, workflow_id, algorithm_id, tab_index, index_in_tab,
                                   static_param, dynamic_param, user_dynamic_param)
        self.fk_portlet = portlet_id
        self.ui_name = ui_name
        self.step_visible = False


    def clone(self):
        """Return clone - not linked to any DB session or Operation."""
        return WorkflowStepView(self.fk_algorithm, self.static_param, self.dynamic_param, self.user_dynamic_param,
                                self.fk_workflow, self.fk_portlet, self.tab_index, self.index_in_tab, self.ui_name)


    def from_dict(self, data):
        super(WorkflowStepView, self).from_dict(data)
        self.ui_name = data['ui_name']



class WorkflowStep(Base, _BaseWorkflowStep):
    """
    Analyze/Simulate step inside a workflow.
    This type of step will generate an operation to be scheduled for execution.
    
    - step_index - the index of the step into the workflow;
    - fk_operation - the operation that resulted from this workflow step.
    
    """
    __tablename__ = 'WORKFLOW_STEPS'
    id = Column(Integer, primary_key=True)

    step_index = Column(Integer)
    fk_operation = Column(Integer, ForeignKey('OPERATIONS.id', ondelete="SET NULL"))

    workflow = relationship(Workflow, backref=backref('WORKFLOW_STEPS', order_by=id, cascade="delete, all"))
    algorithm = relationship(Algorithm, backref=backref('WORKFLOW_STEPS', order_by=id))


    def __init__(self, algorithm_id, static_param=None, dynamic_param=None, user_dynamic_param=None,
                 workflow_id=None, step_index=None, tab_index=None, index_in_tab=None):
        """Fill current step parameters."""
        _BaseWorkflowStep.__init__(self, workflow_id, algorithm_id, tab_index, index_in_tab,
                                   static_param, dynamic_param, user_dynamic_param)
        self.step_index = step_index


    def clone(self):
        """Return clone - not linked to any DB session or Operation."""
        return WorkflowStep(self.fk_algorithm, self.static_param, self.dynamic_param, self.user_dynamic_param,
                            self.fk_workflow, self.step_index, self.tab_index, self.index_in_tab)


    def from_dict(self, data):
        super(WorkflowStep, self).from_dict(data)
        self.step_index = data['step_index']
    
