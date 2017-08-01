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
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""

import json
import copy
import six
from datetime import datetime
from sqlalchemy import Integer, String, DateTime, Column, ForeignKey
from sqlalchemy.orm import relationship, backref
from tvb.core.utils import string2date
from tvb.core.entities.exportable import Exportable
from tvb.core.entities.model.model_base import Base
from tvb.core.entities.model.model_project import Project
from tvb.basic.traits.types_basic import MapAsJson
from tvb.core.utils import parse_json_parameters, format_timedelta


NUMBER_OF_PORTLETS_PER_TAB = 4
KEY_PARAMETER_CHECKED = 'checked'
KEY_SAVED_VALUE = 'value'

BURST_INFO_FILE = "bursts_info.json"
BURSTS_DICT_KEY = "bursts_dict"
DT_BURST_MAP = "dt_mapping"

PARAM_RANGE_PREFIX = 'range_'
RANGE_PARAMETER_1 = "range_1"
RANGE_PARAMETER_2 = "range_2"

PARAM_CONNECTIVITY = 'connectivity'
PARAM_SURFACE = 'surface'
PARAM_MODEL = 'model'
PARAM_INTEGRATOR = 'integrator'

PARAMS_MODEL_PATTERN = 'model_parameters_option_%s_%s'


class BurstConfiguration(Base, Exportable):
    """
    Contains information required to rebuild the interface of a burst that was
    already launched.
    
    - simulator_configuration - hold a dictionary with what entries were displayed in the simulator part
    - dynamic_ids - A list of dynamic id's associated with the connectivity nodes. Used by the region parameters page

    """
    __tablename__ = "BURST_CONFIGURATIONS"

    BURST_RUNNING = 'running'
    BURST_ERROR = 'error'
    BURST_FINISHED = 'finished'
    BURST_CANCELED = 'canceled'

    id = Column(Integer, primary_key=True)
    fk_project = Column(Integer, ForeignKey('PROJECTS.id', ondelete="CASCADE"))
    name = Column(String)
    status = Column(String)
    error_message = Column(String)

    start_time = Column(DateTime)
    finish_time = Column(DateTime)
    workflows_number = Column(Integer)
    datatypes_number = Column(Integer)

    _simulator_configuration = Column(String)
    _dynamic_ids = Column(String, default='[]', nullable=False)

    ### Transient attributes start from bellow
    simulator_configuration = {}

    nr_of_tabs = 3
    selected_tab = 0
    ## This is the default portlet configuration on a 'blank' new burst. When selecting
    ## portlets each entry can be replaced with a [portlet_id, portlet_name] pair
    DEFAULT_PORTLET_CONFIGURATION = [[[-1, 'None'], [-1, 'None'], [-1, 'None'], [-1, 'None']],
                                     [[-1, 'None'], [-1, 'None'], [-1, 'None'], [-1, 'None']],
                                     [[-1, 'None'], [-1, 'None'], [-1, 'None'], [-1, 'None']]]

    parent = relationship(Project, backref=backref("BURST_CONFIGURATIONS", cascade="all,delete"))


    def __init__(self, project_id, status="running", simulator_configuration=None, name=None):
        self.fk_project = project_id
        if simulator_configuration is None:
            self.simulator_configuration = {}
        else:
            self.simulator_configuration = simulator_configuration
        self.tabs = [TabConfiguration() for _ in range(self.nr_of_tabs)]
        self.name = name
        self.selected_tab = 0
        self.status = status
        self._dynamic_ids = '[]'
        self.dynamic_ids = []


    def from_dict(self, dictionary):
        self.name = dictionary['name']
        self.status = dictionary['status']
        self.error_message = dictionary['error_message']
        self.start_time = string2date(dictionary['start_time'])
        self.finish_time = string2date(dictionary['finish_time'])
        self.workflows_number = int(dictionary['workflows_number'])
        self.datatypes_number = int(dictionary['datatypes_number'])
        self._simulator_configuration = dictionary['_simulator_configuration']


    def prepare_after_load(self):
        """
        Load Simulator configuration from JSON string, as it was stored in DB.
        """
        self.tabs = [TabConfiguration() for _ in range(self.nr_of_tabs)]
        self.simulator_configuration = parse_json_parameters(self._simulator_configuration)
        self.dynamic_ids = json.loads(self._dynamic_ids)


    def prepare_before_save(self):
        """
        From dictionary, compose JSON string for DB storage of burst configuration parameters.
        """
        self._simulator_configuration = json.dumps(self.simulator_configuration, cls=MapAsJson.MapAsJsonEncoder)
        self._dynamic_ids = json.dumps(self.dynamic_ids)
        self.start_time = datetime.now()


    @property
    def is_group(self):
        """
        :returns: True, when current burst configuration will generate a group.
        """
        ## After launch and storage, if we have multiple workflows in current burst, it means there is a group.
        if self.workflows_number > 1:
            return True

        for param in [RANGE_PARAMETER_1, RANGE_PARAMETER_2]:
            try:
                if self.simulator_configuration[param][KEY_SAVED_VALUE] != '0':
                    return True
            except KeyError:
                pass
        return False


    @property
    def process_time(self):
        if self.finish_time is not None and self.start_time is not None:
            return format_timedelta(self.finish_time - self.start_time)
        return ''


    def update_simulator_configuration(self, new_values):
        """
        Update the stored simulator configuration given the input dictionary 
        :param new_values: new values
        """
        for param_name in new_values:
            self.update_simulation_parameter(param_name, new_values[param_name])


    def update_simulation_parameter(self, param_name, param_value, specific_key=KEY_SAVED_VALUE):
        """
        Update single simulator parameter value or checked state.
        """
        if param_name not in self.simulator_configuration:
            self.simulator_configuration[param_name] = {KEY_PARAMETER_CHECKED: False}

        self.simulator_configuration[param_name][specific_key] = param_value


    def get_simulation_parameter_value(self, param_name):
        """
        Read value set for simulation parameter.
        Return None, when no previous value was set.
        """
        if param_name not in self.simulator_configuration:
            self.simulator_configuration[param_name] = {}
        return self.simulator_configuration[param_name].get(KEY_SAVED_VALUE)


    def get_all_simulator_values(self):
        """
        :returns: dictionary {simulator_attribute_name: value}
        """
        result = {}
        any_checked = False
        for key, value in six.iteritems(self.simulator_configuration):
            if value.get(KEY_PARAMETER_CHECKED):
                any_checked = True
            if KEY_SAVED_VALUE not in value:
                continue
            result[key] = value[KEY_SAVED_VALUE]
        return result, any_checked


    def clone(self):
        """
        Return an exact copy of the entity with the exception than none of it's
        sub-entities (tabs, portlets, workflow steps) are persisted in db.
        """
        # todo - mh: should this return a deep copy?
        # The simulator_configuration assignment suggests not, the tabs suggest yes.
        # why are these clone's used (tab portlet etc)?
        new_burst = BurstConfiguration(self.fk_project)
        new_burst.name = self.name
        new_burst.simulator_configuration = self.simulator_configuration
        new_burst.selected_tab = self.selected_tab
        new_burst.status = self.BURST_RUNNING
        new_burst.dynamic_ids = self.dynamic_ids[:]
        new_tabs = []
        for tab in self.tabs:
            if tab is not None:
                new_tabs.append(tab.clone())
            else:
                new_tabs.append(None)
        new_burst.tabs = new_tabs
        return new_burst


    def reset_tabs(self):
        """
        Set all tabs to default configuration (empty constructor).
        """
        self.tabs = [TabConfiguration() for _ in range(self.nr_of_tabs)]


    def __repr__(self):
        return "Burst_configuration(simulator_config=%s, name=%s)" % (
            str(self.simulator_configuration), str(self.name))


    def update_selected_portlets(self):
        """
        Update the DEFAULT_PORTLET_CONFIGURATION with the selected entries from the
        current burst config.
        """
        updated_config = copy.deepcopy(self.DEFAULT_PORTLET_CONFIGURATION)
        for tab_idx, tab_value in enumerate(self.tabs):
            for idx_in_tab, portlet in enumerate(tab_value.portlets):
                if portlet is not None:
                    updated_config[tab_idx][idx_in_tab] = [portlet.portlet_id, portlet.name]
        return updated_config


    def set_portlet(self, tab_index, index_in_tab, portlet_configuration):
        """
        Set in the select with index 'index_in_tab' from the tab with index 'tab_index'
        the given 'portlet_configuration'.
        """
        self.tabs[tab_index].portlets[index_in_tab] = portlet_configuration


## TabConfiguration entity is not moved in the "transient" module, although it's not stored in DB,
## because it is directly referenced from the BurstConfiguration class above.
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
