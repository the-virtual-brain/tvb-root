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
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""

import json
import copy
from datetime import datetime
from sqlalchemy import Integer, String, DateTime, Column, ForeignKey
from sqlalchemy.orm import relationship, backref

from tvb.core.utils import string2date
from tvb.core.entities.exportable import Exportable
from tvb.core.entities.model.model_base import Base
from tvb.core.entities.model.model_project import Project
from tvb.basic.traits.types_basic import MapAsJson
from tvb.core.utils import parse_json_parameters, timedelta2string


NUMBER_OF_PORTLETS_PER_TAB = 4
KEY_PARAMETER_CHECKED = 'checked'
KEY_SAVED_VALUE = 'value'

BURST_INFO_FILE = "bursts_info.json"
BURSTS_DICT_KEY = "bursts_dict"
DT_BURST_MAP = "dt_mapping"


PARAM_RANGE_PREFIX = 'range_'
RANGE_PARAMETER_1 = "range_1"
RANGE_PARAMETER_2 = "range_2"


class BurstConfiguration(Base, Exportable):
    """
    Contains information required to rebuild the interface of a burst that was
    already launched.
    
    - simulator_configuration - hold a dictionary with what entries were displayed in the simulator part
    
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
    disk_size = Column(Integer)

    _simulator_configuration = Column(String)

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


    def from_dict(self, dictionary):
        self.name = dictionary['name']
        self.status = dictionary['status']
        self.error_message = dictionary['error_message']
        self.start_time = string2date(dictionary['start_time'])
        self.finish_time = string2date(dictionary['finish_time'])
        self.workflows_number = int(dictionary['workflows_number'])
        self.datatypes_number = int(dictionary['datatypes_number'])
        self.disk_size = int(dictionary['disk_size'])
        self._simulator_configuration = dictionary['_simulator_configuration']


    def prepare_after_load(self):
        """
        Load Simulator configuration from JSON string, as it was stored in DB.
        """
        self.tabs = [TabConfiguration() for _ in range(self.nr_of_tabs)]
        self.simulator_configuration = parse_json_parameters(self._simulator_configuration)


    def prepare_before_save(self):
        """
        From dictionary, compose JSON string for DB storage of burst configuration parameters.
        """
        self._simulator_configuration = json.dumps(self.simulator_configuration, cls=MapAsJson.MapAsJsonEncoder)
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
            if param in self.simulator_configuration and KEY_SAVED_VALUE in self.simulator_configuration[param] \
                    and self.simulator_configuration[param][KEY_SAVED_VALUE] != '0':
                return True
        return False


    @property
    def current_weight(self):
        """
        Return a dictionary with information about current burst's weight.
        """
        result = {'process_time': None,
                  'datatypes_number': self.datatypes_number,
                  'disk_size': str(self.disk_size) + " KB" if self.disk_size is not None else None,
                  'number_of_workflows': self.workflows_number,
                  'start_time': self.start_time,
                  'error': self.error_message}

        if self.finish_time is not None and self.start_time is not None:
            result['process_time'] = timedelta2string(self.finish_time - self.start_time)

        return result


    def mark_status(self, error=False, success=False, cancel=False):
        """
        Mark current burst as finished with error.
        Prepare entity for DB storage (e.g. populate JSON semi-transient attributes).
        DO NOT directly set status to error on Burst entity and save in DB,a s you might ruin some data.
        """
        if error:
            self.status = self.BURST_ERROR
        elif success:
            self.status = self.BURST_FINISHED
        elif cancel:
            self.status = self.BURST_CANCELED
        self.finish_time = datetime.now()


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
        if KEY_SAVED_VALUE not in self.simulator_configuration[param_name]:
            return None
        return self.simulator_configuration[param_name][KEY_SAVED_VALUE]


    def get_all_simulator_values(self):
        """
        :returns: dictionary {simulator_attribute_name: value}
        """
        result = {}
        any_checked = False
        for key in self.simulator_configuration:
            if KEY_PARAMETER_CHECKED in self.simulator_configuration[key] \
                    and self.simulator_configuration[key][KEY_PARAMETER_CHECKED]:
                any_checked = True
            if KEY_SAVED_VALUE not in self.simulator_configuration[key]:
                continue
            result[key] = self.simulator_configuration[key][KEY_SAVED_VALUE]
        return result, any_checked


    def clone(self):
        """
        Return an exact copy of the entity with the exception than none of it's
        sub-entities (tabs, portlets, workflow steps) are persisted in db.
        """
        new_burst = BurstConfiguration(self.fk_project)
        new_burst.name = self.name
        new_burst.simulator_configuration = self.simulator_configuration
        new_burst.selected_tab = self.selected_tab
        new_burst.status = self.BURST_RUNNING
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
        for idx in xrange(len(self.portlets)):
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
    
    
           
        