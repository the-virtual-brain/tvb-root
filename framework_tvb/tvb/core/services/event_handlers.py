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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
import copy
import threading
import Queue
import xml.dom.minidom
from time import sleep
from xml.dom.minidom import Node
from tvb.basic.profile import TvbProfile
from tvb.basic.logger.builder import get_logger
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.xml_reader import TYPE_STR, ATT_TYPE, ATT_VALUE, ATT_NAME
from tvb.core.services.flow_service import FlowService
from tvb.core.services.operation_service import OperationService


LOCKS_QUEUE = Queue.Queue(0)
for i in max(2, range(TvbProfile.current.MAX_THREADS_NUMBER)):
    LOCKS_QUEUE.put(1)
#Global variable to store events as read from XML files.
EXECUTORS_DICT = {}

LOGGER = get_logger(__name__)

EVENTS_SUFIX = "events.xml"

# This variable names can be used in XML, 
# to notify a variable which will be filled at runtime.
RUNTIME_USERNAME = "$$CURRENT_USER$$"
RUNTIME_PROJECT = "$$CURRENT_PROJECT$$"

# XML tags
ELEM_ADAPTER = "adapter"
ELEM_LAUNCHER = "launcher"
ELEM_METHOD = "method"
ELEM_ARGS = "args"
ELEM_KWARGS = "kwargs"
ATT_HOOKPOINT = "hookpoint"
ATT_MODULE = "module"
ATT_CLASS = "class"
ATT_OPERATION_HIDDEN = "operation-hidden"
ATT_PRIMITIVE = "primitive"
ATT_UID = OperationService.ATT_UID
ATT_INSTANCE = "instance"
TYPE_FILE = "file"



class GenericEventExecutor(object):
    """
    Thread for executing a generic event. 
    The event in this case is any method from any service.
    """
    runtime_mapping = {RUNTIME_USERNAME: 'self.current_user',
                       RUNTIME_PROJECT: 'self.current_project'}
    delay = 0


    def __init__(self, event_node, **kwargs):
        self.event_node = event_node
        self.event = None
        self.callable_object = None
        self.call_method = "__init__"   # Dummy default __init__
        self.arguments = kwargs
        self.current_user = None
        self.current_project = None


    def _prepare_parameters(self):
        """Main method to prepare call parameters.
           Will handle 'runtime' evaluated params."""
        if self.event is None:
            self.parse_event_node()

        prepared_parameters = {}
        for arg in self.arguments:
            for key in self.runtime_mapping:
                if type(self.arguments[arg]) in (str, unicode) and key in self.arguments[arg]:
                    runtime_value = eval(self.arguments[arg].replace(key, self.runtime_mapping[key]))
                    if runtime_value is not None:
                        prepared_parameters[arg] = runtime_value
                    break
            else:
                prepared_parameters[arg] = self._prepare_custom_parameter(arg)
        return prepared_parameters


    def _prepare_custom_parameter(self, arg):
        """ 
        Treat executor specific parameter.
        Default behavior implemented bellow, is just read value from self.arguments.
        """
        return self.arguments[arg]


    def run(self):
        """
        This method will be executed in a different thread.
        Executed on the event mentioned in description.
        """
        #Try to get a spot to launch own operation.
        LOCKS_QUEUE.get(True)
        try:
            parameters = self._prepare_parameters()
            method = getattr(self.callable_object, self.call_method)
            method(**parameters)
        except Exception:
            LOGGER.exception("Could not execute operation!")
        LOCKS_QUEUE.put(1)


    def set_runtime_parameters(self, user=None, project=None):
        """Fill 'runtime' variables."""
        self.current_user = user
        self.current_project = project


    def parse_event_node(self):
        """Parse the stored event node to get required data and arguments."""
        kw_parameters = {}
        for one_arg in self.event_node.childNodes:
            if one_arg.nodeType != Node.ELEMENT_NODE:
                continue
            if one_arg.nodeName == ELEM_LAUNCHER:
                module = one_arg.getAttribute(ATT_MODULE)
                class_name = one_arg.getAttribute(ATT_CLASS)
                call_obj = __import__(module, globals(), locals(), [class_name])
                call_obj = eval('call_obj.' + class_name)
                if eval(one_arg.getAttribute(ATT_INSTANCE)):
                    call_obj = call_obj()
                self.callable_object = call_obj
                continue
            if one_arg.nodeName == ELEM_METHOD:
                self.call_method = one_arg.getAttribute(ATT_NAME)
                continue
            if one_arg.nodeName == ELEM_ARGS:
                kw_parameters.update(_parse_arguments(one_arg))
                continue
            LOGGER.info("Ignored undefined node %s", str(one_arg.nodeName))

        self.arguments.update(kw_parameters)



class AdapterEventExecutor(GenericEventExecutor):
    """
    Thread for firing a custom event. 
    The event in this case is a custom method, on an Adapter instance. 
    """


    def __init__(self, events_node, **kwargs):
        GenericEventExecutor.__init__(self, events_node, **kwargs)
        self.operation_visible = True


    def _prepare_custom_parameter(self, arg):
        """ Overwrite to prepare specific parameters. """
        current_value = self.arguments[arg]

        if isinstance(current_value, dict) and ATT_UID in current_value:
            uid = current_value[ATT_UID]
            current_type = model.DataType
            ## Search current entity's type in adapter.get_input_tree result
            full_input_tree = self.callable_object.get_input_tree()
            for att_def in full_input_tree:
                if att_def[ATT_NAME] == arg:
                    current_type = att_def[ATT_TYPE]
                    break
            ### Retrieve entity of the correct Type from DB.
            current_value = dao.get_last_data_with_uid(uid, current_type)
        return current_value


    def run(self):
        """
        Fire TVB operation which makes sure the Adapter method is called.
        """
        LOCKS_QUEUE.get(True)
        try:
            if self.delay > 0:
                sleep(self.delay)
            parameters = self._prepare_parameters()
            FlowService().fire_operation(self.callable_object, self.current_user, self.current_project.id,
                                         method_name=self.call_method, visible=self.operation_visible, **parameters)
        except Exception:
            LOGGER.exception("Could not execute operation!")
        LOCKS_QUEUE.put(1)


    def parse_event_node(self):
        """
        Parse the stored event node to get required data and arguments.
        """
        kw_parameters = {}
        for one_arg in self.event_node.childNodes:
            if one_arg.nodeType != Node.ELEMENT_NODE:
                continue
            if one_arg.nodeName == ELEM_ADAPTER:
                #TODO: so far there is no need for it, but we should maybe
                #handle cases where same module/class but different init parameter
                group = dao.find_group(one_arg.getAttribute(ATT_MODULE), one_arg.getAttribute(ATT_CLASS))
                adapter = ABCAdapter.build_adapter(group)
                result_uid = one_arg.getAttribute(ATT_UID)
                if result_uid:
                    kw_parameters[ATT_UID] = result_uid
                LOGGER.debug("Adapter used is %s", str(adapter.__class__))
                self.callable_object = adapter
                continue
            if one_arg.nodeName == ELEM_METHOD:
                self.call_method = one_arg.getAttribute(ATT_NAME)
                if one_arg.getAttribute(ATT_OPERATION_HIDDEN):
                    self.operation_visible = False
                continue
            if one_arg.nodeName == ELEM_ARGS:
                kw_parameters.update(_parse_arguments(one_arg))
                continue
            LOGGER.info("Ignored undefined node %s", str(one_arg.nodeName))

        self.arguments.update(kw_parameters)



def handle_event(hookpoint, current_user=None, current_project=None):
    """
    This method looks in the event dictionary for events that
    correspond to the current method_name, then launches those events.
    """
    if hookpoint in EXECUTORS_DICT:
        for executor in EXECUTORS_DICT[hookpoint]:
            executor = copy.deepcopy(executor)
            executor.set_runtime_parameters(current_user, current_project)
            threading.Thread(target=executor.run).start()
    else:
        LOGGER.debug("No events are declared for method %s", hookpoint)



def read_events(path):
    """
    For a list of file paths read all the events defined in files that 
    end with events.xml
    """
    for one_path in path:
        all_files = os.listdir(one_path)
        for f_name in all_files:
            if f_name.endswith(EVENTS_SUFIX):
                read_from_file(os.path.join(one_path, f_name), EXECUTORS_DICT)



def read_from_file(file_name, executor_dict):
    """
    For an XML file that defines some events, get these events in a dictionary.
    """
    doc_xml = xml.dom.minidom.parse(file_name)
    for one_event in doc_xml.lastChild.childNodes:
        if one_event.nodeType != Node.ELEMENT_NODE:
            continue
        else:
            event_trigger = one_event.getAttribute(ATT_HOOKPOINT)
            LOGGER.debug("Found event for hook point %s", event_trigger)
            if one_event.getAttribute('type') == 'generic':
                event_executor = GenericEventExecutor(one_event)
            else:
                event_executor = AdapterEventExecutor(one_event)
                delay = one_event.getAttribute('delay')
                if delay is not None and delay != '':
                    event_executor.delay = int(delay)
            if event_trigger in executor_dict:
                executor_dict[event_trigger].append(event_executor)
            else:
                executor_dict[event_trigger] = [event_executor]



def _parse_arguments(xml_node):
    """
    From a given XML node, retrieve all children, 
    and parse them an input parameters to adapter.
    """
    kw_parameters = {}
    for kwarg in xml_node.childNodes:
        if kwarg.nodeType != Node.ELEMENT_NODE:
            continue
        current_name = kwarg.getAttribute(ATT_NAME)
        current_type = kwarg.getAttribute(ATT_TYPE)
        current_value = kwarg.getAttribute(ATT_VALUE)
        try:
            module = kwarg.getAttribute(ATT_MODULE)
        except Exception:
            module = None

        if current_type == TYPE_STR:
            current_value = str(current_value).lstrip().rstrip()
        elif current_type == TYPE_FILE:
            try:
                current_value = os.path.normpath(current_value)
                if module:
                    python_module = __import__(str(module), globals(), locals(), ["__init__"])
                    root_folder = os.path.dirname(os.path.abspath(python_module.__file__))
                    current_value = os.path.join(root_folder, current_value)
            except ImportError:
                LOGGER.exception("Argument reference towards a demo-data file is invalid!")
        elif current_type == ATT_UID:
            current_value = {ATT_UID: current_value}
        elif current_type == ATT_PRIMITIVE:
            current_value = eval(current_value)
        kw_parameters[str(current_name)] = current_value

    return kw_parameters


    
    