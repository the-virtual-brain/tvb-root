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

from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.input_tree import InputTreeManager
from tvb.core.entities.model import WorkflowStep, WorkflowStepView
from tvb.core.entities.transient.burst_configuration_entities import PortletConfiguration, AdapterConfiguration
from tvb.core.entities.transient.burst_configuration_entities import WorkflowStepConfiguration
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.portlets.xml_reader import XMLPortletReader, KEY_STATIC, KEY_DYNAMIC, ATT_OVERWRITE


# The root of each prefix. The index in the adapter chain for each adapter will
# be added to form specific prefixes for each adapter from the portlet.
ADAPTER_PREFIX_ROOT = 'portlet_step_'

## Cache XML readers, to avoid parsing the XML too many times
PORTLET_XML_READERS = {}


class PortletConfigurer(object):
    """
    Helper class that handles all the functionality required from a portlet.
    Given a portlet entity, this will allow the following:
    
    - return a configurable interface in the form of a dictionary 
        equivalent to the input tree of an adapter
    - create a new PortletConfiguration entity, or update a already
        created one with a new set of parameters
        
    """
    log = get_logger(__name__)


    def __init__(self, portlet_entity):
        if portlet_entity.xml_path not in PORTLET_XML_READERS:
            PORTLET_XML_READERS[portlet_entity.xml_path] = XMLPortletReader(portlet_entity.xml_path)

        self.reader = PORTLET_XML_READERS[portlet_entity.xml_path]
        self.portlet_entity = portlet_entity


    @property
    def portlet_id(self):
        """Portlet DB identifier"""
        return self.portlet_entity.id


    @property
    def algo_identifier(self):
        """Unique identifier for current portlet."""
        return self.portlet_entity.algorithm_identifier


    @property
    def ui_name(self):
        """ Portlet name to be displayed in UI"""
        return self.portlet_entity.name


    def get_configurable_interface(self):
        """
        Given an algorithm identifier, go trough the adapter chain, and merge
        their input tree with the declared overwrites 
        """
        chain_adapters = self.reader.get_adapters_chain(self.algo_identifier)
        result = []
        for adapter_declaration in chain_adapters:
            adapter_instance = self.build_adapter_from_declaration(adapter_declaration)

            all_portlet_defined_params = self.reader.get_inputs(self.algo_identifier)
            specific_adapter_overwrites = [entry for entry in all_portlet_defined_params
                                           if ATT_OVERWRITE in entry and entry[ATT_OVERWRITE] ==
                                           adapter_declaration[ABCAdapter.KEY_NAME]]
            alg_inputs = adapter_instance.get_input_tree()
            replace_values = self._prepare_input_tree(alg_inputs, specific_adapter_overwrites)
            adapter_configuration = AdapterConfiguration(replace_values, adapter_instance.stored_adapter)
            result.append(adapter_configuration)
        return result


    @staticmethod
    def build_adapter_from_declaration(adapter_declaration):
        """
        Build and adapter from the declaration in the portlets xml.
        """
        adapter_import_path = adapter_declaration[ABCAdapter.KEY_TYPE]
        class_name = adapter_import_path.split('.')[-1]
        module_name = adapter_import_path.replace('.' + class_name, '')
        algo = dao.get_algorithm_by_module(module_name, class_name)
        if algo is not None:
            return ABCAdapter.build_adapter(algo)
        else:
            return None


    def _prepare_input_tree(self, input_list, default_values, prefix=''):
        """
        Replace the default values from the portlet interface in the adapters 
        interfaces.
        :param input_list: the adapter input tree
        :param default_values: the dictionary of overwrites declared in the 
            portlet xml
        :param prefix: in case of a group adapter, the prefix to be added to 
            each name for the selected subalgorithm
        """
        for param in input_list:
            for one_value in default_values:
                if one_value[ABCAdapter.KEY_NAME] == prefix + param[ABCAdapter.KEY_NAME]:
                    param[ABCAdapter.KEY_DEFAULT] = one_value[ABCAdapter.KEY_DEFAULT]
                    if one_value[ABCAdapter.KEY_TYPE] == KEY_DYNAMIC:
                        ## For now just display all dynamic parameters as being disabled.
                        ## If at some point we would like user to be able to select dynamic entry
                        ## should treat this case differently.
                        param[ABCAdapter.KEY_DISABLED] = True
                        param[KEY_DYNAMIC] = True

            if param.get(ABCAdapter.KEY_OPTIONS) is not None:
                new_prefix = prefix + param[ABCAdapter.KEY_NAME] + ABCAdapter.KEYWORD_PARAMS
                self._prepare_input_tree(param[ABCAdapter.KEY_OPTIONS], default_values, new_prefix)

            if param.get(ABCAdapter.KEY_ATTRIBUTES) is not None:
                new_prefix = prefix
                if param.get(ABCAdapter.KEY_TYPE) == 'dict':
                    new_prefix = prefix + param[ABCAdapter.KEY_NAME] + ABCAdapter.KEYWORD_PARAMS
                self._prepare_input_tree(param[ABCAdapter.KEY_ATTRIBUTES], default_values, new_prefix)
        return input_list


    @staticmethod
    def update_default_values(portlet_interface, portlet_configuration):
        """
        :param portlet_interface: a list of AdapterConfiguration entities.
        :param portlet_configuration: a PortletConfiguration entity.
        
        Update the defaults from each AdapterConfiguration entity with the 
        values stored in the corresponding workflow step held in the 
        PortletConfiguration entity.
        """
        # Check for any defaults first in analyzer steps
        if portlet_configuration.analyzers:
            for adapter_idx in range(len(portlet_interface[:-1])):
                saved_configuration = portlet_configuration.analyzers[adapter_idx]
                replaced_defaults_dict = InputTreeManager.fill_defaults(portlet_interface[adapter_idx].interface,
                                                                        saved_configuration.static_param)
                portlet_interface[adapter_idx].interface = replaced_defaults_dict

        # Check for visualization defaults
        if portlet_configuration.visualizer:
            saved_configuration = portlet_configuration.visualizer
            replaced_defaults_dict = InputTreeManager.fill_defaults(portlet_interface[-1].interface,
                                                                    saved_configuration.static_param)
            portlet_interface[-1].interface = replaced_defaults_dict


    def _portlet_dynamic2workflow_step(self, value):
        """
        Given a value in portlet specific declaration, eg: step_0[0], return a 
        dictionary as expected from a workflow step:
            {'step_idx' : 0, 'datatype_idx' : 0}
        """
        step_idx, datatype_idx = value.replace('step_', '').replace(']', '').split('[')
        try:
            datatype_idx = int(datatype_idx)
            self.log.debug("%s defines an output as an entry to a workflow step." % (value,))
        except ValueError:
            self.log.debug("%s defines an input as an entry to a workflow step." % (value,))
        workflow_value = {WorkflowStepConfiguration.STEP_INDEX_KEY: int(step_idx),
                          WorkflowStepConfiguration.DATATYPE_INDEX_KEY: datatype_idx}
        return workflow_value


    def prefix_adapters_parameters(self, adapter_config_list):
        """
        Prepend separate prefix to the name of each entry of the adapter interfaces.

        :param adapter_config_list: a list of AdapterConfiguration entities.
        :returns: same list with the difference that a separate prefix is prepended to
                  the name of each parameter from the adapter interface, specific to the step it
                  is in the adapter chain.

        """
        for index, adapter_config in enumerate(adapter_config_list):
            specific_prefix = ADAPTER_PREFIX_ROOT + str(index)
            self._prepend_prefix(adapter_config.interface, specific_prefix)


    def _prepend_prefix(self, input_list, prefix):
        """
        Prepend a prefix to the name of each entry form the given input tree.
        :param input_list: the adapter input tree
        :param prefix: the prefix to be added to each name
        """
        for param in input_list:
            param[ABCAdapter.KEY_NAME] = prefix + param[ABCAdapter.KEY_NAME]
            if param.get(ABCAdapter.KEY_OPTIONS) is not None:
                for option in param[ABCAdapter.KEY_OPTIONS]:
                    if option.get(ABCAdapter.KEY_ATTRIBUTES) is not None:
                        self._prepend_prefix(option[ABCAdapter.KEY_ATTRIBUTES], prefix)
            if param.get(ABCAdapter.KEY_ATTRIBUTES) is not None:
                self._prepend_prefix(param[ABCAdapter.KEY_ATTRIBUTES], prefix)


    def create_new_portlet_configuration(self, name=''):
        """
        Create a PortletConfiguration entity with the default values from the portlet
        XML declaration and the adapter input trees.
        """
        chain_adapters = self.reader.get_adapters_chain(self.algo_identifier)
        analyze_steps = []
        view_step = None

        idx = 0
        for adapter_declaration in chain_adapters:
            adapter_instance = self.build_adapter_from_declaration(adapter_declaration)
            alg_inputs = adapter_instance.flaten_input_interface()
            ###################################################################

            ### Get the overwrites defined in the portlet configuration #######
            ### for this specific adapter in the adapter chain          #######
            ### split in static and dynamic ones                        #######
            prepared_params = {KEY_STATIC: {}, KEY_DYNAMIC: {}}
            all_portlet_defined_params = self.reader.get_inputs(self.algo_identifier)
            specific_adapter_overwrites = [entry for entry in all_portlet_defined_params
                                           if ATT_OVERWRITE in entry and entry[ATT_OVERWRITE] ==
                                           adapter_declaration[ABCAdapter.KEY_NAME]]

            for entry in specific_adapter_overwrites:
                if ABCAdapter.KEY_DEFAULT in entry:
                    declared_value = entry[ABCAdapter.KEY_DEFAULT]
                elif ABCAdapter.KEY_VALUE in entry:
                    declared_value = entry[ABCAdapter.KEY_VALUE]
                else:
                    declared_value = ''
                if entry[ABCAdapter.KEY_TYPE] == KEY_DYNAMIC:
                    prepared_params[KEY_DYNAMIC][entry[ABCAdapter.KEY_NAME]] = declared_value
                else:
                    prepared_params[KEY_STATIC][entry[ABCAdapter.KEY_NAME]] = declared_value
            ###################################################################

            ### Now just fill the rest of the adapter inputs if they are not ##
            ### present in neither dynamic or static overwrites. In case of  ##
            ### sub-algorithms also add as static the algorithm : value pair ##
            for input_dict in alg_inputs:
                input_name = input_dict[ABCAdapter.KEY_NAME]
                if input_name not in prepared_params[KEY_STATIC] and input_name not in prepared_params[KEY_DYNAMIC]:
                    if ABCAdapter.KEY_DEFAULT in input_dict:
                        input_value = input_dict[ABCAdapter.KEY_DEFAULT]
                    else:
                        input_value = ''
                    prepared_params[KEY_STATIC][input_name] = input_value
            ###################################################################

            ### Now parse the dynamic inputs declared in the portlets XML ######
            ### into workflow_step specific format.                        ####
            for param_name in prepared_params[KEY_DYNAMIC]:
                new_value = self._portlet_dynamic2workflow_step(prepared_params[KEY_DYNAMIC][param_name])
                prepared_params[KEY_DYNAMIC][param_name] = new_value
            ###################################################################

            algo_id = adapter_instance.stored_adapter.id
            if idx == len(chain_adapters) - 1:
                view_step = WorkflowStepView(algorithm_id=algo_id, portlet_id=self.portlet_id,
                                             ui_name=name, static_param=prepared_params[KEY_STATIC],
                                             dynamic_param=prepared_params[KEY_DYNAMIC])
            else:
                workflow_step = WorkflowStep(algorithm_id=algo_id, static_param=prepared_params[KEY_STATIC],
                                             dynamic_param=prepared_params[KEY_DYNAMIC])
                analyze_steps.append(workflow_step)
            idx += 1
        portlet_configuration = PortletConfiguration(self.portlet_id)
        portlet_configuration.set_analyzers(analyze_steps)
        portlet_configuration.set_visualizer(view_step)
        return portlet_configuration


    @staticmethod
    def update_portlet_configuration(portlet_configuration, submited_parameters):
        """
        :param portlet_configuration: the portlet configuration that need to be updated
        :param submited_parameters: a list of parameters as submitted from the UI. This 
                                    All names are prefixed with adapter specific generated prefix.
        """
        adapter_index = 0
        relaunch_needed = False
        for analyze_step in portlet_configuration.analyzers:
            ## For each step, create the corresponding prefix and update the ##
            ## static values with those submitted from the UI. If decision to ##
            ## let the user choose the dynamic parameters too is made, here  ##
            ## would be the place to update those entries as well.            ##
            adapter_prefix = ADAPTER_PREFIX_ROOT + str(adapter_index)
            static_parameters = analyze_step.static_param
            for param_name, submited_value in submited_parameters.items():
                if param_name.startswith(adapter_prefix):
                    if str(static_parameters[param_name.replace(adapter_prefix, '')]) != str(submited_value):
                        relaunch_needed = True
                    static_parameters[param_name.replace(adapter_prefix, '')] = submited_value
            analyze_step.static_param = static_parameters
            adapter_index += 1

        visualizer_prefix = ADAPTER_PREFIX_ROOT + str(adapter_index)
        visualizer = portlet_configuration.visualizer
        static_parameters = visualizer.static_param
        for param_name, submited_value in submited_parameters.items():
            if param_name.startswith(visualizer_prefix):
                static_parameters[param_name.replace(visualizer_prefix, '')] = submited_value
        visualizer.static_param = static_parameters

        return relaunch_needed


    @staticmethod
    def clear_data_for_portlet(stored_portlet):
        """
        Remove any reference towards a given portlet already selected in a BurstConfiguration.
        """
        view_step = dao.get_configured_portlets_for_id(stored_portlet.id)
        for step in view_step:
            analizers = dao.get_workflow_steps_for_position(step.fk_workflow, step.tab_index, step.index_in_tab)
            for analyzer in analizers:
                analyzer.tab_index = None
                analyzer.index_in_tab = None
                dao.store_entity(analyzer)
            dao.remove_entity(step.__class__, step.id)
