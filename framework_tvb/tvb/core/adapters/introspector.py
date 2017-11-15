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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os
import inspect
import datetime
import tvb.core.removers_factory as removers
from types import ModuleType
from tvb.basic.logger.builder import get_logger
from tvb.basic.traits.types_mapped import MappedType
from tvb.core.entities import model
from tvb.core.entities.storage import dao, SA_SESSIONMAKER
from tvb.core.portlets.xml_reader import XMLPortletReader, ATT_OVERWRITE
from tvb.core.adapters.abcremover import ABCRemover
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.constants import ATT_TYPE, ATT_NAME, ATT_REQUIRED, ELEM_CONDITIONS, ELEM_INPUTS
from tvb.core.adapters.exceptions import XmlParserException
from tvb.core.portlets.portlet_configurer import PortletConfigurer


ALL_VARIABLE = "__all__"
LAUNCHABLE = 'launchable'
DISPLAYER = 'display'
RAWINPUT = 'rawinput'
ORDER = 'order_nr'
STATE = 'defaultdatastate'



class Introspector:
    """
    This class will handle any external module introspection.
    By introspecting other Python modules, we expect to find dynamic adapters and DataTypes.
    """


    def __init__(self, introspected_module):
        self.module_name = introspected_module
        self.logger = get_logger(self.__class__.__module__)


    def get_removers_dict(self):
        """
        Returns the removers for the datatypes of the current introspected module.
        """
        result = {'default': ABCRemover}
        for one_path in self.removers_path:
            try:
                module = __import__(one_path, globals(), locals(), ["__init__"])
                removers_ = module.REMOVERS_FACTORY
                result.update(removers_)
            except Exception:
                pass
        return result


    def introspect(self, do_create):
        """
        Introspect a given module to: 
            - create tables for custom DataType;
            - populate adapter algorithms references. 
        """
        self.logger.debug("Introspection into module:" + self.module_name)
        module = __import__(self.module_name, globals(), locals(), ["__init__"])
        try:
            path_adapters = module.ADAPTERS
            self.path_types = module.DATATYPES_PATH
            self.removers_path = module.REMOVERS_PATH
            self.path_portlets = getattr(module, 'PORTLETS_PATH', [])
        except Exception as excep:
            self.logger.warning("Module " + self.module_name + " is not fully introspect compatible!")
            self.logger.warning(excep.message)
            return

        if do_create:
            self.logger.debug("Found Datatypes_Path=" + str(self.path_types))
            # DataTypes only need to be imported for adding to DB tables
            for path in self.path_types:
                self.__get_datatypes(path)

            session = SA_SESSIONMAKER()
            model.Base.metadata.create_all(bind=session.connection())
            session.commit()
            session.close()

            self.logger.debug("Found Adapters_Dict=" + str(path_adapters))
            for category_name in path_adapters:
                category_details = path_adapters[category_name]
                launchable = bool(category_details.get(LAUNCHABLE))
                rawinput = bool(category_details.get(RAWINPUT))
                display = bool(category_details.get(DISPLAYER))
                order_nr = category_details.get(ORDER, 999)
                category_instance = dao.filter_category(category_name, rawinput, display, launchable, order_nr)
                if category_instance is not None:
                    category_instance.last_introspection_check = datetime.datetime.now()
                    category_instance.removed = False
                else:
                    category_state = category_details.get(STATE, '')
                    category_instance = model.AlgorithmCategory(category_name, launchable, rawinput, display,
                                                                category_state, order_nr, datetime.datetime.now())
                category_instance = dao.store_entity(category_instance)
                for actual_module in path_adapters[category_name]['modules']:
                    self.__read_adapters(category_instance.id, actual_module)

            for path in self.path_portlets:
                self.__get_portlets(path)
        ### Register Remover instances for current introspected module
        removers.update_dictionary(self.get_removers_dict())


    def __read_adapters(self, category_key, module_name):
        """
        Add or update lines into STORED_ADAPTERS table:
        One line for each custom class found which is extending from ABCAdapter.
        """
        for adapters_file in Introspector.__read_module_variable(module_name):
            try:
                adapters_module = __import__(module_name + "." + adapters_file, globals(), locals(), [adapters_file])
                for ad_class in dir(adapters_module):
                    ad_class = adapters_module.__dict__[ad_class]
                    if Introspector._is_concrete_subclass(ad_class, ABCAdapter):
                        if ad_class.can_be_active():
                            stored_adapter = model.Algorithm(ad_class.__module__, ad_class.__name__, category_key,
                                                                 ad_class.get_group_name(), ad_class.get_group_description(),
                                                                 ad_class.get_ui_name(), ad_class.get_ui_description(),
                                                                 ad_class.get_ui_subsection(), datetime.datetime.now())
                            adapter_inst = ad_class()
                            in_params = adapter_inst.get_input_tree()
                            req_type, param_name, flt = self.__get_required_input(in_params)
                            stored_adapter.required_datatype = req_type
                            stored_adapter.parameter_name = param_name
                            stored_adapter.datatype_filter = flt
                            stored_adapter.outputlist = str(adapter_inst.get_output())

                            inst_from_db = dao.get_algorithm_by_module(ad_class.__module__, ad_class.__name__)
                            if inst_from_db is not None:
                                stored_adapter.id = inst_from_db.id

                            stored_adapter = dao.store_entity(stored_adapter, inst_from_db is not None)
                            ad_class.stored_adapter = stored_adapter
                        else:
                            self.logger.warning("Skipped Adapter(probably because MATLAB not found):" + str(ad_class))

            except Exception:
                self.logger.exception("Could not introspect Adapters file:" + adapters_file)


    def __get_portlets(self, path_portlets):
        """
        Given a path in the form of a python package e.g.: "tvb.portlets', import
        the package, get it's folder and look for all the XML files defined 
        there, then read all the portlets defined there and store them in DB.
        """
        portlet_package = __import__(path_portlets, globals(), locals(), ["__init__"])
        portlet_folder = os.path.dirname(portlet_package.__file__)
        portlets_list = []
        for file_n in os.listdir(portlet_folder):
            try:
                if file_n.endswith('.xml'):
                    complete_file_path = os.path.join(portlet_folder, file_n)
                    portlet_reader = XMLPortletReader(complete_file_path)
                    portlet_list = portlet_reader.get_algorithms_dictionary()
                    self.logger.debug("Starting to verify currently declared portlets in %s." % (file_n,))
                    for algo_identifier in portlet_list:
                        adapters_chain = portlet_reader.get_adapters_chain(algo_identifier)
                        is_valid = True
                        for adapter in adapters_chain:
                            class_name = adapter[ABCAdapter.KEY_TYPE].split('.')[-1]
                            module_name = adapter[ABCAdapter.KEY_TYPE].replace('.' + class_name, '')
                            try:
                                #Check that module is properly declared
                                module = __import__(module_name, globals(), fromlist=[class_name])
                                if type(module) != ModuleType:
                                    is_valid = False
                                    self.logger.error("Wrong module %s in portlet %s" % (module_name, algo_identifier))
                                    continue
                                #Check that class is properly declared
                                if not hasattr(module, class_name):
                                    is_valid = False
                                    self.logger.error("Wrong class %s in portlet %s." % (class_name, algo_identifier))
                                    continue
                                #Check inputs that refers to this adapter
                                portlet_inputs = portlet_list[algo_identifier][ELEM_INPUTS]
                                adapter_instance = PortletConfigurer.build_adapter_from_declaration(adapter)
                                if adapter_instance is None:
                                    is_valid = False
                                    self.logger.warning("No group having class=%s stored for "
                                                        "portlet %s." % (class_name, algo_identifier))
                                    continue
                                adapter_input_names = [entry[ABCAdapter.KEY_NAME] for entry
                                                       in adapter_instance.flaten_input_interface()]
                                for input_entry in portlet_inputs.values():
                                    if input_entry[ATT_OVERWRITE] == adapter[ABCAdapter.KEY_NAME]:
                                        if input_entry[ABCAdapter.KEY_NAME] not in adapter_input_names:
                                            self.logger.error("Invalid input %s for adapter %s" % (
                                                input_entry[ABCAdapter.KEY_NAME], adapter_instance))
                                            is_valid = False
                            except ImportError:
                                self.logger.error("Invalid adapter declaration %s in portlet %s" % (
                                                  adapter[ABCAdapter.KEY_TYPE], algo_identifier))
                                is_valid = False
                        if is_valid:
                            portlets_list.append(model.Portlet(algo_identifier, complete_file_path,
                                                               portlet_list[algo_identifier]['name']))
            except XmlParserException as excep:
                self.logger.exception(excep)
                self.logger.error("Invalid Portlet description File " + file_n + " will continue without it!!")

        self.logger.debug("Refreshing portlets from xml declarations.")
        stored_portlets = dao.get_available_portlets()
        #First update old portlets from DB
        for stored_portlet in stored_portlets:
            for verified_portlet in portlets_list:
                if stored_portlet.algorithm_identifier == verified_portlet.algorithm_identifier:
                    stored_portlet.xml_path = verified_portlet.xml_path
                    stored_portlet.last_introspection_check = datetime.datetime.now()
                    stored_portlet.name = verified_portlet.name
                    dao.store_entity(stored_portlet)
                    break

        #Now add portlets that were not in DB at previous run but are valid now
        for portlet in portlets_list:
            db_entity = dao.get_portlet_by_identifier(portlet.algorithm_identifier)
            if db_entity is None:
                self.logger.debug("Will now store portlet %s" % (str(portlet),))
                dao.store_entity(portlet)

    @staticmethod
    def _is_concrete_subclass(clz, super_cls):
        return inspect.isclass(clz) and not inspect.isabstract(clz) and issubclass(clz, super_cls)


    def __get_datatypes(self, path_types):
        """
        Imports each DataType to update the DB model, by creating a new table for each DataType.
        """
        for my_type in Introspector.__read_module_variable(path_types):
            try:
                module_ref = __import__(path_types, globals(), locals(), [my_type])
                module_ref = getattr(module_ref, my_type)
                tree = inspect.getmembers(module_ref, lambda c: self._is_concrete_subclass(c, MappedType))
                for class_name, class_ref in tree:
                    self.logger.debug("Importing class for DB table to be created: " + class_name)
            except Exception as excep1:
                self.logger.error('Could not import DataType!' + my_type)
                self.logger.exception(excep1)
        self.logger.debug('DB Model update finished for ' + path_types)


    def __get_class_ref(self, full_class_name):
        """
        Given the full name of a class as a string this method
        will return a reference to that class.
        """
        if '.' in full_class_name:
            module, class_name = full_class_name.rsplit('.', 1)
            mod = __import__(module, fromlist=[class_name])
            return getattr(mod, class_name)
        self.logger.error("The location of the adapter class is incorrect. It should be placed in a module.")
        raise Exception("The location of the adapter class is incorrect. It should be placed in a module.")


    def __get_required_input(self, input_tree):
        """
        Checks in the input interface for required fields of a type
        that extends from DataType.
        """
        required_datatypes = []
        req_param_name = []
        req_param_filters = []
        all_datatypes = []
        all_param_name = []
        all_param_filters = []
        for input_field in input_tree:
            if ATT_TYPE not in input_field:
                continue
            class_name = self.__get_classname(input_field[ATT_TYPE])
            if class_name is not None and not inspect.isabstract(class_name):
                all_datatypes.append(class_name.__name__)
                all_param_name.append(input_field[ATT_NAME])
                if ELEM_CONDITIONS in input_field:
                    j = input_field[ELEM_CONDITIONS].to_json()
                    all_param_filters.append(j)
                else:
                    all_param_filters.append('None')
                if input_field.get(ATT_REQUIRED):
                    required_datatypes.append(class_name.__name__)
                    req_param_name.append(input_field[ATT_NAME])
                    if ELEM_CONDITIONS in input_field:
                        j = input_field[ELEM_CONDITIONS].to_json()
                        req_param_filters.append(j)
                    else:
                        req_param_filters.append('None')
        if (not required_datatypes and len(all_datatypes) > 1) or len(required_datatypes) > 1:
            return None, None, None
        elif len(required_datatypes) == 1:
            return required_datatypes[0], req_param_name[0], req_param_filters[0]
        elif len(all_datatypes) == 1:
            return all_datatypes[0], all_param_name[0], all_param_filters[0]
        return None, None, None


    def __get_classname(self, param_to_test):
        """
        Check if param_to_test is either a class or a string that points
        to a class. Returns the class or None.
        """
        if inspect.isclass(param_to_test):
            return param_to_test
        if param_to_test in ABCAdapter.STATIC_ACCEPTED_TYPES:
            return None
        try:
            module, class_name = param_to_test.rsplit('.', 1)
            reference = __import__(module, globals(), locals(), [class_name])
            return getattr(reference, class_name)
        except Exception as excep:
            self.logger.debug("Could not import class:" + str(excep))
            return None


    @staticmethod
    def __read_module_variable(module_path, variable_name=ALL_VARIABLE):
        """
        Retrieve variable with name 'variable_name' from the given Python module.
        Result will be a list of Strings.
        """
        module = __import__(module_path, globals(), locals(), ["__init__"])
        return [str(a) for a in getattr(module, variable_name, [])]
