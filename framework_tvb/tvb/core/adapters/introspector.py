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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os
import inspect
import datetime
import tvb.core.removers_factory as removers
from types import ModuleType
from tvb.basic.profile import TvbProfile
from tvb.basic.logger.builder import get_logger
from tvb.basic.traits.types_mapped import MappedType
from tvb.core.entities import model
from tvb.core.entities.storage import dao, SA_SESSIONMAKER
from tvb.core.portlets.xml_reader import XMLPortletReader, ATT_OVERWRITE
from tvb.core.adapters.abcremover import ABCRemover
from tvb.core.adapters.abcadapter import ABCAdapter, ABCGroupAdapter
from tvb.core.adapters.xml_reader import ATT_TYPE, ATT_NAME, INPUTS_KEY
from tvb.core.adapters.xml_reader import ATT_REQUIRED, ELEM_CONDITIONS, XMLGroupReader
from tvb.core.adapters.exceptions import XmlParserException
from tvb.core.portlets.portlet_configurer import PortletConfigurer
from tvb.core.utils import extract_matlab_doc_string


ALL_VARIABLE = "__all__"
XML_FOLDERS_VARIABLE = "__xml_folders__"
MATLAB_ADAPTER = "MatlabAdapter"
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


    def get_events_path(self):
        """
        Returns the EVENTS_FOLDER variable for a given module.
        """
        module = __import__(self.module_name, globals(), locals(), ["__init__"])
        try:
            event_path = __import__(module.EVENTS_FOLDER, globals(), locals(), ["__init__"])
            return os.path.dirname(event_path.__file__)
        except Exception, exception:
            self.logger.warning("Could not import events folder.\n" + str(exception))
            return None


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
        except Exception, excep:
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
                launchable = category_details.get(LAUNCHABLE)
                rawinput = category_details.get(RAWINPUT)
                display = category_details.get(DISPLAYER)
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
                    self.__populate_algorithms(category_instance.id, actual_module)

            for path in self.path_portlets:
                self.__get_portlets(path)
        ### Register Remover instances for current introspected module
        removers.update_dictionary(self.get_removers_dict())


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
                    portlet_reader = XMLPortletReader.get_instance(complete_file_path)
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
                                portlet_inputs = portlet_list[algo_identifier][INPUTS_KEY]
                                adapter_instance, _ = PortletConfigurer.build_adapter_from_declaration(adapter)
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
                            except ImportError, _:
                                self.logger.error("Invalid adapter declaration %s in portlet %s" % (
                                                  adapter[ABCAdapter.KEY_TYPE], algo_identifier))
                                is_valid = False
                        if is_valid:
                            portlets_list.append(model.Portlet(algo_identifier, complete_file_path,
                                                               portlet_list[algo_identifier]['name']))
            except XmlParserException, excep:
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


    def __get_datatypes(self, path_types):
        """
        Imports each DataType to update the DB model, by creating a new table for each DataType.
        """
        for my_type in Introspector.__get_variable(path_types):
            try:
                module_ref = __import__(path_types, globals(), locals(), [my_type])
                module_ref = eval("module_ref." + my_type)
                tree = [module_ref.__dict__[j] for j in [i for i in dir(module_ref)
                          if (inspect.isclass(module_ref.__dict__[i]) and not inspect.isabstract(module_ref.__dict__[i])
                             and issubclass(module_ref.__dict__[i], MappedType))]]
                for class_ref in tree:
                    self.logger.debug("Importing class for DB table to be created: " + str(class_ref.__name__))
            except Exception, excep1:
                self.logger.error('Could not import DataType!' + my_type)
                self.logger.exception(excep1)
        self.logger.debug('DB Model update finished for ' + path_types)


    def __populate_algorithms(self, category_key, module_name):
        """
        Add lines to ALGORITHMS table, 
        one line for each custom class found extending from ABCAdapter.
        """
        groups = []
        for adapter_file in Introspector.__get_variable(module_name):
            try:
                adapter = __import__(module_name, globals(), locals(), [adapter_file])
                adapter = eval("adapter." + adapter_file)
                tree = [adapter.__dict__[j] for j in [i for i in dir(adapter)
                           if (inspect.isclass(adapter.__dict__[i]) and not inspect.isabstract(adapter.__dict__[i])
                               and issubclass(adapter.__dict__[i], ABCAdapter))]]
                for class_ref in tree:
                    group = self.__create_instance(category_key, class_ref)
                    if group is not None:
                        groups.append(group)
            except Exception, excep:
                self.logger.error("Could not introspect Adapters file:" + adapter_file)
                self.logger.exception(excep)

        xml_folders = Introspector.__get_variable(module_name, XML_FOLDERS_VARIABLE)
        for folder in xml_folders:
            folder_path = os.path.join(TvbProfile.current.web.CURRENT_DIR, folder)
            files = os.listdir(folder_path)
            for file_ in files:
                if file_.endswith(".xml"):
                    try:
                        reader = XMLGroupReader.get_instance(os.path.join(folder_path, file_))
                        adapter_class = reader.get_type()
                        class_ref = self.__get_class_ref(adapter_class)
                        group = self.__create_instance(category_key, class_ref, os.path.join(folder, file_))
                        if group is not None:
                            group.displayname = reader.get_ui_name()
                            group.subsection_name = reader.subsection_name
                            group.description = reader.get_ui_description()
                            groups.append(group)
                    except XmlParserException, excep:
                        self.logger.error("Could not parse XML file: " + os.path.join(folder_path, file_))
                        self.logger.exception(excep)

        # Set the last_introspection_check flag so they will pass the validation check done after introspection
        self.__update_references_last_check_timestamp(groups, category_key)

        for group in groups:
            group_inst_from_db = dao.find_group(group.module, group.classname, group.init_parameter)
            adapter = ABCAdapter.build_adapter(group)
            has_sub_algorithms = False
            ui_name = group.displayname
            if isinstance(adapter, ABCGroupAdapter):
                group.algorithm_param_name = adapter.get_algorithm_param()
                has_sub_algorithms = True
            if group_inst_from_db is None:
                self.logger.info(str(group.module) + " will be stored new in DB")
                group = model.AlgorithmGroup(group.module, group.classname, category_key,
                                             group.algorithm_param_name, group.init_parameter,
                                             datetime.datetime.now(), subsection_name=group.subsection_name,
                                             description=group.description)
            else:
                self.logger.info(str(group.module) + " will be updated")
                group = group_inst_from_db
            if hasattr(adapter, "_ui_name"):
                ui_name = adapter._ui_name
            elif ui_name is None or len(ui_name) == 0:
                ui_name = group.classname
            if hasattr(adapter, "_ui_description"):
                group.description = adapter._ui_description
            if hasattr(adapter, "_ui_subsection"):
                group.subsection_name = adapter._ui_subsection
            group.displayname = ui_name
            group.removed = False
            group.last_introspection_check = datetime.datetime.now()
            group_inst_from_db = dao.store_entity(group)
            self.__store_algorithms_for_group(group_inst_from_db, adapter, has_sub_algorithms)


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


    def __store_algorithms_for_group(self, group, adapter, has_sub_algorithms):
        """
        For the group passed as parameter do the following:
        If it has sub-algorithms, get the list of them, add sub-algorithm 
        references into the DB with all the required fields.
        If it is not a GroupAdapter add a single algorithm into the DB with an
        empty identifier.
        """
        if has_sub_algorithms:
            algos = adapter.get_algorithms_dictionary()
            for algo_ident in algos:
                in_params = adapter.get_input_for_algorithm(algo_ident)
                req_type, param_name, flt = self.__get_required_input(in_params)
                outputs = adapter.get_output_for_algorithm(algo_ident)
                algo_description = ""
                if self.__is_matlab_parent(adapter.__class__):
                    root_folder = adapter.get_matlab_file_root()
                    file_name = adapter.get_matlab_file(algo_ident)
                    if file_name:
                        algo_description = extract_matlab_doc_string(os.path.join(root_folder, file_name))
                algorithm = dao.get_algorithm_by_group(group.id, algo_ident)
                if algorithm is None:
                    #Create new
                    algorithm = model.Algorithm(group.id, algo_ident, algos[algo_ident][ATT_NAME],
                                                req_type, param_name, str(outputs), flt, description=algo_description)
                else:
                    #Edit previous
                    algorithm.name = algos[algo_ident][ATT_NAME]
                    algorithm.required_datatype = req_type
                    algorithm.parameter_name = param_name
                    algorithm.outputlist = str(outputs)
                    algorithm.datatype_filter = flt
                    algorithm.description = algo_description
                dao.store_entity(algorithm)
        else:
            input_tree = adapter.get_input_tree()
            req_type, param_name, flt = self.__get_required_input(input_tree)
            outputs = str(adapter.get_output())
            algorithm = dao.get_algorithm_by_group(group.id, None)
            if hasattr(adapter, '_ui_name'):
                algo_name = adapter._ui_name
            else:
                algo_name = adapter.__class__.__name__
            if algorithm is None:
                #Create new
                algorithm = model.Algorithm(group.id, None, algo_name, req_type, param_name, outputs, flt)
            else:
                #Edit previous
                algorithm.name = algo_name
                algorithm.required_datatype = req_type
                algorithm.parameter_name = param_name
                algorithm.outputlist = str(outputs)
                algorithm.datatype_filter = flt
            dao.store_entity(algorithm)


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
        except Exception, excep:
            self.logger.debug("Could not import class:" + str(excep))
            return None


    @staticmethod
    def __update_references_last_check_timestamp(current_groups, category_key):
        """
        For all the algorithms found in the db for category given by category_key, update their
        'last_introspection_check' timestamp if they are part of current_groups.
        
        :param current_groups: a list of algorithm groups that recently passed our validations
            from the introspection and are still valid
        :param category_key: the algorithm group category for which we match the algorithms found
            in the db with those we recently validated on introspection
        """
        db_groups = dao.get_groups_by_categories([category_key], filter_removed=False)
        for group in db_groups:
            for curr_group in current_groups:
                if group.module == curr_group.module and group.classname == curr_group.classname:
                    group.last_introspection_check = datetime.datetime.now()
                    dao.store_entity(group)
                    break


    def __create_instance(self, category_key, class_ref, init_parameter=None):
        """
        Validate Class reference.
        Return None or Algorithm instance, from class reference.
        """
        if class_ref.can_be_active():
            return model.AlgorithmGroup(class_ref.__module__, class_ref.__name__, category_key,
                                        init_parameter=init_parameter, last_introspection_check=datetime.datetime.now())
        else:
            self.logger.warning("Skipped Adapter(probably because MATLAB not found):" + str(class_ref))
            return None


    def __is_matlab_parent(self, adapter_class):
        """ Check if current class is MatlabAdapter"""
        return adapter_class.__name__.find(MATLAB_ADAPTER) >= 0


    @staticmethod
    def __get_variable(module_path, variable_name=ALL_VARIABLE):
        """
        Retrieve variable with name 'variable_name' from the given Python module.
        Result will be a list of Strings.
        """
        module = __import__(module_path, globals(), locals(), ["__init__"])
        return [str(a) for a in getattr(module, variable_name, [])]
