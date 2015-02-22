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
Service layer, for executing computational steps in the application.
Code related to launching/duplicating operations is placed here.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from copy import copy
from tvb.basic.traits.exceptions import TVBException
from tvb.basic.filters.chain import FilterChain
from tvb.basic.logger.builder import get_logger
from tvb.core import utils
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.exceptions import IntrospectionException
from tvb.core.services.exceptions import OperationException
from tvb.core.services.operation_service import OperationService
from tvb.core.portlets.xml_reader import KEY_DYNAMIC



class FlowService:
    """
    Service Layer for all TVB generic Work-Flow operations.
    """

    MAXIMUM_DATA_TYPES_DISPLAYED = 50
    KEY_WARNING = "warning"
    WARNING_OVERFLOW = "Too many entities in storage; some of them were not returned, to avoid overcrowding. " \
                       "Use filters, to make the list small enough to fit in here!"


    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.file_helper = FilesHelper()
    
    
    def get_category_by_id(self, identifier):
        """ Pass to DAO the retrieve of category by ID operation."""
        try:
            return dao.get_category_by_id(identifier)
        except Exception, excep:
            self.logger.warning("Wrong step!")
            self.logger.exception(excep)
            raise OperationException(excep.message)
    
    
    @staticmethod
    def get_uploader_categories():
        """Retrieve all algorithm categories with Upload mechanism"""
        return dao.get_uploader_categories()
    
    @staticmethod
    def get_raw_categories():
        """:returns: AlgorithmCategory list of entities that have results in RAW state (Creators/Uploaders)"""
        return dao.get_raw_categories()
    
    @staticmethod
    def get_visualisers_category():
        """Retrieve all Algorithm categories, with display capability"""
        result = dao.get_visualisers_categories()
        if result is None or len(result) < 1:
            raise Exception("View Category not found!!!")
        return result[0]
    
    
    @staticmethod
    def get_launchable_non_viewers():
        """Retrieve all Algorithm categories, with display capability"""
        result = dao.get_launchable_categories(elimin_viewers=True)
        if result is None or len(result) < 1:
            raise Exception("Analyze Category not found!!!")
        return result[0]
    
    
    @staticmethod
    def get_groups_for_categories(categories):
        """
        Retrieve the list of all Adapters names from a  given Category
        """
        categories_ids = [categ.id for categ in categories]
        return dao.get_groups_by_categories(categories_ids)
    
    
    def get_algorithm_by_identifier(self, ident):
        """
        Retrieve Algorithm entity by ID.
        Return None, if ID is not found in DB.
        """
        try:
            return dao.get_algorithm_by_id(ident)
        except Exception, excep:
            self.logger.exception(excep)
            return None
    
    
    def get_algo_group_by_identifier(self, ident):
        """
        Retrieve Algorithm Group entity by ID.
        Return None, if ID is not found in DB.
        """
        try:
            return dao.get_algo_group_by_id(ident)
        except Exception, excep:
            self.logger.exception(excep)
            return None
        
    
    @staticmethod
    def load_operation(operation_id):
        """ Retrieve previously stored Operation from DB, and load operation.burst attribute"""
        operation = dao.get_operation_by_id(operation_id)
        operation.burst = dao.get_burst_for_operation_id(operation_id)
        return operation


    @staticmethod
    def get_operation_numbers(proj_id):
        """ Count total number of operations started for current project. """
        return dao.get_operation_numbers(proj_id)
    
    
    def build_adapter_instance(self, group):
        """
        Having a module and a class name, create an instance of ABCAdapter.
        """
        try:
            return ABCAdapter.build_adapter(group)
        except IntrospectionException, excep:
            if group is None:
                self.logger.error('The given algorithm group is None.')
                self.logger.exception(excep)
                raise OperationException("Could not prepare the algo- group.")
            self.logger.error('Not found: ' + group.classname + ' in:' + group.module)
            self.logger.exception(excep)
            raise OperationException("Could not prepare " + group.classname)
              
        
    def prepare_adapter(self, project_id, algo_group):
        """
        Having a given Adapter, specified by Module and ClassName, 
        create a instance of it and return the instance.
        Actually return a Tuple: Adapter Instance, Dictionary for Adapter 
        Interface specification.
        """
        adapter_module = algo_group.module.replace('-', '.')
        adapter_name = algo_group.classname
        try:
            # Prepare Adapter Interface, by populating with existent data,
            # in case of a parameter of type DataType.
            group = dao.find_group(adapter_module, adapter_name, algo_group.init_parameter)
            adapter_instance = self.build_adapter_instance(group)
            interface = adapter_instance.get_input_tree()
            interface = self.prepare_parameters(interface, project_id, group.fk_category)
            interface = ABCAdapter.prepare_param_names(interface)
            return group, interface
        except Exception, excep:
            self.logger.exception(excep)
            self.logger.error('Not found:' + adapter_name + ' in:' + adapter_module)
            raise OperationException("Could not prepare " + adapter_name)
    
    
    @staticmethod
    def get_algorithm_by_module_and_class(module, classname):
        """
        Get the db entry from the algorithm table for the given module and 
        class.
        """
        group = dao.find_group(module, classname)
        algo = dao.get_algorithm_by_group(group.id)
        return algo, group
    
    
    def get_available_datatypes(self, project_id, data_name, filters=None):
        """
        Return all dataTypes that match a given name and some filters.
        """
        data_class = FilterChain._get_class_instance(data_name)
        if data_class is None:
            self.logger.warning("Invalid Class specification:" + str(data_name))
            return [], 0
        else:
            self.logger.debug('Filtering:' + str(data_class))
            return dao.get_values_of_datatype(project_id, data_class, filters, self.MAXIMUM_DATA_TYPES_DISPLAYED)
        
    
    @staticmethod
    def populate_values(data_list, type_, category_key, complex_dt_attributes=None):
        """
        Populate meta-data fields for data_list (list of DataTypes).
        """
        values = []
        all_field_values = ''
        for value in data_list:
            # Here we only populate with DB data, actual
            # XML check will be done after select and submit.
            entity_gid = value[2]
            actual_entity = dao.get_generic_entity(type_, entity_gid, "gid")
            display_name = ''
            if actual_entity is not None and len(actual_entity) > 0 and isinstance(actual_entity[0], model.DataType):
                display_name = actual_entity[0].display_name
            display_name = display_name + ' - ' + (value[3] or "None ")
            if value[5]:
                display_name = display_name + ' - From: ' + str(value[5])
            else:
                display_name = display_name + utils.date2string(value[4])
            if value[6]:
                display_name = display_name + ' - ' + str(value[6])
            display_name = display_name + ' - ID:' + str(value[0])
            all_field_values = all_field_values + str(entity_gid) + ','
            values.append({ABCAdapter.KEY_NAME: display_name, ABCAdapter.KEY_VALUE: entity_gid})
            if complex_dt_attributes is not None:
                ### TODO apply filter on sub-attributes
                values[-1][ABCAdapter.KEY_ATTRIBUTES] = complex_dt_attributes
        if category_key is not None:
            category = dao.get_category_by_id(category_key)
            if (not category.display) and (not category.rawinput) and len(data_list) > 1:
                values.insert(0, {ABCAdapter.KEY_NAME: "All", ABCAdapter.KEY_VALUE: all_field_values[:-1]})
        return values
     
     
    def prepare_parameters(self, attributes_list, project_id, category_key):
        """
        Private method, to be called recursively.
        It will receive a list of Attributes, and it will populate 'options'
        entry with data references from DB.
        """
        result = []
        for param in attributes_list:
            if param.get(ABCAdapter.KEY_UI_HIDE):
                continue
            transformed_param = copy(param)

            if (ABCAdapter.KEY_TYPE in param) and not (param[ABCAdapter.KEY_TYPE] in ABCAdapter.STATIC_ACCEPTED_TYPES):

                if ABCAdapter.KEY_CONDITION in param:
                    filter_condition = param[ABCAdapter.KEY_CONDITION]
                else:
                    filter_condition = FilterChain('')
                filter_condition.add_condition(FilterChain.datatype + ".visible", "==", True)

                data_list, total_count = self.get_available_datatypes(project_id, param[ABCAdapter.KEY_TYPE],
                                                                      filter_condition)

                if total_count > self.MAXIMUM_DATA_TYPES_DISPLAYED:
                    transformed_param[self.KEY_WARNING] = self.WARNING_OVERFLOW

                complex_dt_attributes = None
                if param.get(ABCAdapter.KEY_ATTRIBUTES):
                    complex_dt_attributes = self.prepare_parameters(param[ABCAdapter.KEY_ATTRIBUTES], 
                                                                    project_id, category_key)
                values = self.populate_values(data_list, param[ABCAdapter.KEY_TYPE], 
                                              category_key, complex_dt_attributes)
                
                if (transformed_param.get(ABCAdapter.KEY_REQUIRED) and len(values) > 0 and
                        transformed_param.get(ABCAdapter.KEY_DEFAULT) in [None, 'None']):
                    transformed_param[ABCAdapter.KEY_DEFAULT] = str(values[-1][ABCAdapter.KEY_VALUE])
                transformed_param[ABCAdapter.KEY_FILTERABLE] = FilterChain.get_filters_for_type(
                    param[ABCAdapter.KEY_TYPE])
                transformed_param[ABCAdapter.KEY_TYPE] = ABCAdapter.TYPE_SELECT
                # If Portlet dynamic parameter, don't add the options instead
                # just add the default value. 
                if KEY_DYNAMIC in param:
                    dynamic_param = {ABCAdapter.KEY_NAME: param[ABCAdapter.KEY_DEFAULT],
                                     ABCAdapter.KEY_VALUE: param[ABCAdapter.KEY_DEFAULT]}
                    transformed_param[ABCAdapter.KEY_OPTIONS] = [dynamic_param]
                else:
                    transformed_param[ABCAdapter.KEY_OPTIONS] = values
                if type(param[ABCAdapter.KEY_TYPE]) == str:
                    transformed_param[ABCAdapter.KEY_DATATYPE] = param[ABCAdapter.KEY_TYPE]
                else:
                    data_type = param[ABCAdapter.KEY_TYPE]
                    transformed_param[ABCAdapter.KEY_DATATYPE] = data_type.__module__ + '.' + data_type.__name__
            
                ### DataType-attributes are no longer necessary, they were already copied on each OPTION
                transformed_param[ABCAdapter.KEY_ATTRIBUTES] = []
                
            else:
                if param.get(ABCAdapter.KEY_OPTIONS) is not None:
                    transformed_param[ABCAdapter.KEY_OPTIONS] = self.prepare_parameters(param[ABCAdapter.KEY_OPTIONS],
                                                                                        project_id, category_key)
                    if (transformed_param.get(ABCAdapter.KEY_REQUIRED) and
                            len(param[ABCAdapter.KEY_OPTIONS]) > 0 and
                            (transformed_param.get(ABCAdapter.KEY_DEFAULT) in [None, 'None'])):
                        def_val = str(param[ABCAdapter.KEY_OPTIONS][-1][ABCAdapter.KEY_VALUE])
                        transformed_param[ABCAdapter.KEY_DEFAULT] = def_val
                    
                if param.get(ABCAdapter.KEY_ATTRIBUTES) is not None:
                    transformed_param[ABCAdapter.KEY_ATTRIBUTES] = self.prepare_parameters(
                        param[ABCAdapter.KEY_ATTRIBUTES], project_id, category_key)
            result.append(transformed_param)   
        return result
    
    
    @staticmethod
    def create_link(data_ids, project_id):
        """
        For a list of dataType IDs and a project id create all the required links.
        """
        for data in data_ids:
            link = model.Links(data, project_id)
            dao.store_entity(link)


    @staticmethod
    def remove_link(dt_id, project_id):
        """
        Remove the link from the datatype given by dt_id to project given by project_id.
        """
        link = dao.get_link(dt_id, project_id)
        if link is not None:
            dao.remove_entity(model.Links, link.id)
    
        
    def fire_operation(self, adapter_instance, current_user, project_id,  
                       method_name=ABCAdapter.LAUNCH_METHOD, visible=True, **data):
        """
        Launch an operation, specified by AdapterInstance, for CurrentUser, 
        Current Project and a given set of UI Input Data.
        """
        operation_name = str(adapter_instance.__class__.__name__) + "." + method_name
        try:
            self.logger.info("Starting operation " + operation_name)
            project = dao.get_project_by_id(project_id)
            tmp_folder = self.file_helper.get_project_folder(project, self.file_helper.TEMP_FOLDER)
            
            result = OperationService().initiate_operation(current_user, project.id, adapter_instance, 
                                                           tmp_folder, method_name, visible, **data)
            self.logger.info("Finished operation:" + operation_name)
            return result

        except TVBException, excep:
            self.logger.exception("Could not launch operation " + operation_name +
                                  " with the given set of input data, because: " + excep.message)
            raise OperationException(excep.message, excep)
        except Exception, excep:
            self.logger.exception("Could not launch operation " + operation_name + " with the given set of input data!")
            raise OperationException(str(excep))      

    
    ##########################################################################
    ######## Methods below are for MeasurePoint selections ###################
    ##########################################################################
    
    @staticmethod
    def get_selections_for_project(project_id, datatype_gid):
        """
        Retrieved from DB saved selections for current project. If a certain selection
        doesn't have all the labels between the labels of the given connectivity than
        this selection will not be returned.
        :returns: List of ConnectivitySelection entities.
        """
        return dao.get_selections_for_project(project_id, datatype_gid)
    
    
    @staticmethod
    def save_measure_points_selection(ui_name, selected_nodes, datatype_gid, project_id):
        """
        Store in DB a ConnectivitySelection.
        """
        select_entities = dao.get_selections_for_project(project_id, datatype_gid, ui_name)

        if select_entities:
            # when the name of the new selection is within the available selections then update that selection:
            select_entity = select_entities[0]
            select_entity.selected_nodes = selected_nodes
        else:
            select_entity = model.MeasurePointsSelection(ui_name, selected_nodes, datatype_gid, project_id)

        dao.store_entity(select_entity)


    @staticmethod
    def get_generic_entity(entity_type, filter_value, select_field):
        return dao.get_generic_entity(entity_type, filter_value, select_field)