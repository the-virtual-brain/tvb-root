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

from tvb.basic.traits.exceptions import TVBException
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.input_tree import InputTreeManager
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.exceptions import IntrospectionException
from tvb.core.services.exceptions import OperationException
from tvb.core.services.operation_service import OperationService



class FlowService:
    """
    Service Layer for all TVB generic Work-Flow operations.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.file_helper = FilesHelper()
        self.input_tree_manager = InputTreeManager()
    
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
        """Retrieve the Analyze Algorithm Category, (first category with launch capability which is not Viewers)"""
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
        return self.input_tree_manager._get_available_datatypes(project_id, data_name, filters)


    def prepare_parameters(self, attributes_list, project_id, category_key):
        """
        For a datatype node in the input tree, load all instances from the db that fit the filters.
        """
        return self.input_tree_manager.fill_input_tree_with_options(attributes_list, project_id, category_key)
    
    
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
    
        
    def fire_operation(self, adapter_instance, current_user, project_id, visible=True, **data):
        """
        Launch an operation, specified by AdapterInstance, for CurrentUser, 
        Current Project and a given set of UI Input Data.
        """
        operation_name = str(adapter_instance.__class__.__name__)
        try:
            self.logger.info("Starting operation " + operation_name)
            project = dao.get_project_by_id(project_id)
            tmp_folder = self.file_helper.get_project_folder(project, self.file_helper.TEMP_FOLDER)
            
            result = OperationService().initiate_operation(current_user, project.id, adapter_instance, 
                                                           tmp_folder, visible, **data)
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