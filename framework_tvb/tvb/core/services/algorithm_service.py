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
Service layer, for executing computational steps in the application.
Code related to launching/duplicating operations is placed here.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from inspect import getmro
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.filters.chain import FilterChain, InvalidFilterChainInput
from tvb.core.entities.model.model_datatype import *
from tvb.core.entities.model.model_operation import AlgorithmTransientGroup
from tvb.core.entities.storage import dao
from tvb.core.services.exceptions import OperationException


class AlgorithmService(object):
    """
    Service Layer for Algorithms manipulation (e.g. find all Uploaders, Filter algo by category, etc)
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.file_helper = FilesHelper()

    @staticmethod
    def get_category_by_id(identifier):
        """ Pass to DAO the retrieve of category by ID operation."""
        return dao.get_category_by_id(identifier)

    @staticmethod
    def get_raw_categories():
        """:returns: AlgorithmCategory list of entities that have results in RAW state (Creators/Uploaders)"""
        return dao.get_raw_categories()

    @staticmethod
    def get_visualisers_category():
        """Retrieve all Algorithm categories, with display capability"""
        result = dao.get_visualisers_categories()
        if not result:
            raise ValueError("View Category not found!!!")
        return result[0]

    @staticmethod
    def get_algorithm_by_identifier(ident):
        """
        Retrieve Algorithm entity by ID.
        Return None, if ID is not found in DB.
        """
        return dao.get_algorithm_by_id(ident)

    @staticmethod
    def get_operation_numbers(proj_id):
        """ Count total number of operations started for current project. """
        return dao.get_operation_numbers(proj_id)

    def prepare_adapter_form(self, adapter_instance, project_id):
        form = adapter_instance.get_form()(project_id=project_id)
        try:
            dt = form.get_traited_datatype()
            if dt is not None:
                form.fill_from_trait(dt)
        except NotImplementedError:
            self.logger.info('This form does not take defaults from a HasTraits entity')

        return form

    def prepare_adapter(self, stored_adapter):

        adapter_module = stored_adapter.module
        adapter_name = stored_adapter.classname
        try:
            # Prepare Adapter Interface, by populating with existent data,
            # in case of a parameter of type DataType.
            adapter_instance = ABCAdapter.build_adapter(stored_adapter)
            return adapter_instance
        except Exception:
            self.logger.exception('Not found:' + adapter_name + ' in:' + adapter_module)
            raise OperationException("Could not prepare " + adapter_name)

    @staticmethod
    def get_algorithm_by_module_and_class(module, classname):
        """
        Get the db entry from the algorithm table for the given module and 
        class.
        """
        return dao.get_algorithm_by_module(module, classname)

    @staticmethod
    def create_link(data_ids, project_id):
        """
        For a list of dataType IDs and a project id create all the required links.
        """
        for data in data_ids:
            link = Links(data, project_id)
            dao.store_entity(link)

    @staticmethod
    def remove_link(dt_id, project_id):
        """
        Remove the link from the datatype given by dt_id to project given by project_id.
        """
        link = dao.get_link(dt_id, project_id)
        if link is not None:
            dao.remove_entity(Links, link.id)

    @staticmethod
    def get_upload_algorithms():
        """
        :return: List of StoredAdapter entities
        """
        categories = dao.get_uploader_categories()
        categories_ids = [categ.id for categ in categories]
        return dao.get_adapters_from_categories(categories_ids)

    @staticmethod
    def get_analyze_groups():
        """
        :return: list of AlgorithmTransientGroup entities
        """
        categories = dao.get_launchable_categories(elimin_viewers=True)
        categories_ids = [categ.id for categ in categories]
        stored_adapters = dao.get_adapters_from_categories(categories_ids)

        groups_list = []
        for adapter in stored_adapters:
            # For empty groups, this time, we fill the actual adapter
            group = AlgorithmTransientGroup(adapter.group_name or adapter.displayname,
                                            adapter.group_description or adapter.description)
            group = AlgorithmService._find_group(groups_list, group)
            group.children.append(adapter)
        return categories[0], groups_list

    @staticmethod
    def _find_group(groups_list, new_group):
        for i in range(len(groups_list) - 1, -1, -1):
            current_group = groups_list[i]
            if current_group.name == new_group.name and current_group.description == new_group.description:
                return current_group
        # Not found in list
        groups_list.append(new_group)
        return new_group

    def get_visualizers_for_group(self, dt_group_gid):

        categories = dao.get_visualisers_categories()
        return self._get_launchable_algorithms(dt_group_gid, categories)[1]

    def get_launchable_algorithms(self, datatype_gid):
        """
        :param datatype_gid: Filter only algorithms compatible with this GUID
        :return: dict(category_name: List AlgorithmTransientGroup)
        """
        categories = dao.get_launchable_categories()
        datatype_instance, filtered_adapters, has_operations_warning = self._get_launchable_algorithms(datatype_gid,
                                                                                                       categories)

        if isinstance(datatype_instance, DataTypeGroup):
            # If part of a group, update also with specific analyzers of the child datatype
            dt_group = dao.get_datatype_group_by_gid(datatype_gid)
            datatypes = dao.get_datatypes_from_datatype_group(dt_group.id)
            if len(datatypes):
                datatype = datatypes[-1]
                analyze_category = dao.get_launchable_categories(True)
                _, inner_analyzers, _ = self._get_launchable_algorithms(datatype.gid, analyze_category)
                filtered_adapters.extend(inner_analyzers)

        categories_dict = dict()
        for c in categories:
            categories_dict[c.id] = c.displayname

        return self._group_adapters_by_category(filtered_adapters, categories_dict), has_operations_warning

    def _get_launchable_algorithms(self, datatype_gid, categories):
        datatype_instance = dao.get_datatype_by_gid(datatype_gid)
        return self.get_launchable_algorithms_for_datatype(datatype_instance, categories)

    def get_launchable_algorithms_for_datatype(self, datatype, categories):
        data_class = datatype.__class__
        all_compatible_classes = [data_class.__name__]
        for one_class in getmro(data_class):
            # from tvb.basic.traits.types_mapped import MappedType

            if issubclass(one_class, DataType) and one_class.__name__ not in all_compatible_classes:
                all_compatible_classes.append(one_class.__name__)

        self.logger.debug("Searching in categories: " + str(categories) + " for classes " + str(all_compatible_classes))
        categories_ids = [categ.id for categ in categories]
        launchable_adapters = dao.get_applicable_adapters(all_compatible_classes, categories_ids)

        filtered_adapters = []
        has_operations_warning = False
        for stored_adapter in launchable_adapters:
            filter_chain = FilterChain.from_json(stored_adapter.datatype_filter)
            try:
                if not filter_chain or filter_chain.get_python_filter_equivalent(datatype):
                    filtered_adapters.append(stored_adapter)
            except (TypeError, InvalidFilterChainInput):
                self.logger.exception("Could not evaluate filter on " + str(stored_adapter))
                has_operations_warning = True

        return datatype, filtered_adapters, has_operations_warning

    def _group_adapters_by_category(self, stored_adapters, categories):
        """
        :param stored_adapters: list StoredAdapter
        :return: dict(category_name: List AlgorithmTransientGroup), empty groups all in the same AlgorithmTransientGroup
        """
        categories_dict = dict()
        for adapter in stored_adapters:
            category_name = categories.get(adapter.fk_category)
            if category_name in categories_dict:
                groups_list = categories_dict.get(category_name)
            else:
                groups_list = []
                categories_dict[category_name] = groups_list
            group = AlgorithmTransientGroup(adapter.group_name, adapter.group_description)
            group = self._find_group(groups_list, group)
            group.children.append(adapter)
        return categories_dict

    @staticmethod
    def get_generic_entity(entity_type, filter_value, select_field):
        return dao.get_generic_entity(entity_type, filter_value, select_field)

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
            select_entity = MeasurePointsSelection(ui_name, selected_nodes, datatype_gid, project_id)

        dao.store_entity(select_entity)

    ##########################################################################
    ##########    Bellow are PSE Filters specific methods   ##################
    ##########################################################################

    @staticmethod
    def get_stored_pse_filters(datatype_group_gid):
        return dao.get_stored_pse_filters(datatype_group_gid)

    @staticmethod
    def save_pse_filter(ui_name, datatype_group_gid, threshold_value, applied_on):
        """
        Store in DB a PSE filter.
        """
        select_entities = dao.get_stored_pse_filters(datatype_group_gid, ui_name)

        if select_entities:
            # when the UI name is already in DB, update the existing entity
            select_entity = select_entities[0]
            select_entity.threshold_value = threshold_value
            select_entity.applied_on = applied_on  # this is the type, as in applied on size or color
        else:
            select_entity = StoredPSEFilter(ui_name, datatype_group_gid, threshold_value, applied_on)

        dao.store_entity(select_entity)
