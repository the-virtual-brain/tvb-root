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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
import datetime
import importlib
import os
import shutil
import threading
from types import ModuleType
from tvb.adapters.datatypes.db import DATATYPE_REMOVERS
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.config import VIEW_MODEL2ADAPTER
from tvb.config.init.datatypes_registry import populate_datatypes_registry
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.config.init.model_manager import initialize_startup, reset_database
from tvb.core import removers_factory
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.constants import ELEM_INPUTS, ATT_TYPE, ATT_NAME
from tvb.core.adapters.exceptions import XmlParserException
from tvb.core.code_versions.code_update_manager import CodeUpdateManager
from tvb.core.entities.file.files_update_manager import FilesUpdateManager
from tvb.core.entities.model.model_operation import Algorithm, AlgorithmCategory
from tvb.core.entities.model.model_project import User, ROLE_ADMINISTRATOR
from tvb.core.entities.model.model_workflow import Portlet
from tvb.core.entities.storage import dao, SA_SESSIONMAKER
from tvb.core.entities.storage.session_maker import build_db_engine
from tvb.core.neotraits.db import Base
from tvb.core.portlets.xml_reader import XMLPortletReader, ATT_OVERWRITE
from tvb.core.services.project_service import initialize_storage
from tvb.core.services.settings_service import SettingsService
from tvb.core.services.user_service import UserService


def reset():
    """
    Service Layer for Database reset.
    """
    reset_database()


def command_initializer(persist_settings=True, skip_import=False):
    if persist_settings and TvbProfile.is_first_run():
        settings_service = SettingsService()
        settings = {}
        # Save default settings
        for key, setting in settings_service.configurable_keys.items():
            settings[key] = setting['value']
        settings_service.save_settings(**settings)
    TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)
    # Build new db engine in case DB URL value changed
    new_db_engine = build_db_engine()
    SA_SESSIONMAKER.configure(bind=new_db_engine)

    # Initialize application
    initialize(skip_import)

def initialize(skip_import=False, skip_updates=False):
    """
    Initialize when Application is starting.
    Check for new algorithms or new DataTypes.
    """
    SettingsService().check_db_url(TvbProfile.current.db.DB_URL)

    # Initialize DB
    is_db_empty = initialize_startup()

    # Create Projects storage root in case it does not exist.
    initialize_storage()

    # Populate DB algorithms, by introspection
    start_introspection_time = datetime.datetime.now()
    # Introspection is always done, even if DB was not empty.
    introspector = Introspector()
    introspector.introspect()

    # Now remove or mark as removed any unverified Algorithm, Algo-Category or Portlet
    to_invalidate, to_remove = dao.get_non_validated_entities(start_introspection_time)
    for entity in to_invalidate:
        entity.removed = True
    dao.store_entities(to_invalidate)
    for entity in to_remove:
        dao.remove_entity(entity.__class__, entity.id)

    if not TvbProfile.is_first_run() and not skip_updates:
        # Create default users.
        if is_db_empty:
            dao.store_entity(
                User(TvbProfile.current.web.admin.SYSTEM_USER_NAME, TvbProfile.current.web.admin.SYSTEM_USER_NAME, None,
                     None, True, None))
            UserService().create_user(username=TvbProfile.current.web.admin.ADMINISTRATOR_NAME,
                                      display_name=TvbProfile.current.web.admin.ADMINISTRATOR_DISPLAY_NAME,
                                      password=TvbProfile.current.web.admin.ADMINISTRATOR_PASSWORD,
                                      email=TvbProfile.current.web.admin.ADMINISTRATOR_EMAIL,
                                      role=ROLE_ADMINISTRATOR, skip_import=skip_import)

        # In case actions related to latest code-changes are needed, make sure they are executed.
        CodeUpdateManager().run_all_updates()

        # In case the H5 version changed, run updates on all DataTypes
        if TvbProfile.current.version.DATA_CHECKED_TO_VERSION < TvbProfile.current.version.DATA_VERSION:
            thread = threading.Thread(target=FilesUpdateManager().run_all_updates)
            thread.start()

        # Clean tvb-first-time-run temporary folder, as we are no longer at the first run:
        shutil.rmtree(TvbProfile.current.FIRST_RUN_STORAGE, True)


class Introspector(object):

    def __init__(self):
        self.introspection_registry = IntrospectionRegistry()
        self.logger = get_logger(self.__class__.__module__)

    def introspect(self):
        self._ensure_datatype_tables_are_created()
        populate_datatypes_registry()
        for algo_category_class in IntrospectionRegistry.ADAPTERS:
            algo_category_id = self._populate_algorithm_categories(algo_category_class)
            self._populate_algorithms(algo_category_class, algo_category_id)
        # self._get_portlets()
        removers_factory.update_dictionary(DATATYPE_REMOVERS)

    @staticmethod
    def _ensure_datatype_tables_are_created():
        session = SA_SESSIONMAKER()
        Base.metadata.create_all(bind=session.connection())
        session.commit()
        session.close()

    @staticmethod
    def _populate_algorithm_categories(algo_category):
        algo_category_instance = dao.filter_category(algo_category.category_name, algo_category.rawinput,
                                                     algo_category.display, algo_category.launchable,
                                                     algo_category.order_nr)

        if algo_category_instance is not None:
            algo_category_instance.last_introspection_check = datetime.datetime.now()
            algo_category_instance.removed = False
        else:
            algo_category_instance = AlgorithmCategory(algo_category.category_name, algo_category.launchable,
                                                       algo_category.rawinput, algo_category.display,
                                                       algo_category.defaultdatastate, algo_category.order_nr,
                                                       datetime.datetime.now())
        algo_category_instance = dao.store_entity(algo_category_instance)

        return algo_category_instance.id

    def _populate_algorithms(self, algo_category_class, algo_category_id):
        for adapter_class in self.introspection_registry.ADAPTERS[algo_category_class]:
            try:
                if not adapter_class.can_be_active():
                    self.logger.warning("Skipped Adapter(probably because MATLAB not found):" + str(adapter_class))

                stored_adapter = Algorithm(adapter_class.__module__, adapter_class.__name__, algo_category_id,
                                           adapter_class.get_group_name(), adapter_class.get_group_description(),
                                           adapter_class.get_ui_name(), adapter_class.get_ui_description(),
                                           adapter_class.get_ui_subsection(), datetime.datetime.now())
                adapter_inst = adapter_class()

                adapter_form = adapter_inst.get_form()
                required_datatype = adapter_form.get_required_datatype()
                if required_datatype is not None:
                    required_datatype = required_datatype.__name__
                filters = adapter_form.get_filters()
                if filters is not None:
                    filters = filters.to_json()

                stored_adapter.required_datatype = required_datatype
                stored_adapter.datatype_filter = filters
                stored_adapter.parameter_name = adapter_form.get_input_name()
                stored_adapter.outputlist = str(adapter_inst.get_output())

                inst_from_db = dao.get_algorithm_by_module(adapter_class.__module__, adapter_class.__name__)
                if inst_from_db is not None:
                    stored_adapter.id = inst_from_db.id

                stored_adapter = dao.store_entity(stored_adapter, inst_from_db is not None)

                view_model_class = adapter_form.get_view_model()
                VIEW_MODEL2ADAPTER[view_model_class] = stored_adapter

                adapter_class.stored_adapter = stored_adapter

            except Exception:
                self.logger.exception("Could not introspect Adapters file:" + adapter_class.__module__)

    def _get_portlets(self):
        """
        Given a path in the form of a python package e.g.: "tvb.portlets', import
        the package, get it's folder and look for all the XML files defined
        there, then read all the portlets defined there and store them in DB.
        """
        portlet_folder = os.path.dirname(IntrospectionRegistry.PORTLETS_MODULE.__file__)
        portlets_list = []
        self._prepare_valid_portlets_list(portlet_folder, portlets_list)

        self.logger.debug("Refreshing portlets from xml declarations.")
        self._update_old_portlets_from_db(portlets_list)
        self._add_new_valid_portlets(portlets_list)

    @staticmethod
    def _build_adapter_from_declaration(adapter_declaration):
        """
        Build and adapter from the declaration in the portlets xml.
        """
        adapter_import_path = adapter_declaration[ATT_TYPE]
        class_name = adapter_import_path.split('.')[-1]
        module_name = adapter_import_path.replace('.' + class_name, '')
        algo = dao.get_algorithm_by_module(module_name, class_name)
        if algo is not None:
            return ABCAdapter.build_adapter(algo)
        else:
            return None

    def _prepare_valid_portlets_list(self, portlet_folder, portlets_list):
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
                            class_name = adapter[ATT_TYPE].split('.')[-1]
                            module_name = adapter[ATT_TYPE].replace('.' + class_name, '')
                            try:
                                # Check that module is properly declared
                                module = importlib.import_module(module_name)
                                if type(module) != ModuleType:
                                    is_valid = False
                                    self.logger.error("Wrong module %s in portlet %s" % (module_name, algo_identifier))
                                    continue
                                # Check that class is properly declared
                                if not hasattr(module, class_name):
                                    is_valid = False
                                    self.logger.error("Wrong class %s in portlet %s." % (class_name, algo_identifier))
                                    continue
                                # Check inputs that refers to this adapter
                                portlet_inputs = portlet_list[algo_identifier][ELEM_INPUTS]
                                adapter_instance = self._build_adapter_from_declaration(adapter)
                                if adapter_instance is None:
                                    is_valid = False
                                    self.logger.warning("No group having class=%s stored for portlet %s."
                                                        % (class_name, algo_identifier))
                                    continue

                                adapter_form = adapter_instance.get_form()
                                adapter_instance.submit_form(adapter_form())
                                # TODO: implement this for neoforms
                                adapter_form_field_names = {}  # adapter_instance.flaten_input_interface()
                                for input_entry in portlet_inputs.values():
                                    if input_entry[ATT_OVERWRITE] == adapter[ATT_NAME]:
                                        if input_entry[ATT_NAME] not in adapter_form_field_names:
                                            self.logger.error("Invalid input %s for adapter %s"
                                                              % (input_entry[ATT_NAME], adapter_instance))
                                            is_valid = False
                            except ImportError:
                                is_valid = False
                                self.logger.error("Invalid adapter declaration %s in portlet %s"
                                                  % (adapter[ATT_TYPE], algo_identifier))

                        if is_valid:
                            portlets_list.append(
                                Portlet(algo_identifier, complete_file_path, portlet_list[algo_identifier]['name']))

            except XmlParserException as excep:
                self.logger.exception(excep)
                self.logger.error("Invalid Portlet description File " + file_n + " will continue without it!!")

    @staticmethod
    def _update_old_portlets_from_db(portlets_list):
        stored_portlets = dao.get_available_portlets()
        # First update old portlets from DB
        for stored_portlet in stored_portlets:
            for verified_portlet in portlets_list:
                if stored_portlet.algorithm_identifier == verified_portlet.algorithm_identifier:
                    stored_portlet.xml_path = verified_portlet.xml_path
                    stored_portlet.last_introspection_check = datetime.datetime.now()
                    stored_portlet.name = verified_portlet.name
                    dao.store_entity(stored_portlet)
                    break

    def _add_new_valid_portlets(self, portlets_list):
        # Now add portlets that were not in DB at previous run but are valid now
        for portlet in portlets_list:
            db_entity = dao.get_portlet_by_identifier(portlet.algorithm_identifier)
            if db_entity is None:
                self.logger.debug("Will now store portlet %s" % (str(portlet),))
                dao.store_entity(portlet)
