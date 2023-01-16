# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
from datetime import datetime
import shutil
import threading
from tvb.adapters.datatypes.db import DATATYPE_REMOVERS
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.config import VIEW_MODEL2ADAPTER
from tvb.config.init.datatypes_registry import populate_datatypes_registry
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.config.init.model_manager import initialize_startup, reset_database
from tvb.core import removers_factory
from tvb.core.code_versions.code_update_manager import CodeUpdateManager
from tvb.core.entities.file.files_update_manager import FilesUpdateManager
from tvb.core.entities.model.model_operation import Algorithm, AlgorithmCategory
from tvb.core.entities.model.model_project import User, ROLE_ADMINISTRATOR
from tvb.core.entities.storage import dao, transactional, SA_SESSIONMAKER
from tvb.core.entities.storage.session_maker import build_db_engine
from tvb.core.neotraits.db import Base
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
    start_introspection_time = datetime.now()
    # Introspection is always done, even if DB was not empty.
    introspector = Introspector()
    introspector.introspect()

    to_invalidate = dao.get_non_validated_entities(start_introspection_time)
    for entity in to_invalidate:
        entity.removed = True
    dao.store_entities(to_invalidate)

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
        thread = None
        if TvbProfile.current.version.DATA_CHECKED_TO_VERSION < TvbProfile.current.version.DATA_VERSION:
            thread = threading.Thread(target=FilesUpdateManager().run_all_updates)
            thread.start()

        # Clean tvb-first-time-run temporary folder, as we are no longer at the first run:
        shutil.rmtree(TvbProfile.current.FIRST_RUN_STORAGE, True)
        return thread


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
            algo_category_instance.last_introspection_check = datetime.now()
            algo_category_instance.removed = False
        else:
            algo_category_instance = AlgorithmCategory(algo_category.category_name, algo_category.launchable,
                                                       algo_category.rawinput, algo_category.display,
                                                       algo_category.defaultdatastate, algo_category.order_nr,
                                                       datetime.now())
        algo_category_instance = dao.store_entity(algo_category_instance)

        return algo_category_instance.id

    @transactional
    def _populate_algorithms(self, algo_category_class, algo_category_id):
        # type: (type, int) -> None
        for adapter_class in self.introspection_registry.ADAPTERS[algo_category_class]:
            try:
                if not adapter_class.can_be_active():
                    self.logger.warning("Skipped Adapter(probably because MATLAB not found):" + str(adapter_class))

                stored_adapter = Algorithm(adapter_class.__module__, adapter_class.__name__, algo_category_id,
                                           adapter_class.get_group_name(), adapter_class.get_group_description(),
                                           adapter_class.get_ui_name(), adapter_class.get_ui_description(),
                                           adapter_class.get_ui_subsection(), datetime.now())
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
