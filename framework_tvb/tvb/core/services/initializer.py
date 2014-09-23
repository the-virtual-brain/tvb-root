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
"""
import datetime
import shutil
from tvb.basic.config.settings import TVBSettings as cfg
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.entities.model_manager import initialize_startup, reset_database
from tvb.core.code_versions.code_update_manager import CodeUpdateManager
from tvb.core.services.project_service import initialize_storage
from tvb.core.services.user_service import UserService
from tvb.core.services.settings_service import SettingsService
from tvb.core.services.event_handlers import read_events
from tvb.core.adapters.introspector import Introspector
from tvb.core.traits import db_events


def reset():
    """
    Service Layer for Database reset.
    """
    reset_database()


def initialize(introspected_modules, load_xml_events=True):
    """
    Initialize when Application is starting.
    Check for new algorithms or new DataTypes.
    """
    SettingsService().check_db_url(cfg.DB_URL)
    
    ## Initialize DB
    is_db_empty = initialize_startup()
    
    ## Create Projects storage root in case it does not exist.
    initialize_storage()
    
    ## Populate DB algorithms, by introspection
    event_folders = []
    start_introspection_time = datetime.datetime.now()
    for module in introspected_modules:
        introspector = Introspector(module)
        # Introspection is always done, even if DB was not empty.
        introspector.introspect(True)
        event_path = introspector.get_events_path()
        if event_path:
            event_folders.append(event_path)

    # Now remove or mark as removed any unverified Algo-Group, Algo-Category or Portlet
    to_invalidate, to_remove = dao.get_non_validated_entities(start_introspection_time)
    for entity in to_invalidate:
        entity.removed = True
    dao.store_entities(to_invalidate)
    for entity in to_remove:
        dao.remove_entity(entity.__class__, entity.id)
   
    ## Populate events
    if load_xml_events:
        read_events(event_folders)
    
    ## Make sure DB events are linked.
    db_events.attach_db_events()
    
    ## Create default users.
    if is_db_empty:
        dao.store_entity(model.User(cfg.SYSTEM_USER_NAME, None, None, True, None))
        UserService().create_user(username=cfg.ADMINISTRATOR_NAME, password=cfg.ADMINISTRATOR_PASSWORD,
                                  email=cfg.ADMINISTRATOR_EMAIL, role=model.ROLE_ADMINISTRATOR)
        
    ## In case actions related to latest code-changes are needed, make sure they are executed.
    CodeUpdateManager().run_all_updates()

    ## Clean tvb-first-time-run temporary folder, if we are no longer at the first run:
    if not SettingsService.is_first_run():
        shutil.rmtree(cfg.FIRST_RUN_STORAGE, True)
            
