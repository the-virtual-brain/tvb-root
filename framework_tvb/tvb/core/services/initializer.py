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
"""

import shutil
import datetime
import threading
from tvb.basic.profile import TvbProfile
from tvb.core.adapters.introspector import Introspector
from tvb.core.code_versions.code_update_manager import CodeUpdateManager
from tvb.core.entities import model
from tvb.core.entities.file.files_update_manager import FilesUpdateManager
from tvb.core.entities.storage import dao
from tvb.core.entities.model_manager import initialize_startup, reset_database
from tvb.core.services.project_service import initialize_storage
from tvb.core.services.user_service import UserService
from tvb.core.services.settings_service import SettingsService



def reset():
    """
    Service Layer for Database reset.
    """
    reset_database()



def initialize(introspected_modules, skip_import=False):
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
    for module in introspected_modules:
        introspector = Introspector(module)
        # Introspection is always done, even if DB was not empty.
        introspector.introspect(True)

    # Now remove or mark as removed any unverified Algorithm, Algo-Category or Portlet
    to_invalidate, to_remove = dao.get_non_validated_entities(start_introspection_time)
    for entity in to_invalidate:
        entity.removed = True
    dao.store_entities(to_invalidate)
    for entity in to_remove:
        dao.remove_entity(entity.__class__, entity.id)

    if not TvbProfile.is_first_run():
        # Create default users.
        if is_db_empty:
            dao.store_entity(model.User(TvbProfile.current.web.admin.SYSTEM_USER_NAME, None, None, True, None))
            UserService().create_user(username=TvbProfile.current.web.admin.ADMINISTRATOR_NAME,
                                      password=TvbProfile.current.web.admin.ADMINISTRATOR_PASSWORD,
                                      email=TvbProfile.current.web.admin.ADMINISTRATOR_EMAIL,
                                      role=model.ROLE_ADMINISTRATOR, skip_import=skip_import)

        # In case actions related to latest code-changes are needed, make sure they are executed.
        CodeUpdateManager().run_all_updates()

        # In case the H5 version changed, run updates on all DataTypes
        if TvbProfile.current.version.DATA_CHECKED_TO_VERSION < TvbProfile.current.version.DATA_VERSION:
            thread = threading.Thread(target=FilesUpdateManager().run_all_updates)
            thread.start()

        # Clean tvb-first-time-run temporary folder, as we are no longer at the first run:
        shutil.rmtree(TvbProfile.current.FIRST_RUN_STORAGE, True)
