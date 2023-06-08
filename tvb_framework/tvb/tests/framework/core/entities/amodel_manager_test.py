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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os
import pytest
from tvb.tests.framework.core.base_testcase import BaseTestCase, init_test_env
from tvb.basic.profile import TvbProfile
from tvb.core.entities.storage import dao
from tvb.config.init.model_manager import initialize_startup, reset_database


class TestsModelManager(BaseTestCase):
    """
    This class contains tests for tvb.config.init.model_manager module.
    """

    def teardown_method(self):
        init_test_env()

    def test_initialize_startup(self):
        """
        Test "reset_database" and "initialize_startup" calls.
        """
        reset_database()
        # Table USERS should not exist:
        with pytest.raises(Exception): dao.get_all_users()

        initialize_startup()
        # Table exists, but no rows
        assert 0 == len(dao.get_all_users())
        assert dao.get_system_user() is None
        # DB revisions folder should exist:
        assert os.path.exists(TvbProfile.current.db.DB_VERSIONING_REPO)
