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

import os
import cherrypy
from tvb.tests.framework.core.base_testcase import BaseTestCase, TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory
from tvb.basic.profile import TvbProfile
from tvb.basic.config.utils import EnhancedDictionary
from tvb.interfaces.web.controllers.common import KEY_PROJECT, KEY_USER


class BaseControllersTest(BaseTestCase):

    """
    Mock CherryPy session
    """
    class CherrypySession(EnhancedDictionary):

        def acquire_lock(self):
            pass

        data = {}
        clear = clean_up = release_lock = acquire_lock

    def _expect_redirect(self, page, method, *args, **kwargs):
        """
        A generic mechanism that calls a method with some arguments and expects a redirect to a given page.
        """
        try:
            method(*args, **kwargs)
            raise AssertionError("Expected redirect to %s." % (page,))
        except cherrypy.HTTPRedirect as redirect:
            url = redirect.urls[0]
            assert url.endswith(page), "Should be redirect to %s not %s" % (page, url)

    def init(self, with_data=True, user_role="test"):
        """
        Have a different name than transactional_setup_method so we can use it safely in transactions and it will
        not be called before running actual test.
        Using transactional_setup_method inheritance here won't WORK!! See TransactionalTest
        """
        cherrypy.session = BaseControllersTest.CherrypySession()

        if with_data:
            # Add 3 entries so we no longer consider this the first run.
            TvbProfile.current.manager.add_entries_to_config_file({'test': 'test',
                                                                   'test1': 'test1',
                                                                   'test2': 'test2'})
            self.test_user = TestFactory.create_user(username="CtrlTstUsr", role=user_role)
            self.test_project = TestFactory.create_project(self.test_user, "Test")

            cherrypy.session[KEY_USER] = self.test_user
            cherrypy.session[KEY_PROJECT] = self.test_project

    def cleanup(self):
        """
        Have a different name than transactional_teardown_method so we can use it safely in transactions and it will
        not be called after running actual test.
        Using transactional_teardown_method here won't WORK!! See TransactionalTest
        """
        if os.path.exists(TvbProfile.current.TVB_CONFIG_FILE):
            os.remove(TvbProfile.current.TVB_CONFIG_FILE)

        TvbProfile._build_profile_class(TvbProfile.CURRENT_PROFILE_NAME)
        # let for the other test the env clean, with is_first_run returning True


class BaseTransactionalControllerTest(TransactionalTestCase, BaseControllersTest):
    """
    Flag class, for assuring inheritance from both the TransactionalTestCase and BaseControllerTest
    """
    pass
