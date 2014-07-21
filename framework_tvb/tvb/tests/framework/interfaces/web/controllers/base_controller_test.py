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

import os
import cherrypy
from tvb.basic.config.settings import TVBSettings as cfg
from tvb.basic.config.utils import EnhancedDictionary
from tvb.interfaces.web.controllers.common import KEY_PROJECT, KEY_USER
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.test_factory import TestFactory



class BaseControllersTest(BaseTestCase):
    class CherrypySession(EnhancedDictionary):
        data = {}
        # mock methods
        def acquire_lock(self):
            pass

        clear = clean_up = release_lock = acquire_lock


    def _expect_redirect(self, page, method, *args, **kwargs):
        """
        A generic mechanism that calls a method with some arguments and expects a redirect
        to a given page.
        """
        try:
            method(*args, **kwargs)
            self.fail("Should be redirect to %s." % (page,))
        except cherrypy.HTTPRedirect, redirect:
            self.assertTrue(redirect.urls[0].endswith(page), "Should be redirect to %s" % (page,))


    def init(self):
        """
        Have a different name than setUp so we can use it safely in transactions and it will
        not be called before running actual test.
        """
        # Add 3 entries so we no longer consider this the first run.
        cfg.add_entries_to_config_file({'test': 'test',
                                        'test1': 'test1',
                                        'test2': 'test2'})
        self.test_user = TestFactory.create_user(username="CtrlTstUsr")
        self.test_project = TestFactory.create_project(self.test_user, "Test")
        cherrypy.session = BaseControllersTest.CherrypySession()
        cherrypy.session[KEY_USER] = self.test_user
        cherrypy.session[KEY_PROJECT] = self.test_project


    def cleanup(self):
        """
        Have a different name than tearDown so we can use it safely in transactions and it will
        not be called after running actual test.
        """
        if os.path.exists(cfg.TVB_CONFIG_FILE):
            os.remove(cfg.TVB_CONFIG_FILE)
            