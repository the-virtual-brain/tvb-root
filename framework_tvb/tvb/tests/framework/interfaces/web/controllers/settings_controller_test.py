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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os
import json
import copy
import shutil
import pytest
import cherrypy
from time import sleep
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest
from tvb.basic.profile import TvbProfile
from tvb.basic.config import stored
from tvb.core.utils import get_matlab_executable, hash_password
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.settings_controller import SettingsController


class TestSettingsController(BaseTransactionalControllerTest):
    """
    Unit tests for SettingsController class
    """

    was_reset = False
    accepted_db_url = ('sqlite:///TestFolder' + os.path.sep + 'tvb-database.db'
                       if TvbProfile.current.db.SELECTED_DB == 'sqlite'
                       else TvbProfile.current.db.DB_URL)

    VALID_SETTINGS = {'KEYCLOAK_CONFIGURATION': 'TestFolder' + os.path.sep + 'keycloak-config.json',
                      'KEYCLOAK_WEB_CONFIGURATION': 'TestFolder' + os.path.sep + 'keycloak-web-config.json',
                      'TVB_STORAGE': "TestFolder",
                      'USR_DISK_SPACE': 1,
                      'MAXIMUM_NR_OF_THREADS': 6,
                      'MAXIMUM_NR_OF_VERTICES_ON_SURFACE': 142,
                      'MAXIMUM_NR_OF_OPS_IN_RANGE': 16,
                      'MATLAB_EXECUTABLE': '',

                      'DEPLOY_CLUSTER': 'True',
                      'CLUSTER_SCHEDULER': TvbProfile.current.cluster.SCHEDULER_OAR,
                      'SELECTED_DB': TvbProfile.current.db.SELECTED_DB,  # Not changeable,due to test profile overwrites
                      'URL_VALUE': accepted_db_url,

                      'WEB_SERVER_PORT': 9999,

                      'ADMINISTRATOR_NAME': 'test_admin',
                      'ADMINISTRATOR_DISPLAY_NAME': 'test_Administrator',
                      'ADMINISTRATOR_PASSWORD': "test_pass",
                      'ADMINISTRATOR_EMAIL': 'admin@test.test'}

    def transactional_setup_method(self):

        self.init(with_data=False)
        self.settings_c = SettingsController()
        assert TvbProfile.is_first_run()

    def transactional_teardown_method(self):
        """ Cleans the testing environment """
        self.cleanup()
        self.clean_database()

        if os.path.exists(self.VALID_SETTINGS['TVB_STORAGE']):
            shutil.rmtree(self.VALID_SETTINGS['TVB_STORAGE'])

    def test_with_invalid_admin_settings(self):

        self._assert_invalid_parameters({'ADMINISTRATOR_NAME': '',
                                         'ADMINISTRATOR_PASSWORD': '',
                                         'ADMINISTRATOR_EMAIL': ''})

        self._assert_invalid_parameters({'ADMINISTRATOR_EMAIL': "bla.com"})

    def test_with_invalid_web_settings(self):

        self._assert_invalid_parameters({'WEB_SERVER_PORT': 'a'})

        self._assert_invalid_parameters({'WEB_SERVER_PORT': '70000'})

    def test_with_invalid_settings(self):

        self._assert_invalid_parameters({'TVB_STORAGE': '',
                                         'SELECTED_DB': '',
                                         'URL_VALUE': ''})

        self._assert_invalid_parameters({'USR_DISK_SPACE': '',
                                         'MAXIMUM_NR_OF_THREADS': '0',
                                         'MAXIMUM_NR_OF_VERTICES_ON_SURFACE': '-1',
                                         'MAXIMUM_NR_OF_OPS_IN_RANGE': '2'})

        self._assert_invalid_parameters({'USR_DISK_SPACE': 'a',
                                         'MAXIMUM_NR_OF_THREADS': '20',
                                         'MAXIMUM_NR_OF_VERTICES_ON_SURFACE': str(256 * 256 * 256 + 1),
                                         'MAXIMUM_NR_OF_OPS_IN_RANGE': '10000'})

        self._assert_invalid_parameters({'MAXIMUM_NR_OF_THREADS': 'c',
                                         'MAXIMUM_NR_OF_VERTICES_ON_SURFACE': 'b',
                                         'MAXIMUM_NR_OF_OPS_IN_RANGE': 'a'})

    def _assert_invalid_parameters(self, params_dictionary):
        """
        Simulate submit of a given params (key:value) and check that they are found in the error response
        """
        submit_data = copy.copy(self.VALID_SETTINGS)
        for key, value in params_dictionary.items():
            submit_data[key] = value

        response = self.settings_c.settings(save_settings=True, **submit_data)

        assert common.KEY_ERRORS in response
        for key in params_dictionary:
            assert key in response[common.KEY_ERRORS], "Not found in errors %s" % key

    def test_with_valid_settings(self):

        submit_data = copy.copy(self.VALID_SETTINGS)
        self.settings_c._restart_services = self._fake_restart_services

        with pytest.raises(cherrypy.HTTPRedirect):
            self.settings_c.settings(save_settings=True, **self.VALID_SETTINGS)

        # wait until 'restart' is done
        sleep(1)
        assert self.was_reset
        assert len(TvbProfile.current.manager.stored_settings) == 20

        assert submit_data['TVB_STORAGE'] == TvbProfile.current.TVB_STORAGE
        assert submit_data['USR_DISK_SPACE'] * 2 ** 10 == TvbProfile.current.MAX_DISK_SPACE
        assert submit_data['MAXIMUM_NR_OF_THREADS'] == TvbProfile.current.MAX_THREADS_NUMBER
        assert submit_data['MAXIMUM_NR_OF_OPS_IN_RANGE'] == TvbProfile.current.MAX_RANGE_NUMBER
        assert submit_data['MAXIMUM_NR_OF_VERTICES_ON_SURFACE'] == TvbProfile.current.MAX_SURFACE_VERTICES_NUMBER

        assert submit_data['DEPLOY_CLUSTER'] == str(TvbProfile.current.cluster.IS_DEPLOY)
        assert submit_data['SELECTED_DB'] == TvbProfile.current.db.SELECTED_DB
        assert submit_data['URL_VALUE'] == TvbProfile.current.db.DB_URL

        assert submit_data['WEB_SERVER_PORT'] == TvbProfile.current.web.SERVER_PORT

        assert submit_data['ADMINISTRATOR_NAME'] == TvbProfile.current.web.admin.ADMINISTRATOR_NAME
        assert submit_data['ADMINISTRATOR_EMAIL'] == TvbProfile.current.web.admin.ADMINISTRATOR_EMAIL
        assert hash_password(
            submit_data['ADMINISTRATOR_PASSWORD']) == TvbProfile.current.web.admin.ADMINISTRATOR_PASSWORD

    def _fake_restart_services(self, should_reset):
        """
        This function will replace the SettingsController._restart_service method,
        to avoid problems in tests due to restart.
        """
        self.was_reset = should_reset
        TvbProfile._build_profile_class(TvbProfile.CURRENT_PROFILE_NAME)

    def test_check_db_url(self):
        """
        Test that for a various DB URLs, the correct check response is returned.
        """
        submit_data = {stored.KEY_STORAGE: TvbProfile.current.TVB_STORAGE,
                       stored.KEY_DB_URL: TvbProfile.current.db.DB_URL}
        result = json.loads(self.settings_c.check_db_url(**submit_data))
        assert result['status'] == 'ok'

        submit_data[stored.KEY_DB_URL] = "this URL should be invalid"
        result = json.loads(self.settings_c.check_db_url(**submit_data))
        assert result['status'] == 'not ok'

    @pytest.mark.skipif(get_matlab_executable() is None, reason="Matlab or Octave not installed!")
    def test_check_matlab_path(self):
        """
        Test that for a various Matlab paths, the correct check response is returned.
        """
        submit_data = {stored.KEY_MATLAB_EXECUTABLE: get_matlab_executable()}
        result = json.loads(self.settings_c.validate_matlab_path(**submit_data))
        assert result['status'] == 'ok'

        submit_data[stored.KEY_MATLAB_EXECUTABLE] = "/this/path/should/be/invalid"
        result = json.loads(self.settings_c.validate_matlab_path(**submit_data))
        assert result['status'] == 'not ok'
