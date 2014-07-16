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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import json
import unittest
import cherrypy
from tvb.core.entities.model import Dynamic
from tvb.core.entities.storage import dao
import tvb.interfaces.web.controllers.common as common
from tvb.interfaces.web.controllers.burst.region_model_parameters_controller import RegionsModelParametersController
from tvb.interfaces.web.controllers.burst.burst_controller import BurstController
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models import Generic2dOscillator, Kuramoto
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseControllersTest
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory
from tvb.tests.framework.adapters.simulator.simulator_adapter_test import SIMULATOR_PARAMETERS


class RegionsModelParametersControllerTest(TransactionalTestCase, BaseControllersTest):
    """ Unit tests for RegionsModelParametersController """
    
    def setUp(self):
        """
        Sets up the environment for testing;
        creates a `RegionsModelParametersController` and a connectivity
        """
        BaseControllersTest.init(self)
        self.region_m_p_c = RegionsModelParametersController()
        BurstController().index()
        stored_burst = cherrypy.session[common.KEY_BURST_CONFIG]
        _, self.connectivity = DatatypesFactory().create_connectivity()
        new_params = {}
        for key, val in SIMULATOR_PARAMETERS.iteritems():
            new_params[key] = {'value': val}
        new_params['connectivity'] = {'value': self.connectivity.gid}
        stored_burst.simulator_configuration = new_params

    def _setup_dynamic(self):
        dynamic_g = Dynamic("test_dyn", self.test_user.id,
                          Generic2dOscillator.__name__,
                          '[["tau", [1.0]], ["a", [5.0]], ["b", [-10.0]], ["c", [10.0]], ["I", [0.0]], ["d", [0.02]], ["e", [3.0]], ["f", [1.0]], ["g", [0.0]], ["alpha", [1.0]], ["beta", [5.0]], ["gamma", [1.0]]]',
                          HeunDeterministic.__name__,
                          None)

        dynamic_k = Dynamic("test_dyn_kura", self.test_user.id,
                          Kuramoto.__name__,
                          '[["omega", [1.0]]]',
                          HeunDeterministic.__name__,
                          None)

        self.dynamic_g = dao.store_entity(dynamic_g)
        self.dynamic_k = dao.store_entity(dynamic_k)

    def tearDown(self):
        """ Clean the testing environment """
        BaseControllersTest.cleanup(self)
    
    
    def test_index(self):
        """
        Verifies that result dictionary has the expected keys / values after call to
        `edit_model_parameters()`
        """
        result_dict = self.region_m_p_c.index()
        self.assertEqual(self.connectivity.gid, result_dict['connectivity_entity'].gid)
        self.assertEqual(result_dict['mainContent'], 'burst/model_param_region')
        self.assertEqual(result_dict['submit_parameters_url'], 
                         '/burst/modelparameters/regions/submit_model_parameters')
        self.assertTrue('dynamics' in result_dict)
        self.assertTrue('dynamics_json' in result_dict)
        self.assertTrue('pointsLabels' in result_dict)
        self.assertTrue('positions' in result_dict)

        json.loads(result_dict['dynamics_json'])

        
    def test_submit_model_parameters_happy(self):
        """
        Verifies call to `submit_model_parameters(...)` correctly redirects to '/burst/'
        """
        self._setup_dynamic()
        self.region_m_p_c.index()

        dynamic_ids = json.dumps([self.dynamic_g.id for _ in range(self.connectivity.number_of_regions)])

        self._expect_redirect('/burst/', self.region_m_p_c.submit_model_parameters, dynamic_ids)
        

    def test_submit_model_parameters_inconsistent_models(self):
        self._setup_dynamic()
        self.region_m_p_c.index()

        dynamic_ids = [self.dynamic_g.id for _ in range(self.connectivity.number_of_regions)]
        dynamic_ids[-1] = self.dynamic_k.id
        dynamic_ids = json.dumps(dynamic_ids)

        self.assertRaises(Exception,  self.region_m_p_c.submit_model_parameters, dynamic_ids)


def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(RegionsModelParametersControllerTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)