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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import json
import unittest
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory
from tvb.interfaces.web.controllers.burst.exploration_controller import ParameterExplorationController


class ExplorationControllerTest(BaseTransactionalControllerTest):
    """
    Unit tests ParameterExplorationController
    """

    def setUp(self):
        """
        Sets up the environment for testing;
        creates a datatype group and a Parameter Exploration Controller
        """
        self.init()
        self.dt_group = DatatypesFactory().create_datatype_group()
        self.controller = ParameterExplorationController()


    def tearDown(self):
        """ Cleans the testing environment """
        self.cleanup()


    def test_draw_discrete_exploration(self):
        """
        Test that Discrete PSE is getting launched and correct fields are prepared.
        """
        result = self.controller.draw_discrete_exploration(self.dt_group.gid, 'burst', None, None)
        self.assertTrue(result['available_metrics'] == DatatypesFactory.DATATYPE_MEASURE_METRIC.keys())
        self.assertEqual(result['color_metric'], DatatypesFactory.DATATYPE_MEASURE_METRIC.keys()[0])
        self.assertEqual(result['size_metric'], None)
        self.assertEqual(DatatypesFactory.RANGE_1[1], json.loads(result['labels_x']))
        self.assertEqual(DatatypesFactory.RANGE_2[1], json.loads(result['labels_y']))
        data = json.loads(result['d3_data'])
        self.assertEqual(len(data), len(DatatypesFactory.RANGE_1[1]))
        for row in data.values():
            self.assertEqual(len(row), len(DatatypesFactory.RANGE_2[1]))
            for entry in row.values():
                self.assertEqual(entry['dataType'], 'Datatype2')
                for key in ['Gid', 'color_weight', 'operationId', 'tooltip']:
                    self.assertTrue(key in entry)


    def test_draw_isocline_exploration(self):
        """
        Test that isocline PSE gets launched.
        """
        result = self.controller.draw_isocline_exploration(self.dt_group.gid, 500, 600)
        self.assertTrue(DatatypesFactory.DATATYPE_MEASURE_METRIC.keys()[0] in result['figureNumbers'])
        self.assertTrue(result['isAdapter'])
        self.assertEqual(DatatypesFactory.DATATYPE_MEASURE_METRIC, result['metrics'])
        self.assertTrue(result['showFullToolbar'])


def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(ExplorationControllerTest))
    return test_suite


if __name__ == "__main__":
    # So you can run tests individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
