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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import unittest
from tvb.basic.profile import TvbProfile
from tvb.adapters.visualizers.pse_discrete import DiscretePSEAdapter
from tvb.adapters.visualizers.pse_isocline import IsoclinePSEAdapter
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory
from tvb.tests.framework.core.base_testcase import TransactionalTestCase


class PSETest(TransactionalTestCase):
    """
    Unit-tests for BrainViewer.
    """


    def setUp(self):
        """
        Sets up the environment for running the tests;
        creates a datatype group
        """
        self.datatypeFactory = DatatypesFactory()
        self.group = self.datatypeFactory.create_datatype_group()


    def test_launch_discrete(self):
        """
        Check that all required keys are present in output from PSE Discrete Adapter launch.
        """
        viewer = DiscretePSEAdapter()
        result = viewer.launch(self.group)

        expected_keys = ['status', 'size_metric', 'series_array', 'min_shape_size', 'min_color', 'd3_data',
                         'max_shape_size', 'max_color', 'mainContent', 'labels_y', 'labels_x', 'isAdapter',
                         'has_started_ops', 'datatype_group_gid', 'datatypes_dict', 'color_metric']
        for key in expected_keys:
            self.assertTrue(key in result)
        self.assertEqual(self.group.gid, result["datatype_group_gid"])
        self.assertEqual('false', result["has_started_ops"])


    def test_launch_isocline(self):
        """
        Check that all required keys are present in output from PSE Discrete Adapter launch.
        """
        viewer = IsoclinePSEAdapter()
        result = viewer.launch(self.group)
        self.assertEqual(viewer._ui_name, result["title"])
        self.assertEqual(TvbProfile.current.web.MPLH5_SERVER_URL, result["mplh5ServerURL"])
        self.assertEqual(1, len(result["figureNumbers"]))
        self.assertEqual(1, len(result["metrics"]))


def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(PSETest))
    return test_suite


if __name__ == "__main__":
    # So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
