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

import unittest
from tvb.core.utils import get_matlab_executable
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.services.operation_service import OperationService
from tvb.core.adapters.exceptions import InvalidParameterException
from tvb.datatypes.connectivity import Connectivity
from tvb.tests.framework.core.test_factory import TestFactory
from tvb.tests.framework.core.base_testcase import TransactionalTestCase



class BCTTest(TransactionalTestCase):
    """
    Test that all BCT analyzers are executed without error.
    We do not verify that the algorithms are correct, because that is outside the purpose of TVB framework.
    """
    EXPECTED_TO_FAIL_VALIDATION = ["CCBU", "CCWU", "EIGUN", "KCCBU", "TBU", "TWU"]


    @unittest.skipIf(get_matlab_executable() is None, "Matlab or Octave not installed!")
    def setUp(self):
        """
        Sets up the environment for running the tests;
        creates a test user, a test project, a connectivity and a list of BCT adapters;
        imports a CFF data-set
        """
        self.test_user = TestFactory.create_user("BCT_User")
        self.test_project = TestFactory.create_project(self.test_user, "BCT-Project")
        ### Make sure Connectivity is in DB
        TestFactory.import_cff(test_user=self.test_user, test_project=self.test_project)
        self.connectivity = dao.get_generic_entity(Connectivity, 'John Doe', 'subject')[0]

        self.algo_groups = dao.get_generic_entity(model.AlgorithmGroup, 'bct', 'algorithm_param_name')

        self.assertTrue(self.algo_groups is not None)
        self.assertEquals(6, len(self.algo_groups))
        self.bct_adapters = []
        for group in self.algo_groups:
            self.bct_adapters.append(TestFactory.create_adapter(group, self.test_project))


    def tearDown(self):
        """
        Cleans the database after the tests
        """
        self.clean_database(True)


    @unittest.skipIf(get_matlab_executable() is None, "Matlab or Octave not installed!")
    def test_bct_all(self):
        """
        Iterate all BCT algorithms and execute them.
        """
        for i in xrange(len(self.bct_adapters)):
            for bct_identifier in self.bct_adapters[i].get_algorithms_dictionary():
                ### Prepare Operation and parameters
                algorithm = dao.get_algorithm_by_group(self.algo_groups[i].id, bct_identifier)
                operation = TestFactory.create_operation(algorithm=algorithm, test_user=self.test_user,
                                                         test_project=self.test_project,
                                                         operation_status=model.STATUS_STARTED)
                self.assertEqual(model.STATUS_STARTED, operation.status)
                ### Launch BCT algorithm
                submit_data = {self.algo_groups[i].algorithm_param_name: bct_identifier,
                               algorithm.parameter_name: self.connectivity.gid}
                try:
                    OperationService().initiate_prelaunch(operation, self.bct_adapters[i], {}, **submit_data)
                    if bct_identifier in BCTTest.EXPECTED_TO_FAIL_VALIDATION:
                        raise Exception("Algorithm %s was expected to throw input validation "
                                        "exception, but did not!" % (bct_identifier,))

                    operation = dao.get_operation_by_id(operation.id)
                    ### Check that operation status after execution is success.
                    self.assertEqual("4-FINISHED", operation.status)
                    ### Make sure at least one result exists for each BCT algorithm
                    results = dao.get_generic_entity(model.DataType, operation.id, 'fk_from_operation')
                    self.assertTrue(len(results) > 0)

                except InvalidParameterException, excep:
                    ## Some algorithms are expected to throw validation exception.
                    if bct_identifier not in BCTTest.EXPECTED_TO_FAIL_VALIDATION:
                        raise excep


    @unittest.skipIf(get_matlab_executable() is None, "Matlab or Octave not installed!")
    def test_bct_descriptions(self):
        """
        Iterate all BCT algorithms and check description not empty.
        """
        for i in xrange(len(self.bct_adapters)):
            for bct_identifier in self.bct_adapters[i].get_algorithms_dictionary():
                ### Prepare Operation and parameters
                algorithm = dao.get_algorithm_by_group(self.algo_groups[i].id, bct_identifier)
                self.assertTrue(len(algorithm.description) > 0,
                                "Description was not loaded properly for algorithm %s" % (str(algorithm, )))



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(BCTTest))
    return test_suite



if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
    
    
    
    