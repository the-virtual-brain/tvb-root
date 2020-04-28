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
"""

import numpy
import pytest
import os
import tvb_data
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.model.model_operation import *
from tvb.core.entities.model.model_datatype import *
from tvb.core.utils import get_matlab_executable
from tvb.core.entities.storage import dao
from tvb.core.services.operation_service import OperationService
from tvb.core.adapters.exceptions import InvalidParameterException
from tvb.datatypes.connectivity import Connectivity
from tvb.tests.framework.core.factory import TestFactory


class TestBCT(TransactionalTestCase):
    """
    Test that all BCT analyzers are executed without error.
    We do not verify that the algorithms are correct, because that is outside the purpose of TVB framework.
    """
    EXPECTED_TO_FAIL_VALIDATION = ["CentralityKCoreness", "CentralityEigenVector",
                                   "ClusteringCoefficientBU", "ClusteringCoefficientWU",
                                   "TransitivityBinaryUnDirected", "TransitivityWeightedUnDirected"]

    @pytest.mark.skipif(get_matlab_executable() is None, reason="Matlab or Octave not installed!")
    def transactional_setup_method(self):
        """
        Sets up the environment for running the tests;
        creates a test user, a test project, a connectivity and a list of BCT adapters;
        imports a CFF data-set
        """
        self.test_user = TestFactory.create_user("BCT_User")
        self.test_project = TestFactory.create_project(self.test_user, "BCT-Project")
        # Make sure Connectivity is in DB
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        conn_index = TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path)
        self.connectivity = ABCAdapter.load_traited_by_gid(conn_index.gid)

        # make weights matrix symmetric, or else some BCT algorithms will run infinitely:
        w = self.connectivity.weights
        self.connectivity.weights = w + w.T - numpy.diag(w.diagonal())

        algorithms = dao.get_generic_entity(Algorithm, 'Brain Connectivity Toolbox', 'group_description')
        assert algorithms is not None
        assert len(algorithms) > 5

        self.bct_adapters = []
        for algo in algorithms:
            self.bct_adapters.append(ABCAdapter.build_adapter(algo))

    def transactional_teardown_method(self):
        """
        Cleans the database after the tests
        """
        self.clean_database(True)

    @pytest.mark.skipif(get_matlab_executable() is None, reason="Matlab or Octave not installed!")
    def test_bct_all(self):
        """
        Iterate all BCT algorithms and execute them.
        """
        for adapter_instance in self.bct_adapters:
            algorithm = adapter_instance.stored_adapter
            operation = TestFactory.create_operation(algorithm=algorithm, test_user=self.test_user,
                                                     test_project=self.test_project,
                                                     operation_status=STATUS_STARTED)
            assert STATUS_STARTED == operation.status
            # Launch BCT algorithm
            submit_data = {algorithm.parameter_name: self.connectivity.gid}
            try:
                OperationService().initiate_prelaunch(operation, adapter_instance, {}, **submit_data)
                if algorithm.classname in TestBCT.EXPECTED_TO_FAIL_VALIDATION:
                    raise Exception("Algorithm %s was expected to throw input validation "
                                    "exception, but did not!" % (algorithm.classname,))

                operation = dao.get_operation_by_id(operation.id)
                # Check that operation status after execution is success.
                assert STATUS_FINISHED == operation.status
                # Make sure at least one result exists for each BCT algorithm
                results = dao.get_generic_entity(DataType, operation.id, 'fk_from_operation')
                assert len(results) > 0

            except InvalidParameterException as excep:
                # Some algorithms are expected to throw validation exception.
                if algorithm.classname not in TestBCT.EXPECTED_TO_FAIL_VALIDATION:
                    raise excep

    @pytest.mark.skipif(get_matlab_executable() is None, reason="Matlab or Octave not installed!")
    def test_bct_descriptions(self):
        """
        Iterate all BCT algorithms and check that description has been extracted from *.m files.
        """
        for adapter_instance in self.bct_adapters:
            assert len(adapter_instance.stored_adapter.description) > 10, "Description was not loaded properly for " \
                                                                          "algorithm %s" % (str(adapter_instance))
