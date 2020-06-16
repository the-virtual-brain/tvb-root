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

import pytest
import os
import tvb_data
from tvb.adapters.analyzers.bct_adapters import BaseBCTModel
from tvb.core.entities.model.model_operation import Algorithm
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.utils import get_matlab_executable
from tvb.core.entities.storage import dao
from tvb.tests.framework.core.factory import TestFactory


class TestBCT(TransactionalTestCase):
    """
    Test that all BCT analyzers are executed without error.
    We do not verify that the algorithms are correct, because that is outside the purpose of TVB framework.
    """

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
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_76.zip')
        self.connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path)

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

        view_model = BaseBCTModel()
        view_model.connectivity = self.connectivity.gid
        algo_category = dao.get_category_by_id(self.bct_adapters[0].stored_adapter.fk_category)

        for adapter_instance in self.bct_adapters:
            results = TestFactory.launch_synchronously(self.test_user, self.test_project, adapter_instance,
                                                       view_model, algo_category)
            assert len(results) > 0

    @pytest.mark.skipif(get_matlab_executable() is None, reason="Matlab or Octave not installed!")
    def test_bct_descriptions(self):
        """
        Iterate all BCT algorithms and check that description has been extracted from *.m files.
        """
        for adapter_instance in self.bct_adapters:
            assert len(adapter_instance.stored_adapter.description) > 10, "Description was not loaded properly for " \
                                                                          "algorithm %s" % (str(adapter_instance))
