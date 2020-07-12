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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import tvb.tests.framework.adapters
from tvb.config.algorithm_categories import AlgorithmCategoryConfig
from tvb.config.init.initializer import Introspector
from tvb.config.init.introspector_registry import import_adapters
from tvb.core.entities.model.model_operation import Algorithm, AlgorithmCategory
from tvb.core.entities.storage import dao
from tvb.tests.framework.core.base_testcase import BaseTestCase


class TestCategory(AlgorithmCategoryConfig):
    category_name = 'AdaptersTest'
    rawinput = False
    order_nr = 42


ALL_TEST_ADAPTERS = ["testadapter1", "testadapter2", "testadapter3", "ndimensionarrayadapter", "testgroupadapter"]


class TestIntrospector(BaseTestCase):
    """
    Test class for the introspection module.
    """

    def test_introspect(self):
        """
        Test that expected categories and groups are found in DB after introspection.
        We also check algorithms introspected during base_testcase.init_test_env
        """
        introspector = Introspector()
        introspector.introspection_registry.ADAPTERS[TestCategory] = import_adapters(tvb.tests.framework.adapters,
                                                                                     ALL_TEST_ADAPTERS)
        introspector.introspect()

        all_categories = dao.get_algorithm_categories()
        category_ids = [cat.id for cat in all_categories if cat.displayname == TestCategory.category_name]
        adapters = dao.get_adapters_from_categories(category_ids)

        assert 5 == len(adapters), "Introspection failed!"
        nr_adapters_mod3 = 0

        for algorithm in adapters:
            assert algorithm.module in ['tvb.tests.framework.adapters.testadapter1',
                                        'tvb.tests.framework.adapters.testadapter2',
                                        'tvb.tests.framework.adapters.testadapter3'
                                        ], "Unknown Adapter Module:" + str(algorithm.module)
            assert algorithm.classname in ["TestAdapter1",
                                           "TestAdapter2", "TestAdapterHugeMemoryRequired",
                                           "TestAdapter3", "TestAdapterHDDRequired"
                                           ], "Unknown Adapter Class:" + str(algorithm.classname)
            if algorithm.module == 'tvb.tests.framework.adapters.testadapter3':
                nr_adapters_mod3 += 1

        assert nr_adapters_mod3 == 3

    def teardown_method(self):
        all_categories = dao.get_algorithm_categories()
        category_ids = [cat.id for cat in all_categories if cat.displayname == TestCategory.category_name]
        adapters = dao.get_adapters_from_categories(category_ids)
        for algorithm in adapters:
            dao.remove_entity(Algorithm, algorithm.id)
        dao.remove_entity(AlgorithmCategory, category_ids[0])
