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

import os
from tvb.config.init.initializer import Introspector
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.basic.profile import TvbProfile
from tvb.core.entities.storage import dao


class TestIntrospector(BaseTestCase):
    """
    Test class for the introspection module.
    """
    old_current_dir = TvbProfile.current.web.CURRENT_DIR

    def setup_method(self):
        """
        Introspect supplementary folder:
        """
        core_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        TvbProfile.current.web.CURRENT_DIR = os.path.dirname(core_path)

        self.introspector = Introspector()
        self.introspector.introspect()

    def teardown_method(self):
        """
        Revert changes settings and remove recently imported algorithms
        """
        TvbProfile.current.web.CURRENT_DIR = self.old_current_dir

    def test_introspect(self):
        """
        Test that expected categories and groups are found in DB after introspection.
        We also check algorithms introspected during base_testcase.init_test_env
        """

        all_categories = dao.get_algorithm_categories()
        category_ids = [cat.id for cat in all_categories if cat.displayname == "AdaptersTest"]
        adapters = dao.get_adapters_from_categories(category_ids)
        assert 8 == len(adapters), "Introspection failed!"
        nr_adapters_mod2 = 0
        for algorithm in adapters:
            assert algorithm.module in ['tvb.tests.framework.adapters.testadapter1',
                                        'tvb.tests.framework.adapters.testadapter2',
                                        'tvb.tests.framework.adapters.testadapter3',
                                        'tvb.tests.framework.adapters.ndimensionarrayadapter',
                                        'tvb.tests.framework.adapters.testgroupadapter'], "Unknown Adapter module:" + str(
                algorithm.module)
            assert algorithm.classname in ["TestAdapter1", "TestAdapterDatatypeInput",
                                           "TestAdapter2", "TestAdapter22", "TestAdapterHugeMemoryRequired",
                                           "TestAdapter3", "TestAdapterHDDRequired",
                                           "NDimensionArrayAdapter"
                                           ], "Unknown Adapter Class:" + str(algorithm.classname)
            if algorithm.module == 'tvb.tests.framework.adapters.testadapter2':
                nr_adapters_mod2 += 1
        assert nr_adapters_mod2 == 2
