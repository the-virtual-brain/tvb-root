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
.. moduleauthor:: Gabriel Florea <gabriel.florea@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
from tvb.adapters.uploaders.networkx_importer import NetworkxImporterModel, NetworkxConnectivityImporter
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestNetworkxImporter(TransactionalTestCase):
    """
    Unit-tests for NetworkxImporter
    """

    upload_file = os.path.join(os.path.dirname(__file__), "test_data", 'connectome_83.gpickle')

    def transactional_setup_method(self):
        self.test_user = TestFactory.create_user('Networkx_User')
        self.test_project = TestFactory.create_project(self.test_user, "Networkx_Project")

    def transactional_teardown_method(self):
        FilesHelper().remove_project_structure(self.test_project.name)

    def test_import(self):
        count_before = self.count_all_entities(ConnectivityIndex)
        assert 0 == count_before

        view_model = NetworkxImporterModel()
        view_model.data_file = self.upload_file
        TestFactory.launch_importer(NetworkxConnectivityImporter, view_model, self.test_user, self.test_project.id)

        count_after = self.count_all_entities(ConnectivityIndex)
        assert 1 == count_after

        conn = self.get_all_entities(ConnectivityIndex)[0]
        assert 83 == conn.number_of_regions
