# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
import os
import pytest
from tvb.adapters.creators.siibra_creator import SiibraCreator, SiibraModel, CLB_AUTH_TOKEN_KEY
from tvb.tests.framework.adapters.creators import siibra_base_test
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory


@pytest.mark.skipif(siibra_base_test.no_ebrains_auth_token(), reason="No EBRAINS AUTH token for accesing the KG was provided!")
class TestSiibraCreator(TransactionalTestCase):
    """ Test Siibra Creator functionalities """

    def transactional_setup_method(self):
        self.siibra_creator = SiibraCreator()
        self.test_user = TestFactory.create_user('Siibra_Creator_Tests_User1')
        self.test_project = TestFactory.create_project(self.test_user, 'Siibra_Creator_Tests_Project1')

    def test_happy_flow_launch(self, operation_factory):
        view_model = SiibraModel()
        view_model.ebrains_token = os.environ[CLB_AUTH_TOKEN_KEY]
        view_model.subject_ids = '010'

        operation = operation_factory(test_user=self.test_user, test_project=self.test_project)
        self.siibra_creator.extract_operation_data(operation)
        results = self.siibra_creator.launch(view_model)

        conn_index = results[0]
        conn_measure_indices = results[1:]

        assert len(conn_measure_indices) == 5  # 5 for each connectivity

        # connectivities
        assert conn_index.has_hemispheres_mask
        assert conn_index.number_of_regions == 294
        assert conn_index.subject == '010'

        # connectivity measures
        for conn_measure in conn_measure_indices:
            assert conn_measure.parsed_shape == (294, 294)
            assert conn_measure.subject == '010'
            assert conn_measure.fk_connectivity_gid == conn_index.gid

