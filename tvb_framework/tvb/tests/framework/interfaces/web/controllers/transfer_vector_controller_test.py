# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#
"""
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.interfaces.web.controllers.simulator.simulator_controller import SimulatorController
from tvb.interfaces.web.controllers.burst.transfer_vector_controller import TransferVectorController
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest
from tvb.core.entities.storage import dao


class TestTransferVectorController(BaseTransactionalControllerTest):
    """ Unit tests for TransferVectorController """

    def transactional_setup_method(self):
        """
        Sets up the environment for testing;
        creates a `RegionsModelParametersController` and a connectivity
        """
        self.init()
        self.controller = TransferVectorController()
        simulator_controller = SimulatorController()
        simulator_controller.index()
        self.simulator = simulator_controller.context.simulator

    def transactional_teardown_method(self):
        self.cleanup()

    def test_index_empty(self, operation_factory, connectivity_index_factory):
        """
        Verifies that result dictionary has the expected keys / values after call to `index()`
        """
        op = operation_factory(test_project=self.test_project)
        conn = connectivity_index_factory(op=op)
        self.simulator.connectivity = conn.gid

        count = dao.count_datatypes(self.test_project.id, ConnectivityMeasureIndex)
        assert 0 == count
        result_dict = self.controller.index()
        assert 'mainContent' in result_dict
        assert 'burst/transfer_function_apply_empty' == result_dict['mainContent']

    def test_index(self, operation_factory, connectivity_index_factory, connectivity_measure_index_factory):
        """
        Verifies that result dictionary has the expected keys / values after call to `index()`
        """
        op = operation_factory(test_project=self.test_project)
        conn = connectivity_index_factory(op=op)
        self.simulator.connectivity = conn.gid

        connectivity_measure_index_factory(conn, op, self.test_project)
        count = dao.count_datatypes(self.test_project.id, ConnectivityMeasureIndex)
        assert 1 == count

        result_dict = self.controller.index()
        expected_keys = ['applyTransferFunctionForm', 'isSingleMode', 'submit_parameters_url',
                         'parametersTransferFunctionPlotForm']
        assert all(x in result_dict for x in expected_keys)
        assert result_dict['baseUrl'] == '/burst/transfer/'
        assert result_dict['mainContent'] == 'burst/transfer_function_apply'

    def test_graph(self, operation_factory, connectivity_index_factory, connectivity_measure_index_factory):
        op = operation_factory(test_project=self.test_project)
        conn = connectivity_index_factory(op=op)
        self.simulator.connectivity = conn.gid
        connectivity_measure_index_factory(conn, op, self.test_project)
        self.controller.index()

        result_dict = self.controller.get_equation_chart()
        assert isinstance(result_dict['allSeries'], str)
