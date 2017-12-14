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
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory
from tvb.interfaces.web.controllers.burst.exploration_controller import ParameterExplorationController


class TestExplorationController(BaseTransactionalControllerTest):
    """
    Unit tests ParameterExplorationController
    """

    def transactional_setup_method(self):
        """
        Sets up the environment for testing;
        creates a datatype group and a Parameter Exploration Controller
        """
        self.init()
        self.dt_group = DatatypesFactory().create_datatype_group()
        self.controller = ParameterExplorationController()


    def transactional_teardown_method(self):
        """ Cleans the testing environment """
        self.cleanup()


    def test_draw_discrete_exploration(self):
        """
        Test that Discrete PSE is getting launched and correct fields are prepared.
        """
        result = self.controller.draw_discrete_exploration(self.dt_group.gid, 'burst', None, None)
        assert result['available_metrics'] == DatatypesFactory.DATATYPE_MEASURE_METRIC.keys()
        assert result['color_metric'] == DatatypesFactory.DATATYPE_MEASURE_METRIC.keys()[0]
        assert result['size_metric'] is None
        assert DatatypesFactory.RANGE_1[1] == json.loads(result['labels_x'])
        assert DatatypesFactory.RANGE_2[1] == json.loads(result['labels_y'])
        data = json.loads(result['d3_data'])
        assert len(data) == len(DatatypesFactory.RANGE_1[1])
        for row in data.values():
            assert len(row) == len(DatatypesFactory.RANGE_2[1])
            for entry in row.values():
                assert entry['dataType'] == 'Datatype2'
                for key in ['Gid', 'color_weight', 'operationId', 'tooltip']:
                    assert key in entry


    def test_draw_isocline_exploration(self):
        """
        Test that isocline PSE gets launched.
        """
        result = self.controller.draw_isocline_exploration(self.dt_group.gid)
        assert isinstance(result['canvasName'], (str, unicode))
        assert isinstance(result['xAxisName'], (str, unicode))
        assert isinstance(result['url_base'], (str, unicode))
        assert DatatypesFactory.DATATYPE_MEASURE_METRIC.keys() == result['available_metrics']
