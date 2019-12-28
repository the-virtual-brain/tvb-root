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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.adapters.visualizers.pse_discrete import DiscretePSEAdapter
from tvb.adapters.visualizers.pse_isocline import IsoclinePSEAdapter


class TestPSE(TransactionalTestCase):
    """
    Unit-tests for BrainViewer.
    """

    def test_launch_discrete(self, datatype_group_factory):
        """
        Check that all required keys are present in output from PSE Discrete Adapter launch.
        """
        group = datatype_group_factory()
        viewer = DiscretePSEAdapter()
        result = viewer.launch(group)

        expected_keys = ['status', 'size_metric', 'series_array', 'min_shape_size', 'min_color', 'd3_data',
                         'max_shape_size', 'max_color', 'mainContent', 'labels_y', 'labels_x', 'isAdapter',
                         'has_started_ops', 'datatype_group_gid', 'datatypes_dict', 'color_metric']
        for key in expected_keys:
            assert key in result
        assert group.gid == result["datatype_group_gid"]
        assert 'false' == result["has_started_ops"]

    def test_launch_isocline(self, datatype_group_factory):
        """
        Check that all required keys are present in output from PSE Discrete Adapter launch.
        """
        viewer = IsoclinePSEAdapter()
        group = datatype_group_factory()
        result = viewer.launch(group)
        assert viewer._ui_name == result["title"]
        assert 1 == len(result["available_metrics"])
