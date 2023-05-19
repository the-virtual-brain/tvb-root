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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import json
import numpy

from tvb.core.entities.storage import dao
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
        dt_group, _ = datatype_group_factory()

        viewer = DiscretePSEAdapter()
        view_model = viewer.get_view_model_class()()
        view_model.datatype_group = dt_group.gid
        result = viewer.launch(view_model)

        expected_keys = ['status', 'size_metric', 'series_array', 'min_shape_size', 'min_color', 'd3_data',
                         'max_shape_size', 'max_color', 'mainContent', 'labels_y', 'labels_x', 'isAdapter',
                         'has_started_ops', 'datatype_group_gid', 'color_metric', 'values_x', 'values_y']
        for key in expected_keys:
            assert key in result
        assert json.loads(result['values_x']) == [1, 3, 5]
        assert json.loads(result['values_y']) == [0.1, 0.4]
        assert dt_group.gid == result["datatype_group_gid"]
        assert 'false' == result["has_started_ops"]

    def _add_extra_operation_in_group(self, dt_group, operation_factory, time_series_factory, time_series_index_factory,
                                      datatype_measure_factory):
        extra_operation = operation_factory(range_values=json.dumps({'row1': 2, 'row2': 0.1}))
        extra_operation.fk_operation_group = dt_group.fk_operation_group
        extra_operation = dao.store_entity(extra_operation)

        ts = time_series_factory()
        datatype = time_series_index_factory(ts=ts, op=extra_operation)
        datatype.fk_datatype_group = dt_group.id
        datatype.operation_id = extra_operation.id
        dao.store_entity(datatype)

        extra_operation_ms = operation_factory(range_values=json.dumps({'row1': 2, 'row2': 0.1}))
        extra_operation_ms.fk_operation_group = dt_group.fk_operation_group
        extra_operation_ms = dao.store_entity(extra_operation_ms)
        datatype_measure_factory(datatype, ts,  extra_operation_ms, dt_group, '{"v": 4}')

    def test_launch_discrete_order_operations(self, datatype_group_factory, operation_factory, time_series_factory,
                                              time_series_index_factory, datatype_measure_factory):
        """
        Check that all required keys are present in output from PSE Discrete Adapter launch.
        """
        dt_group, _ = datatype_group_factory()
        self._add_extra_operation_in_group(dt_group, operation_factory, time_series_factory, time_series_index_factory,
                                           datatype_measure_factory)

        viewer = DiscretePSEAdapter()
        view_model = viewer.get_view_model_class()()
        view_model.datatype_group = dt_group.gid
        result = viewer.launch(view_model)

        expected_keys = ['status', 'size_metric', 'series_array', 'min_shape_size', 'min_color', 'd3_data',
                         'max_shape_size', 'max_color', 'mainContent', 'labels_y', 'labels_x', 'isAdapter',
                         'has_started_ops', 'datatype_group_gid', 'color_metric', 'values_x', 'values_y']
        for key in expected_keys:
            assert key in result
        assert json.loads(result['values_x']) == [1, 2, 3, 5]
        assert json.loads(result['values_y']) == [0.1, 0.4]
        d3_data = json.loads(result['d3_data'])
        assert d3_data['3']['0.1']['color_weight'] == 3
        assert d3_data['2']['0.1']['color_weight'] == 4
        assert dt_group.gid == result["datatype_group_gid"]
        assert 'false' == result["has_started_ops"]

    def test_launch_isocline(self, datatype_group_factory):
        """
        Check that all required keys are present in output from PSE Discrete Adapter launch.
        """
        dt_group, _ = datatype_group_factory()

        viewer = IsoclinePSEAdapter()
        view_model = viewer.get_view_model_class()()
        view_model.datatype_group = dt_group.gid
        result = viewer.launch(view_model)

        assert viewer._ui_name == result["title"]
        assert 1 == len(result["available_metrics"])

    def test_launch_isocline_order_operations(self, datatype_group_factory, operation_factory, time_series_factory,
                                              time_series_index_factory, datatype_measure_factory):
        dt_group, _ = datatype_group_factory()
        self._add_extra_operation_in_group(dt_group, operation_factory, time_series_factory, time_series_index_factory,
                                           datatype_measure_factory)

        viewer = IsoclinePSEAdapter()
        view_model = viewer.get_view_model_class()()
        view_model.datatype_group = dt_group.gid
        result = viewer.launch(view_model)

        assert viewer._ui_name == result["title"]
        assert 1 == len(result["available_metrics"])
        matrix_shape = json.loads(result['matrix_shape'])
        matrix_data = json.loads(result['matrix_data'])
        matrix_data = numpy.reshape(matrix_data, matrix_shape)
        assert matrix_data[0][1] == 4
        # We replace NaN with vmin-1, which is 2 in this case:
        assert matrix_data[1][1] == 2
