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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>

Data for Parameter Space Exploration view will be defined here.
The purpose of this entities is to be used in Jinja2 UI, or for populating visualizer.
"""

import json
import math
import numpy
from tvb.adapters.datatypes.db.mapped_value import DatatypeMeasureIndex
from tvb.basic.config.utils import EnhancedDictionary
from tvb.core.entities.storage import dao

KEY_GID = "Gid"
KEY_TOOLTIP = "tooltip"
LINE_SEPARATOR = "<br/>"


class PSEModel(object):
    def __init__(self, operation):
        self.operation = operation
        self.datatype_measure = self.determine_operation_result()
        self.metrics = dict()
        if self.datatype_measure:
            self.metrics = json.loads(self.datatype_measure.metrics)
        self.range1_key = None
        self.range1_value = None
        self.range2_key = None
        self.range2_value = 0
        self._extract_range_values()

    def _extract_range_values(self):
        ranges = json.loads(self.operation.range_values)
        range_keys = list(ranges.keys())
        self.range1_key = range_keys[0]
        self.range1_value = ranges[self.range1_key]

        if len(range_keys) > 1:
            self.range2_key = range_keys[1]
            self.range2_value = ranges[self.range2_key]

    def __is_float(self, value):
        is_float = True
        try:
            float(value)
        except ValueError:
            is_float = False
        return is_float

    def is_range1_float(self):
        return self.__is_float(self.range1_value)

    def is_range2_float(self):
        if not self.range2_key:
            return False
        return self.__is_float(self.range2_value)

    def _determine_type_and_label(self, value):
        if self.__is_float(value):
            return value

        label = dao.get_datatype_by_gid(value).display_name
        return label

    def get_range1_label(self):
        return self._determine_type_and_label(self.range1_value)

    def get_range2_label(self):
        if not self.range2_key:
            return ''
        return self._determine_type_and_label(self.range2_value)

    def prepare_node_info(self):
        """
        Build a dictionary with all the required information to be displayed for a given node.
        """
        KEY_NODE_TYPE = "dataType"
        KEY_OPERATION_ID = "operationId"

        node_info = dict()
        if self.operation.has_finished and self.datatype_measure is not None:
            node_info[KEY_GID] = self.datatype_measure.gid
            node_info[KEY_NODE_TYPE] = self.datatype_measure.type
            node_info[KEY_OPERATION_ID] = self.operation.id
            node_info[KEY_TOOLTIP] = self._prepare_node_text()
        else:
            tooltip = "No result available. Operation is in status: %s" % self.operation.status.split('-')[1]
            node_info[KEY_TOOLTIP] = tooltip
        return node_info

    def _prepare_node_text(self):
        """
        Prepare text to display as tooltip for a PSE circle
        :return: str
        """
        return str("Operation id: " + str(self.operation.id) + LINE_SEPARATOR +
                   "Datatype gid: " + str(self.datatype_measure.gid) + LINE_SEPARATOR +
                   "Datatype type: " + str(self.datatype_measure.type) + LINE_SEPARATOR +
                   "Datatype subject: " + str(self.datatype_measure.subject) + LINE_SEPARATOR +
                   "Datatype invalid: " + str(self.datatype_measure.invalid))

    def determine_operation_result(self):
        datatype_measure = None
        if self.operation.has_finished:
            datatypes = dao.get_results_for_operation(self.operation.id)
            if len(datatypes) > 0:
                datatype = datatypes[0]
                if datatype.type == "DatatypeMeasureIndex":
                    # Load proper entity class from DB.
                    measures = dao.get_generic_entity(DatatypeMeasureIndex, datatype.gid, 'gid')
                else:
                    measures = dao.get_generic_entity(DatatypeMeasureIndex, datatype.gid, 'fk_source_gid')
                if len(measures) > 0:
                    datatype_measure = measures[0]

        return datatype_measure


class PSEGroupModel(object):
    def __init__(self, datatype_group_gid):
        self.datatype_group = dao.get_datatype_group_by_gid(datatype_group_gid)

        if self.datatype_group is None:
            raise Exception("Selected DataTypeGroup is no longer present in the database. "
                            "It might have been remove or the specified id is not the correct one.")

        self.operation_group = dao.get_operationgroup_by_id(self.datatype_group.fk_operation_group)
        self.operations = dao.get_operations_in_group(self.operation_group.id)
        self.pse_model_list = self.parse_pse_data_for_display()
        self.all_metrics = dict()
        self._prepare_ranges_data()

    @property
    def range1_values(self):
        if self.pse_model_list[0].is_range1_float():
            return self.range1_orig_values
        return list(range(len(self.range1_orig_values)))

    @property
    def range2_values(self):
        if self.pse_model_list[0].is_range2_float():
            return self.range2_orig_values
        return list(range(len(self.range2_orig_values)))

    def parse_pse_data_for_display(self):
        pse_model_list = []
        for operation in self.operations:
            pse_model = PSEModel(operation)
            pse_model_list.append(pse_model)
        return pse_model_list

    def get_range1_key(self):
        return self.pse_model_list[0].range1_key

    def get_range2_key(self):
        return self.pse_model_list[0].range2_key

    def _prepare_ranges_data(self):
        value_to_label1 = dict()
        value_to_label2 = dict()
        for pse_model in self.pse_model_list:
            label1 = pse_model.get_range1_label()
            value_to_label1.update({pse_model.range1_value: label1})
            label2 = pse_model.get_range2_label()
            value_to_label2.update({pse_model.range2_value: label2})

        value_to_label1 = dict(sorted(value_to_label1.items()))
        value_to_label2 = dict(sorted(value_to_label2.items()))
        self.range1_orig_values = list(value_to_label1.keys())
        self.range1_labels = list(value_to_label1.values())
        self.range2_orig_values = list(value_to_label2.keys())
        self.range2_labels = list(value_to_label2.values())

    def get_all_node_info(self):
        all_node_info = dict()
        for pse_model in self.pse_model_list:
            node_info = pse_model.prepare_node_info()
            range1_val = self.range1_values[self.range1_orig_values.index(pse_model.range1_value)]
            range2_val = self.range2_values[self.range2_orig_values.index(pse_model.range2_value)]
            if not range1_val in all_node_info:
                all_node_info[range1_val] = {}
            all_node_info[range1_val][range2_val] = node_info
        return all_node_info

    def get_all_metrics(self):
        if len(self.all_metrics) == 0:
            for pse_model in self.pse_model_list:
                if pse_model.datatype_measure:
                    self.all_metrics.update({pse_model.datatype_measure.gid: pse_model.metrics})
        return self.all_metrics

    def get_available_metric_keys(self):
        return list(self.pse_model_list[0].metrics)


class DiscretePSE(EnhancedDictionary):
    def __init__(self, datatype_group_gid, color_metric, size_metric, back_page):
        super(DiscretePSE, self).__init__()
        self.datatype_group_gid = datatype_group_gid
        self.min_color = float('inf')
        self.max_color = - float('inf')
        self.min_shape_size = float('inf')
        self.max_shape_size = - float('inf')
        self.has_started_ops = False
        self.status = 'started'
        self.title_x, self.title_y = "", ""
        self.values_x, self.labels_x, self.values_y, self.labels_y = [], [], [], []
        self.series_array, self.available_metrics = [], []
        self.d3_data = {}
        self.color_metric = color_metric
        self.size_metric = size_metric
        self.pse_back_page = back_page

    def prepare_individual_jsons(self):
        """
        Apply JSON.dumps on all attributes which can not be passes as they are towards UI.
        """
        self.labels_x = json.dumps(self.labels_x)
        self.labels_y = json.dumps(self.labels_y)
        self.values_x = json.dumps(self.values_x)
        self.values_y = json.dumps(self.values_y)
        self.d3_data = json.dumps(self.d3_data)
        self.has_started_ops = json.dumps(self.has_started_ops)


class PSEDiscreteGroupModel(PSEGroupModel):

    def __init__(self, datatype_group_gid, color_metric, size_metric, back_page):
        super(PSEDiscreteGroupModel, self).__init__(datatype_group_gid)
        self.color_metric = color_metric
        self.size_metric = size_metric
        self.determine_default_metrics()
        self.pse_context = DiscretePSE(datatype_group_gid, self.color_metric, self.size_metric, back_page)
        self.fill_pse_context()
        self.prepare_display_data()

    def determine_default_metrics(self):
        if self.color_metric is None and self.size_metric is None:
            metrics = self.get_available_metric_keys()
            if len(metrics) > 0:
                self.color_metric = metrics[0]
                self.size_metric = metrics[0]
            if len(metrics) > 1:
                self.size_metric = metrics[1]

    def fill_pse_context(self):
        sizes_array = [metrics[self.size_metric] for metrics in self.get_all_metrics().values()]
        colors_array = [metrics[self.color_metric] for metrics in self.get_all_metrics().values()]
        if len(sizes_array) > 0:
            self.pse_context.min_shape_size = numpy.min(sizes_array)
            self.pse_context.max_shape_size = numpy.max(sizes_array)
            self.pse_context.min_color = numpy.min(colors_array)
            self.pse_context.max_color = numpy.max(colors_array)
        self.pse_context.available_metrics = self.get_available_metric_keys()
        self.pse_context.title_x = self.get_range1_key()
        self.pse_context.title_y = self.get_range2_key()
        self.pse_context.values_x = self.range1_values
        self.pse_context.values_y = self.range2_values
        self.pse_context.labels_x = self.range1_labels
        self.pse_context.labels_y = self.range2_labels

    def _prepare_coords_json(self, val1, val2):
        return '{"x":' + str(val1) + ', "y":' + str(val2) + '}'

    def prepare_display_data(self):
        d3_data = self.get_all_node_info()
        series_array = []

        for val1 in d3_data.keys():
            for val2 in d3_data[val1].keys():
                current = d3_data[val1][val2]
                coords = self._prepare_coords_json(val1, val2)
                datatype_gid = None
                if KEY_GID in current:
                    # This means the operation was finished
                    datatype_gid = current[KEY_GID]

                color_weight, shape_type_1 = self.__get_color_weight(datatype_gid)
                if (shape_type_1 is not None) and (datatype_gid is not None):
                    d3_data[val1][val2][KEY_TOOLTIP] += LINE_SEPARATOR + " Color metric has NaN values"
                current['color_weight'] = color_weight

                radius, shape_type_2 = self.__get_node_size(datatype_gid, len(self.range1_values),
                                                            len(self.range2_values))
                if (shape_type_2 is not None) and (datatype_gid is not None):
                    d3_data[val1][val2][KEY_TOOLTIP] += LINE_SEPARATOR + " Size metric has NaN values"
                symbol = shape_type_1 or shape_type_2

                series_array.append(self.__get_node_json(symbol, radius, coords))

        self.pse_context.d3_data = d3_data
        self.pse_context.series_array = self.__build_series_json(series_array)

    @staticmethod
    def __build_series_json(list_of_series):
        """ Given a list with all the data points, build the final FLOT JSON. """
        final_json = "["
        for i, value in enumerate(list_of_series):
            if i:
                final_json += ","
            final_json += value
        final_json += "]"
        return final_json

    @staticmethod
    def __get_node_json(symbol, radius, coords):
        """
        For each data point entry, build the FLOT specific JSON.
        """
        series = '{"points": {'
        if symbol is not None:
            series += '"symbol": "' + symbol + '", '
        series += '"radius": ' + str(radius) + '}, '
        series += '"coords": ' + coords + '}'
        return series

    def __get_node_size(self, datatype_gid, range1_length, range2_length):
        """
        Computes the size of the shape used for representing the dataType with GID given.
        """
        min_size, max_size = self.__get_boundaries(range1_length, range2_length)
        if datatype_gid is None:
            return max_size / 2.0, "cross"
        node_info = self.get_all_metrics()[datatype_gid]

        if self.size_metric in node_info:
            valid_metric = True
            try:
                if math.isnan(float(node_info[self.size_metric])) or math.isinf(float(node_info[self.size_metric])):
                    valid_metric = False
            except ValueError:
                valid_metric = False

            if valid_metric:
                shape_weight = node_info[self.size_metric]
                values_range = self.pse_context.max_shape_size - self.pse_context.min_shape_size
                shape_range = max_size - min_size
                if values_range != 0:
                    return min_size + (shape_weight - self.pse_context.min_shape_size) / float(
                        values_range) * shape_range, None
                else:
                    return min_size, None
            else:
                return max_size / 2.0, "cross"
        return max_size / 2.0, None

    def __get_color_weight(self, datatype_gid):
        """
        Returns the color weight of the shape used for representing the dataType which id equal to 'datatype_gid'.
        :param: datatype_gid - It should exists into the 'datatype_indexes' dictionary.
        """
        if datatype_gid is None:
            return 0, "cross"
        node_info = self.get_all_metrics()[datatype_gid]
        valid_metric = True

        if self.color_metric in node_info:
            try:
                if math.isnan(float(node_info[self.color_metric])) or math.isinf(float(node_info[self.color_metric])):
                    valid_metric = False
            except ValueError:
                valid_metric = False
            if valid_metric:
                return node_info[self.color_metric], None
            else:
                return 0, "cross"
        return 0, None

    @staticmethod
    def __get_boundaries(range1_length, range2_length):
        """
        Returns the MIN and the max values of the interval from
        which may be set the size of a certain shape.
        """
        # the arrays 'intervals' and 'values' should have the same size
        intervals = [0, 3, 5, 8, 10, 20, 30, 40, 50, 60, 90, 110, 120]
        values = [(10, 50), (10, 40), (10, 33), (8, 25), (5, 15), (4, 10),
                  (3, 8), (2, 6), (2, 5), (1, 4), (1, 3), (1, 2), (1, 2)]

        max_length = max([range1_length, range2_length])
        if max_length <= intervals[0]:
            return values[0]
        elif max_length >= intervals[-1]:
            return values[-1]
        else:
            for i, interval in enumerate(intervals):
                if max_length <= interval:
                    return values[i - 1]
