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
.. moduleauthor:: Dan Pop <dan.pop@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import numpy
import json
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.model import DataTypeGroup, OperationGroup, STATUS_STARTED
from tvb.core.entities.storage import dao
from tvb.core.adapters.exceptions import LaunchException
from tvb.datatypes.mapped_values import DatatypeMeasure
from tvb.basic.filters.chain import FilterChain


class PseIsoModel(object):
    def __init__(self, range1, range2, apriori_data, metrics, datatype_gids):
        self.log = get_logger(self.__class__.__name__)
        # ranges
        all_numbers_range1, self.range1_name, self.range1 = OperationGroup.load_range_numbers(range1)
        all_numbers_range2, self.range2_name, self.range2 = OperationGroup.load_range_numbers(range2)

        # Data from which to interpolate larger 2-D space
        self.apriori_x = self._prepare_axes(self.range1, all_numbers_range1)
        self.apriori_y = self._prepare_axes(self.range2, all_numbers_range2)
        self.apriori_data = apriori_data
        self.datatypes_gids = datatype_gids
        self.metrics = metrics

    @classmethod
    def from_db(cls, operation_group_id):
        """
        Collects from db the information about the operation group that is required by the isocline view.
        """
        operations = dao.get_operations_in_group(operation_group_id)
        operation_group = dao.get_operationgroup_by_id(operation_group_id)

        self = cls(operation_group.range1, operation_group.range2, {},
                   PseIsoModel._find_metrics(operations), None)

        self._fill_apriori_data(operations)
        return self

    @staticmethod
    def _find_metrics(operations):
        """ Search for an operation with results. Then get the metrics of the generated data type"""
        dt_measure = None

        for operation in operations:

            if not operation.has_finished:
                raise LaunchException("Can not display until all operations from this range are finished!")

            op_results = dao.get_results_for_operation(operation.id)
            if len(op_results):
                datatype = op_results[0]
                if datatype.type == "DatatypeMeasure":
                    ## Load proper entity class from DB.
                    dt_measure = dao.get_generic_entity(DatatypeMeasure, datatype.id)[0]
                else:
                    dt_measure = dao.get_generic_entity(DatatypeMeasure, datatype.gid, '_analyzed_datatype')
                    if dt_measure:
                        dt_measure = dt_measure[0]
                break

        if dt_measure:
            return dt_measure.metrics
        else:
            raise LaunchException("No datatypes were generated due to simulation errors. Nothing to display.")

    def _fill_apriori_data(self, operations):
        """ Gather apriori data from the operations. Also gather the datatype gid's"""
        for metric in self.metrics:
            self.apriori_data[metric] = numpy.zeros((self.apriori_x.size, self.apriori_y.size))

        # An 2D array of GIDs which is used later to launch overlay for a DataType
        self.datatypes_gids = [[None for _ in self.range2] for _ in self.range1]

        for operation in operations:
            self.log.debug("Gathering data from operation : %s" % operation.id)
            range_values = eval(operation.range_values)
            key_1 = range_values[self.range1_name]
            index_x = self.range1.index(key_1)
            key_2 = range_values[self.range2_name]
            index_y = self.range2.index(key_2)
            if operation.status == STATUS_STARTED:
                raise LaunchException("Not all operations from this range are complete. Cannot view until then.")

            operation_results = dao.get_results_for_operation(operation.id)
            if operation_results:
                datatype = operation_results[0]
                self.datatypes_gids[index_x][index_y] = str(datatype.gid)

                if datatype.type == "DatatypeMeasure":
                    measures = dao.get_generic_entity(DatatypeMeasure, datatype.id)
                else:
                    measures = dao.get_generic_entity(DatatypeMeasure, datatype.gid, '_analyzed_datatype')
            else:
                self.datatypes_gids[index_x][index_y] = None
                measures = None

            for metric in self.metrics:
                if measures:
                    self.apriori_data[metric][index_x][index_y] = measures[0].metrics[metric]
                else:
                    self.apriori_data[metric][index_x][index_y] = numpy.NaN

    @staticmethod
    def _prepare_axes(original_range_values, is_numbers):
        result = numpy.array(original_range_values)
        if not is_numbers:
            result = numpy.arange(len(original_range_values))
        return result


class IsoclinePSEAdapter(ABCDisplayer):
    """
    Visualization adapter for Parameter Space Exploration.
    Will be used as a generic visualizer, accessible when input entity is DataTypeGroup.
    Will also be used in Burst as a supplementary navigation layer.
    """

    _ui_name = "Isocline Parameter Space Exploration"
    _ui_subsection = "pse_iso"

    def __init__(self):
        ABCDisplayer.__init__(self)
        self.interp_models = {}
        self.nan_indices = {}

    def get_input_tree(self):
        """
        Take as Input a Connectivity Object.
        """
        return [{'name': 'datatype_group',
                 'label': 'Datatype Group',
                 'type': DataTypeGroup,
                 'required': True,
                 'conditions': FilterChain(fields=[FilterChain.datatype + ".no_of_ranges"],
                                           operations=["=="], values=[2])}]


    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        # Don't know how much memory is needed.
        return -1


    def burst_preview(self, datatype_group_gid):
        """
        Generate the preview for the burst page.
        """
        datatype_group = dao.get_datatype_group_by_gid(datatype_group_gid)
        return self.launch(datatype_group=datatype_group)


    def get_metric_matrix(self, datatype_group, selected_metric=None):
        self.model = PseIsoModel.from_db(datatype_group.fk_operation_group)
        if selected_metric is None:
            selected_metric = self.model.metrics.keys()[0]

        data_matrix = self.model.apriori_data[selected_metric]
        data_matrix = numpy.rot90(data_matrix)
        data_matrix = numpy.flipud(data_matrix)
        matrix_data = ABCDisplayer.dump_with_precision(data_matrix.flat)
        matrix_guids = self.model.datatypes_gids
        matrix_guids = numpy.rot90(matrix_guids)
        matrix_shape = json.dumps(data_matrix.squeeze().shape)
        x_min = self.model.apriori_x[0]
        x_max = self.model.apriori_x[self.model.apriori_x.size - 1]
        y_min = self.model.apriori_y[0]
        y_max = self.model.apriori_y[self.model.apriori_y.size - 1]
        vmin = data_matrix.min()
        vmax = data_matrix.max()
        return dict(matrix_data=matrix_data,
                    matrix_guids=json.dumps(matrix_guids.flatten().tolist()),
                    matrix_shape=matrix_shape,
                    color_metric=selected_metric,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    vmin=vmin,
                    vmax=vmax)


    @staticmethod
    def prepare_node_data(datatype_group):
        if datatype_group is None:
            raise Exception("Selected DataTypeGroup is no longer present in the database. "
                            "It might have been remove or the specified id is not the correct one.")

        operation_group = dao.get_operationgroup_by_id(datatype_group.fk_operation_group)
        operations = dao.get_operations_in_group(operation_group.id)
        node_info_dict = dict()
        for operation_ in operations:
            datatypes = dao.get_results_for_operation(operation_.id)
            if len(datatypes) > 0:
                datatype = datatypes[0]
                node_info_dict[datatype.gid] = dict(operation_id=operation_.id,
                                                    datatype_gid=datatype.gid,
                                                    datatype_type=datatype.type,
                                                    datatype_subject=datatype.subject,
                                                    datatype_invalid=datatype.invalid)
        return node_info_dict


    def launch(self, datatype_group, **kwargs):
        params = self.get_metric_matrix(datatype_group)
        params["title"] = self._ui_name
        params["canvasName"] = "Interpolated values for PSE metric: "
        params["xAxisName"] = self.model.range1_name
        params["yAxisName"] = self.model.range2_name
        params["url_base"] = "/burst/explore/get_metric_matrix/" + datatype_group.gid
        params["node_info_url"] = "/burst/explore/get_node_matrix/" + datatype_group.gid
        params["available_metrics"] = self.model.metrics.keys()
        return self.build_display_result('pse_isocline/view', params,
                                         pages=dict(controlPage="pse_isocline/controls"))
