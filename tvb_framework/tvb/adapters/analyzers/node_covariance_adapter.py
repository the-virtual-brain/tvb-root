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
Adapter that uses the traits module to generate interfaces for FFT Analyzer.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""
import uuid

import numpy
from tvb.adapters.datatypes.db.graph import CovarianceIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.h5.graph_h5 import CovarianceH5
from tvb.basic.neotraits.info import narray_describe
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.graph import Covariance
from tvb.datatypes.time_series import TimeSeries


class NodeCovarianceAdapterModel(ViewModel):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label="Time Series",
        required=True,
        doc="""The timeseries to which the NodeCovariance is to be applied."""
    )


class NodeCovarianceAdapterForm(ABCAdapterForm):
    def __init__(self):
        super(NodeCovarianceAdapterForm, self).__init__()
        self.time_series = TraitDataTypeSelectField(NodeCovarianceAdapterModel.time_series, name=self.get_input_name(),
                                                    conditions=self.get_filters(), has_all_option=True)

    @staticmethod
    def get_view_model():
        return NodeCovarianceAdapterModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_input_name():
        return 'time_series'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.data_ndim'], operations=["=="], values=[4])


class NodeCovarianceAdapter(ABCAdapter):
    """ TVB adapter for calling the NodeCovariance algorithm. """
    _ui_name = "Temporal covariance of nodes"
    _ui_description = "Compute Temporal Node Covariance for a TimeSeries input DataType."
    _ui_subsection = "covariance"

    def get_form_class(self):
        return NodeCovarianceAdapterForm

    def get_output(self):
        return [CovarianceIndex]

    def configure(self, view_model):
        # type: (NodeCovarianceAdapterModel) -> None
        """
        Store the input shape to be later used to estimate memory usage.
        """
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series)
        self.input_shape = (self.input_time_series_index.data_length_1d,
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)

    def get_required_memory_size(self, view_model):
        # type: (NodeCovarianceAdapterModel) -> int
        """
        Return the required memory to run this algorithm.
        """
        used_shape = (self.input_shape[0], 1, self.input_shape[2], 1)
        input_size = numpy.prod(used_shape) * 8.0
        output_size = self._result_size(used_shape)
        return input_size + output_size

    def get_required_disk_size(self, view_model):
        # type: (NodeCovarianceAdapterModel) -> int
        """
        Returns the required disk size to be able to run the adapter ( in kB).
        """
        used_shape = (self.input_shape[0], 1, self.input_shape[2], 1)
        return self.array_size2kb(self._result_size(used_shape))

    def launch(self, view_model):
        # type: (NodeCovarianceAdapterModel) -> [CovarianceIndex]
        """ 
        Launch algorithm and build results.
        :param view_model: the ViewModel keeping the algorithm inputs
        :return: the `CovarianceIndex` built with the given time_series index as source
        """
        # -------------------- Prepare result entities ---------------------##
        covariance_index = CovarianceIndex()
        covariance_h5_path = self.path_for(CovarianceH5, covariance_index.gid)
        covariance_h5 = CovarianceH5(covariance_h5_path)

        # ------------ NOTE: Assumes 4D, Simulator timeSeries -------------##
        node_slice = [slice(self.input_shape[0]), None, slice(self.input_shape[2]), None]
        ts_h5 = h5.h5_file_for_index(self.input_time_series_index)

        for mode in range(self.input_shape[3]):
            for var in range(self.input_shape[1]):
                small_ts = TimeSeries()
                node_slice[1] = slice(var, var + 1)
                node_slice[3] = slice(mode, mode + 1)
                small_ts.data = ts_h5.read_data_slice(tuple(node_slice))
                partial_cov = self._compute_node_covariance(small_ts, ts_h5)
                covariance_h5.write_data_slice(partial_cov.array_data)

        ts_h5.close()

        partial_cov.source.gid = view_model.time_series
        partial_cov.gid = uuid.UUID(covariance_index.gid)

        covariance_index.fill_from_has_traits(partial_cov)
        self.fill_index_from_h5(covariance_index, covariance_h5)

        covariance_h5.store(partial_cov, scalars_only=True)
        covariance_h5.close()
        return covariance_index

    def _compute_node_covariance(self, small_ts, input_ts_h5):
        """
        Compute the temporal covariance between nodes in a TimeSeries dataType.
        A nodes x nodes matrix is returned for each (state-variable, mode).
        """
        data_shape = small_ts.data.shape

        # (nodes, nodes, state-variables, modes)
        result_shape = (data_shape[2], data_shape[2], data_shape[1], data_shape[3])
        self.log.info("result shape will be: %s" % str(result_shape))

        result = numpy.zeros(result_shape)

        # One inter-node temporal covariance matrix for each state-var & mode.
        for mode in range(data_shape[3]):
            for var in range(data_shape[1]):
                data = input_ts_h5.data[:, var, :, mode]
                data = data - data.mean(axis=0)[numpy.newaxis, 0]
                result[:, :, var, mode] = numpy.cov(data.T)

        self.log.debug("result")
        self.log.debug(narray_describe(result))

        covariance = Covariance(source=small_ts, array_data=result)
        return covariance

    @staticmethod
    def _result_size(input_shape):
        """
        Returns the storage size in Bytes of the NodeCovariance result.
        """
        result_size = numpy.prod([input_shape[2], input_shape[2], input_shape[1], input_shape[3]]) * 8.0  # Bytes
        return result_size
