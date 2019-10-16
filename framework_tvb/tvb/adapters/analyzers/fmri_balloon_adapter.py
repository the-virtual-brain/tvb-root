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
Adapter that uses the traits module to generate interfaces for BalloonModel Analyzer.

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import uuid
import numpy
from tvb.analyzers.fmri_balloon import BalloonModel
from tvb.datatypes.time_series import TimeSeries
from tvb.core.adapters.abcadapter import ABCAsynchronous, ABCAdapterForm
from tvb.core.entities.filters.chain import FilterChain
from tvb.basic.logger.builder import get_logger
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesRegionH5
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex, TimeSeriesRegionIndex
from tvb.core.neotraits.forms import DataTypeSelectField, ScalarField
from tvb.core.neotraits.db import prepare_array_shape_meta
from tvb.core.neocom import h5

LOG = get_logger(__name__)


class BalloonModelAdapterForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(BalloonModelAdapterForm, self).__init__(prefix, project_id)
        self.time_series = DataTypeSelectField(self.get_required_datatype(), self, name=self.get_input_name(),
                                               required=True, label=BalloonModel.time_series.label,
                                               doc=BalloonModel.time_series.doc, conditions=self.get_filters(),
                                               has_all_option=True)
        self.dt = ScalarField(BalloonModel.dt, self)
        self.neural_input_transformation = ScalarField(BalloonModel.neural_input_transformation, self)
        self.bold_model = ScalarField(BalloonModel.bold_model, self)
        self.RBM = ScalarField(BalloonModel.RBM, self)

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.data_ndim'], operations=["=="], values=[4])

    @staticmethod
    def get_input_name():
        return 'time_series'

    def get_traited_datatype(self):
        return BalloonModel()


class BalloonModelAdapter(ABCAsynchronous):
    """
    TVB adapter for calling the BalloonModel algorithm.
    """

    _ui_name = "Balloon Model "
    _ui_description = "Compute BOLD signals for a TimeSeries input DataType."
    _ui_subsection = "balloon"

    def get_form_class(self):
        return BalloonModelAdapterForm

    def get_output(self):
        return [TimeSeriesRegionIndex]

    def configure(self, time_series, dt=None, bold_model=None, RBM=None, neural_input_transformation=None):
        """
        Store the input shape to be later used to estimate memory usage. Also
        create the algorithm instance.
        """
        self.input_time_series_index = time_series
        self.input_shape = (self.input_time_series_index.data_length_1d,
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)

        LOG.debug("time_series shape is %s" % str(self.input_shape))
        # -------------------- Fill Algorithm for Analysis -------------------##
        algorithm = BalloonModel()

        if dt is not None:
            algorithm.dt = dt
        else:
            algorithm.dt = time_series.sample_period / 1000.

        if bold_model is not None:
            algorithm.bold_model = bold_model
        if RBM is not None:
            algorithm.RBM = RBM
        if neural_input_transformation is not None:
            algorithm.neural_input_transformation = neural_input_transformation

        self.algorithm = algorithm

    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        used_shape = self.input_shape
        input_size = numpy.prod(used_shape) * 8.0
        output_size = self.algorithm.result_size(used_shape)
        return input_size + output_size

    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter.(in kB)
        """
        used_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
        return self.array_size2kb(self.algorithm.result_size(used_shape))

    def launch(self, time_series, dt=None, bold_model=None, RBM=None, neural_input_transformation=None):
        """
        Launch algorithm and build results.

        :param time_series: the input time-series used as neural activation in the Balloon Model
        :returns: the simulated BOLD signal
        :rtype: `TimeSeries`
        """
        input_time_series_h5 = h5.h5_file_for_index(time_series)
        time_line = input_time_series_h5.read_time_page(0, self.input_shape[0])

        bold_signal_index = TimeSeriesRegionIndex()
        bold_signal_h5_path = h5.path_for(self.storage_path, TimeSeriesRegionH5, bold_signal_index.gid)
        bold_signal_h5 = TimeSeriesRegionH5(bold_signal_h5_path)
        bold_signal_h5.gid.store(uuid.UUID(bold_signal_index.gid))
        self._fill_result_h5(bold_signal_h5, input_time_series_h5)

        ##---------- Iterate over slices and compose final result ------------##

        node_slice = [slice(self.input_shape[0]), slice(self.input_shape[1]), None, slice(self.input_shape[3])]
        small_ts = TimeSeries()
        small_ts.sample_period = self.input_time_series_index.sample_period
        small_ts.time = time_line

        for node in range(self.input_shape[2]):
            node_slice[2] = slice(node, node + 1)
            small_ts.data = input_time_series_h5.read_data_slice(tuple(node_slice))
            self.algorithm.time_series = small_ts
            partial_bold = self.algorithm.evaluate()
            bold_signal_h5.write_data_slice_on_grow_dimension(partial_bold.data, grow_dimension=2)

        bold_signal_h5.write_time_slice(time_line)
        bold_signal_shape = bold_signal_h5.data.shape
        bold_signal_h5.nr_dimensions.store(len(bold_signal_shape))
        bold_signal_h5.close()
        input_time_series_h5.close()

        self._fill_result_index(bold_signal_index, bold_signal_shape)
        return bold_signal_index

    def _fill_result_index(self, result_index, result_signal_shape):
        result_index.time_series_type = type(result_index).__name__
        result_index.data_ndim = len(result_signal_shape)
        result_index.data_length_1d, result_index.data_length_2d, \
        result_index.data_length_3d, result_index.data_length_3d = \
            prepare_array_shape_meta(result_signal_shape)

        result_index.connectivity_gid = self.input_time_series_index.connectivity_gid
        result_index.region_mapping_gid = self.input_time_series_index.region_mapping_gid
        result_index.region_mapping_volume_gid = self.input_time_series_index.region_mapping_volume_gid

        result_index.sample_period = self.input_time_series_index.sample_period
        result_index.sample_period_unit = self.input_time_series_index.sample_period_unit
        result_index.sample_rate = self.input_time_series_index.sample_rate
        result_index.labels_ordering = self.input_time_series_index.labels_ordering
        result_index.labels_dimensions = self.input_time_series_index.labels_dimensions
        result_index.has_volume_mapping = self.input_time_series_index.has_volume_mapping
        result_index.title = self.input_time_series_index.title

    def _fill_result_h5(self, result_h5, input_h5):
        result_h5.sample_period.store(self.input_time_series_index.sample_period)
        result_h5.sample_period_unit.store(self.input_time_series_index.sample_period_unit)
        result_h5.sample_rate.store(input_h5.sample_rate.load())
        result_h5.start_time.store(input_h5.start_time.load())
        result_h5.labels_ordering.store(input_h5.labels_ordering.load())
        result_h5.labels_dimensions.store(input_h5.labels_dimensions.load())
        result_h5.connectivity.store(input_h5.connectivity.load())
        result_h5.region_mapping_volume.store(input_h5.region_mapping_volume.load())
        result_h5.region_mapping.store(input_h5.region_mapping.load())
        result_h5.title.store(input_h5.title.load())
