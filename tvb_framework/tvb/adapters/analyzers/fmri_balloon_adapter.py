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
Adapter that uses the traits module to generate interfaces for BalloonModel Analyzer.

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import uuid
import numpy

from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesRegionH5
from tvb.analyzers.fmri_balloon import BalloonModel, BoldModels, NeuralInputTransformations
from tvb.basic.neotraits.api import Float, Attr, EnumAttr
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.db import prepare_array_shape_meta
from tvb.core.neotraits.forms import TraitDataTypeSelectField, FloatField, StrField, BoolField, SelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.time_series import TimeSeries, TimeSeriesRegion


class BalloonModelAdapterModel(ViewModel):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeriesRegion,
        label="Time Series",
        required=True,
        doc="""The timeseries that represents the input neural activity"""
    )

    tau_s = Float(
        label=r":math:`\tau_s`",
        default=1.54,
        required=True,
        doc="""Balloon model parameter. Time of signal decay (s)""")

    tau_f = Float(
        label=r":math:`\tau_f`",
        default=1.44,
        required=True,
        doc=""" Balloon model parameter. Time of flow-dependent elimination or
            feedback regulation (s). The average  time blood take to traverse the
            venous compartment. It is the  ratio of resting blood volume (V0) to
            resting blood flow (F0).""")

    neural_input_transformation = EnumAttr(
        label="Neural input transformation",
        default=NeuralInputTransformations.NONE,
        doc=""" This represents the operation to perform on the state-variable(s) of
            the model used to generate the input TimeSeries. ``none`` takes the
            first state-variable as neural input; `` abs_diff`` is the absolute
            value of the derivative (first order difference) of the first state variable; 
            ``sum``: sum all the state-variables of the input TimeSeries."""
    )

    bold_model = EnumAttr(
        label="Select BOLD model equations",
        default=BoldModels.NONLINEAR,
        doc="""Select the set of equations for the BOLD model."""
    )

    RBM = Attr(
        field_type=bool,
        label="Revised BOLD Model",
        default=True,
        required=True,
        doc="""Select classical vs revised BOLD model (CBM or RBM).
            Coefficients  k1, k2 and k3 will be derived accordingly."""
    )

    normalize_neural_input = Attr(
        field_type=bool,
        label="Normalize neural input",
        default=False,
        required=True,
        doc="""Set if the mean should be subtracted from the neural input."""
    )


class BalloonModelAdapterForm(ABCAdapterForm):

    def __init__(self):
        super(BalloonModelAdapterForm, self).__init__()
        self.time_series = TraitDataTypeSelectField(BalloonModelAdapterModel.time_series, name=self.get_input_name(),
                                                    conditions=self.get_filters(), has_all_option=True)
        self.tau_s = FloatField(BalloonModelAdapterModel.tau_s)
        self.tau_f = FloatField(BalloonModelAdapterModel.tau_f)
        self.neural_input_transformation = SelectField(BalloonModelAdapterModel.neural_input_transformation)
        self.bold_model = SelectField(BalloonModelAdapterModel.bold_model)
        self.RBM = BoolField(BalloonModelAdapterModel.RBM)
        self.normalize_neural_input = BoolField(BalloonModelAdapterModel.normalize_neural_input)

    @staticmethod
    def get_view_model():
        return BalloonModelAdapterModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesRegionIndex

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.data_ndim'], operations=["=="], values=[4])

    @staticmethod
    def get_input_name():
        return 'time_series'


class BalloonModelAdapter(ABCAdapter):
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

    def configure(self, view_model):
        # type: (BalloonModelAdapterModel) -> None
        """
        Store the input shape to be later used to estimate memory usage. Also
        create the algorithm instance.
        """
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series)
        self.input_shape = (self.input_time_series_index.data_length_1d,
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)

        self.log.debug("time_series shape is %s" % str(self.input_shape))
        # -------------------- Fill Algorithm for Analysis -------------------##
        algorithm = BalloonModel()
        if view_model.tau_s is not None:
            algorithm.tau_s = view_model.tau_s
        if view_model.tau_f is not None:
            algorithm.tau_f = view_model.tau_f
        if view_model.bold_model is not None:
            algorithm.bold_model = view_model.bold_model
        if view_model.neural_input_transformation is not None:
            algorithm.neural_input_transformation = view_model.neural_input_transformation
        algorithm.RBM = view_model.RBM
        algorithm.normalize_neural_input = view_model.normalize_neural_input

        self.algorithm = algorithm

    def get_required_memory_size(self, view_model):
        # type: (BalloonModelAdapterModel) -> int
        """
        Return the required memory to run this algorithm.
        """
        used_shape = self.input_shape
        input_size = numpy.prod(used_shape) * 8.0
        output_size = self.algorithm.result_size(used_shape)
        return input_size + output_size

    def get_required_disk_size(self, view_model):
        # type: (BalloonModelAdapterModel) -> int
        """
        Returns the required disk size to be able to run the adapter.(in kB)
        """
        used_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
        return self.array_size2kb(self.algorithm.result_size(used_shape))

    def launch(self, view_model):
        # type: (BalloonModelAdapterModel) -> [TimeSeriesRegionIndex]
        """
        Launch algorithm and build results.
        :param view_model: the ViewModel keeping the algorithm inputs
        :return: the simulated BOLD signal
        """
        input_time_series_h5 = h5.h5_file_for_index(self.input_time_series_index)
        time_line = input_time_series_h5.read_time_page(0, self.input_shape[0])

        bold_signal_index = TimeSeriesRegionIndex()
        bold_signal_h5_path = self.path_for(TimeSeriesRegionH5, bold_signal_index.gid)
        bold_signal_h5 = TimeSeriesRegionH5(bold_signal_h5_path)
        bold_signal_h5.gid.store(uuid.UUID(bold_signal_index.gid))
        self._fill_result_h5(bold_signal_h5, input_time_series_h5)

        # ---------- Iterate over slices and compose final result ------------##

        node_slice = [slice(self.input_shape[0]), slice(self.input_shape[1]), None, slice(self.input_shape[3])]
        small_ts = TimeSeries()
        small_ts.sample_period = self.input_time_series_index.sample_period
        small_ts.sample_period_unit = self.input_time_series_index.sample_period_unit
        small_ts.time = time_line

        for node in range(self.input_shape[2]):
            node_slice[2] = slice(node, node + 1)
            small_ts.data = input_time_series_h5.read_data_slice(tuple(node_slice))
            self.algorithm.time_series = small_ts
            partial_bold = self.algorithm.evaluate()
            bold_signal_h5.write_data_slice_on_grow_dimension(partial_bold.data, grow_dimension=2)

        input_time_series_h5.close()

        bold_signal_h5.write_time_slice(time_line)
        bold_signal_shape = bold_signal_h5.data.shape
        bold_signal_h5.nr_dimensions.store(len(bold_signal_shape))
        bold_signal_h5.close()

        self._fill_result_index(bold_signal_index, bold_signal_shape)
        return bold_signal_index

    def _fill_result_index(self, result_index, result_signal_shape):
        result_index.time_series_type = TimeSeriesRegion.__name__
        result_index.data_ndim = len(result_signal_shape)
        result_index.data_length_1d, result_index.data_length_2d, \
        result_index.data_length_3d, result_index.data_length_4d = prepare_array_shape_meta(result_signal_shape)

        result_index.fk_connectivity_gid = self.input_time_series_index.fk_connectivity_gid
        result_index.fk_region_mapping_gid = self.input_time_series_index.fk_region_mapping_gid
        result_index.fk_region_mapping_volume_gid = self.input_time_series_index.fk_region_mapping_volume_gid

        result_index.sample_period = self.input_time_series_index.sample_period
        result_index.sample_period_unit = self.input_time_series_index.sample_period_unit
        result_index.sample_rate = self.input_time_series_index.sample_rate
        result_index.labels_ordering = self.input_time_series_index.labels_ordering
        result_index.labels_dimensions = self.input_time_series_index.labels_dimensions
        result_index.has_volume_mapping = self.input_time_series_index.has_volume_mapping
        result_index.has_surface_mapping = self.input_time_series_index.has_surface_mapping
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
