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
Adapter that uses the traits module to generate interfaces for ... Analyzer.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>

"""

import json
import uuid

import numpy
from scipy.signal.signaltools import correlate
from tvb.adapters.datatypes.db.graph import CorrelationCoefficientsIndex
from tvb.adapters.datatypes.db.temporal_correlations import CrossCorrelationIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex, TimeSeriesEEGIndex, TimeSeriesMEGIndex, \
    TimeSeriesSEEGIndex
from tvb.adapters.datatypes.h5.temporal_correlations_h5 import CrossCorrelationH5
from tvb.basic.neotraits.api import HasTraits, Attr, Float
from tvb.basic.neotraits.info import narray_describe
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import FloatField, TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.graph import CorrelationCoefficients
from tvb.datatypes.temporal_correlations import CrossCorrelation
from tvb.datatypes.time_series import TimeSeries


class CrossCorrelate(HasTraits):
    """
    Model class defining the traited attributes used by the CrossCorrelateAdapter.
    """
    time_series = Attr(
        field_type=TimeSeries,
        label="Time Series",
        required=True,
        doc="""The time-series for which the cross correlation sequences are calculated.""")


class CrossCorrelateAdapterModel(ViewModel, CrossCorrelate):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label="Time Series",
        required=True,
        doc="""The time-series for which the cross correlation sequences are calculated."""
    )


class CrossCorrelateAdapterForm(ABCAdapterForm):

    def __init__(self, project_id=None):
        super(CrossCorrelateAdapterForm, self).__init__(project_id)
        self.time_series = TraitDataTypeSelectField(CrossCorrelateAdapterModel.time_series, self.project_id,
                                                    name=self.get_input_name(), conditions=self.get_filters(),
                                                    has_all_option=True)

    @staticmethod
    def get_view_model():
        return CrossCorrelateAdapterModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_input_name():
        return 'time_series'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.data_ndim'], operations=["=="], values=[4])


class CrossCorrelateAdapter(ABCAdapter):
    """ TVB adapter for calling the CrossCorrelate algorithm. """
    _ui_name = "Cross-correlation of nodes"
    _ui_description = "Cross-correlate two one-dimensional arrays."
    _ui_subsection = "crosscorr"

    def get_form_class(self):
        return CrossCorrelateAdapterForm

    def get_output(self):
        return [CrossCorrelationIndex]

    def configure(self, view_model):
        # type: (CrossCorrelateAdapterModel) -> None
        """
        Store the input shape to be later used to estimate memory usage.

        :param time_series: the input time-series index for which cross correlation should be computed
        """
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series)
        self.input_shape = (self.input_time_series_index.data_length_1d,
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)

    def get_required_memory_size(self, view_model):
        # type: (CrossCorrelateAdapterModel) -> int
        """
        Returns the required memory to be able to run the adapter.
        """
        # Not all the data is loaded into memory at one time here.
        used_shape = (self.input_shape[0], 1, self.input_shape[2], self.input_shape[3])
        input_size = numpy.prod(used_shape) * 8.0
        output_size = self._result_size(used_shape)
        return input_size + output_size

    def get_required_disk_size(self, view_model):
        # type: (CrossCorrelateAdapterModel) -> int
        """
        Returns the required disk size to be able to run the adapter (in kB).
        """
        used_shape = (self.input_shape[0], 1, self.input_shape[2], self.input_shape[3])
        return self.array_size2kb(self._result_size(used_shape))

    def launch(self, view_model):
        # type: (CrossCorrelateAdapterModel) -> [CrossCorrelationIndex]
        """ 
        Launch algorithm and build results.
        Compute the node-pairwise cross-correlation of the source 4D TimeSeries represented by the index given as input.

        Return a CrossCorrelationIndex. Create a CrossCorrelationH5 that contains the cross-correlation
        sequences for all possible combinations of the nodes.

        See: http://www.scipy.org/doc/api_docs/SciPy.signal.signaltools.html#correlate

        :param time_series: the input time series index for which the correlation should be computed
        :returns: the cross correlation index for the given time series
        :rtype: `CrossCorrelationIndex`
        """
        # --------- Prepare CrossCorrelationIndex and CrossCorrelationH5 objects for result ------------##
        cross_corr_index = CrossCorrelationIndex()
        cross_corr_h5_path = h5.path_for(self.storage_path, CrossCorrelationH5, cross_corr_index.gid)
        cross_corr_h5 = CrossCorrelationH5(cross_corr_h5_path)

        node_slice = [slice(self.input_shape[0]), None, slice(self.input_shape[2]), slice(self.input_shape[3])]
        # ---------- Iterate over slices and compose final result ------------##
        small_ts = TimeSeries()

        with h5.h5_file_for_index(self.input_time_series_index) as ts_h5:
            small_ts.sample_period = ts_h5.sample_period.load()
            small_ts.sample_period_unit = ts_h5.sample_period_unit.load()
            partial_cross_corr = None
            labels_ordering = ts_h5.labels_ordering.load()
            for var in range(self.input_shape[1]):
                node_slice[1] = slice(var, var + 1)
                small_ts.data = ts_h5.read_data_slice(tuple(node_slice))
                partial_cross_corr = self._compute_cross_correlation(small_ts, ts_h5)
                cross_corr_h5.write_data_slice(partial_cross_corr)
            ts_array_metadata = cross_corr_h5.array_data.get_cached_metadata()

        cross_corr_h5.time.store(partial_cross_corr.time)
        cross_corr_labels_ordering = list(partial_cross_corr.labels_ordering)
        cross_corr_labels_ordering[1] = labels_ordering[2]
        cross_corr_labels_ordering[2] = labels_ordering[2]
        cross_corr_h5.labels_ordering.store(json.dumps(tuple(cross_corr_labels_ordering)))
        cross_corr_h5.source.store(uuid.UUID(self.input_time_series_index.gid))
        cross_corr_h5.gid.store(uuid.UUID(cross_corr_index.gid))

        cross_corr_index.fk_source_gid = self.input_time_series_index.gid
        cross_corr_index.labels_ordering = cross_corr_h5.labels_ordering.load()
        cross_corr_index.type = type(cross_corr_index).__name__
        cross_corr_index.array_data_min = ts_array_metadata.min
        cross_corr_index.array_data_max = ts_array_metadata.max
        cross_corr_index.array_data_mean = ts_array_metadata.mean

        cross_corr_h5.close()
        return cross_corr_index

    def _compute_cross_correlation(self, small_ts, input_ts_h5):
        """
        Cross-correlate two one-dimensional arrays. Return a CrossCorrelation datatype with result.
        """
        # (tpts, nodes, nodes, state-variables, modes)
        result_shape = self._result_shape(small_ts.data.shape)
        self.log.info("result shape will be: %s" % str(result_shape))

        result = numpy.zeros(result_shape)

        # TODO: For region level, 4s, 2000Hz, this takes ~3hours...(which makes node_coherence seem positively speedy
        # Probably best to add a keyword for offsets, so we just compute +- some "small" range...
        # One inter-node correlation, across offsets, for each state-var & mode.
        for mode in range(result_shape[4]):
            for var in range(result_shape[3]):
                data = input_ts_h5.data[:, var, :, mode]
                data = data - data.mean(axis=0)[numpy.newaxis, :]
                # TODO: Work out a way around the 4 level loop:
                for n1 in range(result_shape[1]):
                    for n2 in range(result_shape[2]):
                        result[:, n1, n2, var, mode] = correlate(data[:, n1], data[:, n2], mode="same")

        self.log.debug("result")
        self.log.debug(narray_describe(result))

        offset = (small_ts.sample_period *
                  numpy.arange(-numpy.floor(result_shape[0] / 2.0), numpy.ceil(result_shape[0] / 2.0)))

        cross_corr = CrossCorrelation(source=small_ts, array_data=result, time=offset)

        return cross_corr

    @staticmethod
    def _result_shape(input_shape):
        """Returns the shape of the main result of ...."""
        result_shape = (input_shape[0], input_shape[2], input_shape[2], input_shape[1], input_shape[3])
        return result_shape

    def _result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the main result of .
        """
        result_size = numpy.sum(list(map(numpy.prod, self._result_shape(input_shape)))) * 8.0  # Bytes
        return result_size


class CorrelationCoefficient(HasTraits):
    """
    Model class defining the traited attributes used by the CorrelationCoefficientAdapter.
    """
    time_series = Attr(
        field_type=TimeSeries,
        label="Time Series",
        required=True,
        doc="""The time-series for which the cross correlation matrices are
        calculated.""")

    t_start = Float(
        label=":math:`t_{start}`",
        default=0.9765625,
        required=True,
        doc="""Time start point (ms). By default it uses the default Monitor sample period.
        The starting time point of a time series is not zero, but the monitor's sample period. """)

    t_end = Float(
        label=":math:`t_{end}`",
        default=1000.,
        required=True,
        doc=""" End time point (ms) """)


class PearsonCorrelationCoefficientAdapterModel(ViewModel, CorrelationCoefficient):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label="Time Series",
        required=True,
        doc="""The time-series for which the cross correlation matrices are
            calculated."""
    )


class PearsonCorrelationCoefficientAdapterForm(ABCAdapterForm):

    def __init__(self, project_id=None):
        super(PearsonCorrelationCoefficientAdapterForm, self).__init__(project_id)
        self.time_series = TraitDataTypeSelectField(PearsonCorrelationCoefficientAdapterModel.time_series,
                                                    self.project_id, name=self.get_input_name(),
                                                    conditions=self.get_filters(), has_all_option=True)
        self.t_start = FloatField(PearsonCorrelationCoefficientAdapterModel.t_start, self.project_id)
        self.t_end = FloatField(PearsonCorrelationCoefficientAdapterModel.t_end, self.project_id)

    @staticmethod
    def get_view_model():
        return PearsonCorrelationCoefficientAdapterModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_input_name():
        return 'time_series'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.data_ndim'], operations=["=="], values=[4])

    def get_traited_datatype(self):
        return CorrelationCoefficient()


class PearsonCorrelationCoefficientAdapter(ABCAdapter):
    """ TVB adapter for calling the Pearson correlation coefficients algorithm. """

    _ui_name = "Pearson correlation coefficients"
    _ui_description = "Cross Correlation"
    _ui_subsection = "ccpearson"

    def get_form_class(self):
        return PearsonCorrelationCoefficientAdapterForm

    def get_output(self):
        return [CorrelationCoefficientsIndex]

    def configure(self, view_model):
        # type: (PearsonCorrelationCoefficientAdapterModel) -> None
        """
        Store the input shape to be later used to estimate memory usage.

        :param time_series: the input time-series index for which correlation coefficient should be computed
        :param t_start: the physical time interval start for the analysis
        :param t_end: physical time, interval end
        """
        if view_model.t_start >= view_model.t_end or view_model.t_start < 0:
            raise LaunchException("Can not launch operation without monitors selected !!!")

        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series)
        self.input_shape = (int((view_model.t_end - view_model.t_start) / self.input_time_series_index.sample_period),
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)

    def get_required_memory_size(self, view_model):
        # type: (PearsonCorrelationCoefficientAdapterModel) -> int
        """
        Returns the required memory to be able to run this adapter.
        """
        in_memory_input = [self.input_shape[0], 1, self.input_shape[2], 1]
        input_size = numpy.prod(in_memory_input) * 8.0
        output_size = self._result_size(self.input_shape)
        return input_size + output_size

    def get_required_disk_size(self, view_model):
        # type: (PearsonCorrelationCoefficientAdapterModel) -> int
        """
        Returns the required disk size to be able to run the adapter (in kB).
        """
        output_size = self._result_size(self.input_shape)
        return self.array_size2kb(output_size)

    def launch(self, view_model):
        # type: (PearsonCorrelationCoefficientAdapterModel) -> [CorrelationCoefficientsIndex]
        """
        Launch algorithm and build results.
        Compute the node-pairwise pearson correlation coefficient of the given input 4D TimeSeries  datatype.

        The result will contain values between -1 and 1, inclusive.

        :param time_series: the input time-series for which correlation coefficient should be computed
        :param t_start: the physical time interval start for the analysis
        :param t_end: physical time, interval end
        :returns: the correlation coefficient for the given time series
        :rtype: `CorrelationCoefficients`
        """
        with h5.h5_file_for_index(self.input_time_series_index) as ts_h5:
            ts_labels_ordering = ts_h5.labels_ordering.load()
            result = self._compute_correlation_coefficients(ts_h5, view_model.t_start, view_model.t_end)

        if isinstance(self.input_time_series_index, TimeSeriesEEGIndex) \
                or isinstance(self.input_time_series_index, TimeSeriesMEGIndex) \
                or isinstance(self.input_time_series_index, TimeSeriesSEEGIndex):
            labels_ordering = ["Sensor", "Sensor", "1", "1"]
        else:
            labels_ordering = list(CorrelationCoefficients.labels_ordering.default)
            labels_ordering[0] = ts_labels_ordering[2]
            labels_ordering[1] = ts_labels_ordering[2]

        corr_coef = CorrelationCoefficients()
        corr_coef.array_data = result
        corr_coef.source = TimeSeries(gid=view_model.time_series)
        corr_coef.labels_ordering = labels_ordering

        return h5.store_complete(corr_coef, self.storage_path)

    def _compute_correlation_coefficients(self, ts_h5, t_start, t_end):
        """
        Compute the correlation coefficients of a 2D array (tpts x nodes).
        Yields an array of size nodes x nodes x state-variables x modes.

        The time interval over which the correlation coefficients are computed
        is defined by t_start, t_end

        See: http://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
        """
        # (nodes, nodes, state-variables, modes)
        input_shape = ts_h5.data.shape
        result_shape = self._result_shape(input_shape)
        self.log.info("result shape will be: %s" % str(result_shape))

        result = numpy.zeros(result_shape)

        t_lo = int(
            (1. / self.input_time_series_index.sample_period) * (t_start - self.input_time_series_index.sample_period))
        t_hi = int(
            (1. / self.input_time_series_index.sample_period) * (t_end - self.input_time_series_index.sample_period))
        t_lo = max(t_lo, 0)
        t_hi = max(t_hi, input_shape[0])

        # One correlation coeff matrix, for each state-var & mode.
        for mode in range(result_shape[3]):
            for var in range(result_shape[2]):
                current_slice = tuple([slice(t_lo, t_hi + 1), slice(var, var + 1),
                                       slice(input_shape[2]), slice(mode, mode + 1)])
                data = ts_h5.data[current_slice].squeeze()
                result[:, :, var, mode] = numpy.corrcoef(data.T)

        self.log.debug("result")
        self.log.debug(narray_describe(result))

        return result

    @staticmethod
    def _result_shape(input_shape):
        """Returns the shape of the main result of ...."""
        result_shape = (input_shape[2], input_shape[2], input_shape[1], input_shape[3])
        return result_shape

    def _result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the main result of .
        """
        result_size = numpy.sum(list(map(numpy.prod, self._result_shape(input_shape)))) * 8.0  # Bytes
        return result_size
