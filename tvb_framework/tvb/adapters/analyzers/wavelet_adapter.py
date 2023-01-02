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
Adapter that uses the traits module to generate interfaces for
ContinuousWaveletTransform Analyzer.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""

import uuid

import numpy
from tvb.adapters.datatypes.db.spectral import WaveletCoefficientsIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.h5.spectral_h5 import WaveletCoefficientsH5
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesH5
from tvb.analyzers.wavelet import compute_continuous_wavelet_transform
from tvb.basic.neotraits.api import Attr, Range, Float
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import FormField, Form, TraitDataTypeSelectField, StrField, FloatField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.time_series import TimeSeries


class WaveletAdapterModel(ViewModel):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label="Time Series",
        required=True,
        doc="""The timeseries to which the wavelet is to be applied.""")

    mother = Attr(
        field_type=str,
        label="Wavelet function",
        default="morlet",
        doc="""The mother wavelet function used in the transform. Default is
            'morlet', possibilities are: 'morlet'...""")

    sample_period = Float(
        label="Sample period of result (ms)",
        default=7.8125,  # 7.8125 => 128 Hz
        doc="""The sampling period of the computed wavelet spectrum. NOTE:
            This should be an integral multiple of the of the sampling period
            of the source time series, otherwise the actual resulting sample
            period will be the first correct value below that requested.""")

    frequencies = Attr(
        field_type=Range,
        label="Frequency range of result (kHz).",
        default=Range(lo=0.008, hi=0.060, step=0.002),
        doc="""The frequency resolution and range returned. Requested
            frequencies are converted internally into appropriate scales.""")

    normalisation = Attr(
        field_type=str,
        label="Normalisation",
        default="energy",
        doc="""The type of normalisation for the resulting wavet spectrum.
            Default is 'energy', options are: 'energy'; 'gabor'.""")

    q_ratio = Float(
        label="Q-ratio",
        default=5.0,
        doc="""NFC. Must be greater than 5. Ratios of the center frequencies to bandwidths.""")


class RangeForm(Form):
    def __init__(self):
        super(RangeForm, self).__init__()
        self.lo = FloatField(
            Float(label='Lo', default=WaveletAdapterModel.frequencies.default.lo, doc='start of range'),
            name='Lo')
        self.hi = FloatField(
            Float(label='Hi', default=WaveletAdapterModel.frequencies.default.hi, doc='end of range'),
            name='Hi')
        self.step = FloatField(
            Float(label='Step', default=WaveletAdapterModel.frequencies.default.step, doc='step of range'),
            name='Step')


class ContinuousWaveletTransformAdapterForm(ABCAdapterForm):

    def __init__(self):
        super(ContinuousWaveletTransformAdapterForm, self).__init__()
        self.time_series = TraitDataTypeSelectField(WaveletAdapterModel.time_series, name=self.get_input_name(),
                                                    conditions=self.get_filters(), has_all_option=True)
        self.mother = StrField(WaveletAdapterModel.mother)
        self.sample_period = FloatField(WaveletAdapterModel.sample_period)
        self.normalisation = StrField(WaveletAdapterModel.normalisation)
        self.q_ratio = FloatField(WaveletAdapterModel.q_ratio)
        self.frequencies = FormField(RangeForm, name='frequencies', label=WaveletAdapterModel.frequencies.label,
                                     doc=WaveletAdapterModel.frequencies.doc)

    @staticmethod
    def get_view_model():
        return WaveletAdapterModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    def fill_trait(self, datatype):
        super(ContinuousWaveletTransformAdapterForm, self).fill_trait(datatype)
        datatype.frequencies.lo = self.frequencies.form.lo.value
        datatype.frequencies.step = self.frequencies.form.step.value
        datatype.frequencies.hi = self.frequencies.form.hi.value

    @staticmethod
    def get_input_name():
        return 'time_series'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.data_ndim'], operations=["=="], values=[4])


class ContinuousWaveletTransformAdapter(ABCAdapter):
    """
    TVB adapter for calling the ContinuousWaveletTransform algorithm.
    """

    _ui_name = "Continuous Wavelet Transform"
    _ui_description = "Compute Wavelet Tranformation for a TimeSeries input DataType."
    _ui_subsection = "wavelet"

    def get_form_class(self):
        return ContinuousWaveletTransformAdapterForm

    def get_output(self):
        return [WaveletCoefficientsIndex]

    def configure(self, view_model):
        """
        Store the input shape to be later used to estimate memory usage
        """
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series)

        input_shape = []
        for length in [self.input_time_series_index.data_length_1d,
                       self.input_time_series_index.data_length_2d,
                       self.input_time_series_index.data_length_3d,
                       self.input_time_series_index.data_length_4d]:
            if length is not None:
                input_shape.append(length)

        self.input_shape = tuple(input_shape)
        self.log.debug("Time series shape is %s" % str(self.input_shape))

    def get_required_memory_size(self, view_model):
        """
        Return the required memory to run this algorithm.
        """
        used_shape = (self.input_shape[0],
                      self.input_shape[1],
                      1,
                      self.input_shape[3])
        input_size = numpy.prod(used_shape) * 8.0
        output_size = self.result_size(view_model.frequencies, view_model.sample_period,
                                       used_shape, self.input_time_series_index.sample_period)
        return input_size + output_size

    def get_required_disk_size(self, view_model):
        """
        Returns the required disk size to be able to run the adapter.(in kB)
        """
        used_shape = (self.input_shape[0],
                      self.input_shape[1],
                      1,
                      self.input_shape[3])
        return self.array_size2kb(self.result_size(view_model.frequencies, view_model.sample_period,
                                                   used_shape, self.input_time_series_index.sample_period))

    def launch(self, view_model):
        # type: (WaveletAdapterModel) -> (WaveletCoefficientsIndex)
        """ 
        Launch algorithm and build results.
        :param view_model: the ViewModel keeping the algorithm inputs
        :return: the wavelet coefficients for the specified time series
        """
        frequencies_array = numpy.array([])
        if view_model.frequencies is not None:
            frequencies_array = view_model.frequencies.to_array()

        time_series_h5 = h5.h5_file_for_index(self.input_time_series_index)
        assert isinstance(time_series_h5, TimeSeriesH5)

        # --------------------- Prepare result entities ----------------------##
        wavelet_index = WaveletCoefficientsIndex()
        dest_path = self.path_for(WaveletCoefficientsH5, wavelet_index.gid)
        wavelet_h5 = WaveletCoefficientsH5(path=dest_path)

        # ------------- NOTE: Assumes 4D, Simulator timeSeries. --------------##
        node_slice = [slice(self.input_shape[0]), slice(self.input_shape[1]), None, slice(self.input_shape[3])]

        # ---------- Iterate over slices and compose final result ------------##
        small_ts = TimeSeries()
        small_ts.sample_period = time_series_h5.sample_period.load()
        small_ts.sample_period_unit = time_series_h5.sample_period_unit.load()
        for node in range(self.input_shape[2]):
            node_slice[2] = slice(node, node + 1)
            small_ts.data = time_series_h5.read_data_slice(tuple(node_slice))
            partial_wavelet = compute_continuous_wavelet_transform(small_ts, view_model.frequencies,
                                                                   view_model.sample_period,
                                                                   view_model.q_ratio, view_model.normalisation,
                                                                   view_model.mother)
            wavelet_h5.write_data_slice(partial_wavelet)

        time_series_h5.close()

        partial_wavelet.source.gid = view_model.time_series
        partial_wavelet.gid = uuid.UUID(wavelet_index.gid)

        wavelet_index.fill_from_has_traits(partial_wavelet)
        self.fill_index_from_h5(wavelet_index, wavelet_h5)

        wavelet_h5.store(partial_wavelet, scalars_only=True)
        wavelet_h5.frequencies.store(frequencies_array)
        wavelet_h5.close()

        return wavelet_index

    @staticmethod
    def result_shape(frequencies, sample_period, input_shape, input_sample_period):
        """
        Returns the shape of the main result (complex array) of the continuous
        wavelet transform.
        """
        freq_len = int((frequencies.hi - frequencies.lo) / frequencies.step)
        temporal_step = max((1, sample_period / input_sample_period))
        nt = int(round(input_shape[0] / temporal_step))
        result_shape = (freq_len, nt,) + input_shape[1:]
        return result_shape

    def result_size(self, frequencies, sample_period, input_shape, input_sample_period):
        """
        Returns the storage size in Bytes of the main result (complex array) of
        the continuous wavelet transform.
        """
        result_size = numpy.prod(
            self.result_shape(frequencies, sample_period, input_shape,
                              input_sample_period)) * 2.0 * 8.0  # complex*Bytes
        return result_size
