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

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Paula Sanz Leon <paula@tvb.invalid>

"""

import numpy
from tvb.adapters.datatypes.db.spectral import ComplexCoherenceSpectrumIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.h5.spectral_h5 import ComplexCoherenceSpectrumH5
from tvb.analyzers.node_complex_coherence import calculate_complex_cross_coherence, complex_coherence_result_shape
from tvb.basic.neotraits.api import Attr, Int, Float
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.time_series import TimeSeries


class NodeComplexCoherenceModel(ViewModel):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label="Time Series",
        required=True,
        doc="""The timeseries for which the CrossCoherence and ComplexCoherence is to be computed."""
    )

    epoch_length = Float(
        label="Epoch length [ms]",
        default=1000.0,
        required=False,
        doc="""In general for lengthy EEG recordings (~30 min), the timeseries are divided into equally
        sized segments (~ 20-40s). These contain the  event that is to be characterized by means of the
        cross coherence. Additionally each epoch block will be further divided into segments to  which
        the FFT will be applied.""")

    segment_length = Float(
        label="Segment length [ms]",
        default=500.0,
        required=False,
        doc="""The timeseries can be segmented into equally sized blocks (overlapping if necessary).
        The segment length determines the frequency resolution of the resulting power spectra --
        longer windows produce finer frequency resolution. """)

    segment_shift = Float(
        label="Segment shift [ms]",
        default=250.0,
        required=False,
        doc="""Time length by which neighboring segments are shifted. e.g.
        `segment shift` = `segment_length` / 2 means 50% overlapping segments.""")

    window_function = Attr(
        field_type=str,
        label="Windowing function",
        default='hanning',
        required=False,
        doc="""Windowing functions can be applied before the FFT is performed. Default is `hanning`,
        possibilities are: 'hamming'; 'bartlett'; 'blackman'; and 'hanning'. See, numpy.<function_name>.""")

    average_segments = Attr(
        field_type=bool,
        label="Average across segments",
        default=True,
        required=False,
        doc="""Flag. If `True`, compute the mean Cross Spectrum across  segments.""")

    subtract_epoch_average = Attr(
        field_type=bool,
        label="Subtract average across epochs",
        default=True,
        required=False,
        doc="""Flag. If `True` and if the number of epochs is > 1, you can optionally subtract the
        mean across epochs before computing the complex coherence.""")

    zeropad = Int(
        label="Zeropadding",
        default=0,
        required=False,
        doc="""Adds `n` zeros at the end of each segment and at the end of window_function.
        It is not yet functional.""")

    detrend_ts = Attr(
        field_type=bool,
        label="Detrend time series",
        default=False,
        required=False,
        doc="""Flag. If `True` removes linear trend along the time dimension before applying FFT.""")

    max_freq = Float(
        label="Maximum frequency",
        default=1024.0,
        required=False,
        doc="""Maximum frequency points (e.g. 32., 64., 128.) represented in the output.
        Default is segment_length / 2 + 1.""")

    npat = Float(
        label="dummy variable",
        default=1.0,
        required=False,
        doc="""This attribute appears to be related to an input projection matrix... Which is not yet implemented""")


class NodeComplexCoherenceForm(ABCAdapterForm):

    def __init__(self):
        super(NodeComplexCoherenceForm, self).__init__()
        self.time_series = TraitDataTypeSelectField(NodeComplexCoherenceModel.time_series, name=self.get_input_name(),
                                                    conditions=self.get_filters())

    @staticmethod
    def get_view_model():
        return NodeComplexCoherenceModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.data_ndim'], operations=["=="], values=[4])

    @staticmethod
    def get_input_name():
        return "time_series"


class NodeComplexCoherenceAdapter(ABCAdapter):
    """ TVB adapter for calling the NodeComplexCoherence algorithm. """

    _ui_name = "Complex Coherence of Nodes"
    _ui_description = "Compute the node complex (imaginary) coherence for a TimeSeries input DataType."
    _ui_subsection = "complexcoherence"

    def get_form_class(self):
        return NodeComplexCoherenceForm

    def get_output(self):
        return [ComplexCoherenceSpectrumIndex]

    def get_required_memory_size(self, view_model):
        # type: (NodeComplexCoherenceModel) -> int
        """
        Return the required memory to run this algorithm.
        """
        input_size = numpy.prod(self.input_shape) * 8.0
        output_size = self.result_size(self.input_shape,
                                       view_model.max_freq, view_model.epoch_length, view_model.segment_length,
                                       view_model.segment_shift, self.input_time_series_index.sample_period,
                                       view_model.zeropad, view_model.average_segments)

        return input_size + output_size

    def get_required_disk_size(self, view_model):
        # type: (NodeComplexCoherenceModel) -> int
        """
        Returns the required disk size to be able to run the adapter (in kB).
        """
        result = self.result_size(self.input_shape, view_model.max_freq, view_model.epoch_length,
                                  view_model.segment_length, view_model.segment_shift,
                                  self.input_time_series_index.sample_period, view_model.zeropad,
                                  view_model.average_segments)
        return self.array_size2kb(result)

    def configure(self, view_model):
        # type: (NodeComplexCoherenceModel) -> None
        """
        Do any configuration needed before launching
        """
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series)
        self.input_shape = (self.input_time_series_index.data_length_1d,
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)
        self.log.debug("Time series shape is %s" % (str(self.input_shape)))

    def launch(self, view_model):
        # type: (NodeComplexCoherenceModel) -> [ComplexCoherenceSpectrumIndex]
        """
        Launch algorithm and build results.
        :param view_model: the ViewModel keeping the algorithm inputs
        :return: the complex coherence for the specified time series
        """
        # TODO ---------- Iterate over slices and compose final result ------------##
        time_series = h5.load_from_index(self.input_time_series_index)
        ht_result = calculate_complex_cross_coherence(time_series, view_model.epoch_length,
                                                      view_model.segment_length,
                                                      view_model.segment_shift,
                                                      view_model.window_function,
                                                      view_model.average_segments,
                                                      view_model.subtract_epoch_average,
                                                      view_model.zeropad, view_model.detrend_ts,
                                                      view_model.max_freq, view_model.npat)
        self.log.debug("got ComplexCoherenceSpectrum result")
        self.log.debug("ComplexCoherenceSpectrum segment_length is %s" % (str(ht_result.segment_length)))
        self.log.debug("ComplexCoherenceSpectrum epoch_length is %s" % (str(ht_result.epoch_length)))
        self.log.debug("ComplexCoherenceSpectrum windowing_function is %s" % (str(ht_result.windowing_function)))

        complex_coherence_index = self.store_complete(ht_result)

        result_path = self.path_for(ComplexCoherenceSpectrumH5, complex_coherence_index.gid)
        ica_h5 = ComplexCoherenceSpectrumH5(path=result_path)

        self.fill_index_from_h5(complex_coherence_index, ica_h5)
        ica_h5.close()

        return complex_coherence_index

    @staticmethod
    def result_size(input_shape, max_freq, epoch_length, segment_length,
                    segment_shift, sample_period, zeropad, average_segments):
        """
        Returns the storage size in Bytes of the main result (complex array) of
        the ComplexCoherence
        """
        result_size = numpy.prod(complex_coherence_result_shape(input_shape, max_freq,
                                                                epoch_length, segment_length,
                                                                segment_shift, sample_period,
                                                                zeropad, average_segments)[0]) * 2.0 * 8.0
        return result_size
