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
Plot the power of a WaveletCoefficients object

.. moduleauthor:: Dan Pop <dan.pop@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
"""

import json

from tvb.adapters.datatypes.db.spectral import WaveletCoefficientsIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.spectral import WaveletCoefficients


class WaveletSpectrogramVisualizerModel(ViewModel):
    input_data = DataTypeGidAttr(
        linked_datatype=WaveletCoefficients,
        label='Wavelet transform Result',
        doc='Wavelet spectrogram to display'
    )


class WaveletSpectrogramVisualizerForm(ABCAdapterForm):
    # TODO: add all fields here
    def __init__(self):
        super(WaveletSpectrogramVisualizerForm, self).__init__()
        self.input_data = TraitDataTypeSelectField(WaveletSpectrogramVisualizerModel.input_data, name='input_data',
                                                   conditions=self.get_filters())

    @staticmethod
    def get_view_model():
        return WaveletSpectrogramVisualizerModel

    @staticmethod
    def get_required_datatype():
        return WaveletCoefficientsIndex

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_input_name():
        return 'input_data'


class WaveletSpectrogramVisualizer(ABCDisplayer):
    """
    Plot the power of a WaveletCoefficients object using SVG an D3.
    """
    _ui_name = "Spectrogram of Wavelet Power"
    _ui_subsection = "wavelet"

    def get_form_class(self):
        return WaveletSpectrogramVisualizerForm

    def get_required_memory_size(self, view_model):
        # type: (WaveletSpectrogramVisualizerModel) -> int
        """
         Return the required memory to run this algorithm.
         """
        wavelet_idx = self.load_entity_by_gid(view_model.input_data)
        shape = wavelet_idx.shape

        return shape[0] * shape[1] * 8

    def launch(self, view_model):
        # type: (WaveletSpectrogramVisualizerModel) -> dict

        with h5.h5_file_for_gid(view_model.input_data) as input_h5:
            shape = input_h5.array_data.shape
            input_sample_period = input_h5.sample_period.load()
            input_frequencies = input_h5.frequencies.load()
            ts_index = self.load_entity_by_gid(input_h5.source.load())

            slices = (slice(shape[0]),
                      slice(shape[1]),
                      slice(0, 1, None),
                      slice(0, shape[3], None),
                      slice(0, 1, None))
            data_matrix = input_h5.power[slices]
            data_matrix = data_matrix.sum(axis=3)

        assert isinstance(ts_index, TimeSeriesIndex)

        wavelet_sample_period = ts_index.sample_period * max((1, int(input_sample_period / ts_index.sample_period)))
        end_time = ts_index.start_time + (wavelet_sample_period * shape[1])

        if len(input_frequencies):
            freq_lo = input_frequencies[0]
            freq_hi = input_frequencies[-1]
        else:
            freq_lo = 0
            freq_hi = 1

        scale_range_start = max(1, int(0.25 * shape[1]))
        scale_range_end = max(1, int(0.75 * shape[1]))
        scale_min = data_matrix[:, scale_range_start:scale_range_end, :].min()
        scale_max = data_matrix[:, scale_range_start:scale_range_end, :].max()
        matrix_data = ABCDisplayer.dump_with_precision(data_matrix.flat)
        matrix_shape = json.dumps(data_matrix.squeeze().shape)

        params = dict(canvasName="Wavelet Spectrogram for: " + ts_index.title,
                      xAxisName="Time (%s)" % str(ts_index.sample_period_unit),
                      yAxisName="Frequency (%s)" % str("kHz"),
                      title=self._ui_name,
                      matrix_data=matrix_data,
                      matrix_shape=matrix_shape,
                      start_time=ts_index.start_time,
                      end_time=end_time,
                      freq_lo=freq_lo,
                      freq_hi=freq_hi,
                      vmin=scale_min,
                      vmax=scale_max)

        return self.build_display_result("wavelet/wavelet_view", params,
                                         pages={"controlPage": "wavelet/controls"})
