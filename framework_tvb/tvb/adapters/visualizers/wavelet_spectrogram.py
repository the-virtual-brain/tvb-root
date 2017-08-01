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
Plot the power of a WaveletCoefficients object

.. moduleauthor:: Dan Pop <dan.pop@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
"""

import json
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.datatypes.spectral import WaveletCoefficients


class WaveletSpectrogramVisualizer(ABCDisplayer):
    """
    Plot the power of a WaveletCoefficients object using SVG an D3.
    """
    _ui_name = "Spectrogram of Wavelet Power"
    _ui_subsection = "wavelet"

    def get_input_tree(self):
        """
        Accept as input result from Continuous wavelet transform analysis.
        """

        return [{'name': 'input_data', 'label': 'Wavelet transform Result',
                 'type': WaveletCoefficients, 'required': True,
                 'description': 'Wavelet spectrogram to display'}]

    def get_required_memory_size(self, **kwargs):
        """
         Return the required memory to run this algorithm.
         """
        input_data = kwargs['input_data']
        shape = input_data.read_data_shape()
        return shape[0] * shape[1] * 8


    def generate_preview(self, input_data, **kwargs):
        return self.launch(input_data)


    def launch(self, input_data, **kwarg):
        shape = input_data.read_data_shape()
        start_time = input_data.source.start_time
        wavelet_sample_period = input_data.source.sample_period * \
                                max((1, int(input_data.sample_period / input_data.source.sample_period)))
        end_time = input_data.source.start_time + (wavelet_sample_period * shape[1])
        if len(input_data.frequencies):
            freq_lo = input_data.frequencies[0]
            freq_hi = input_data.frequencies[-1]
        else:
            freq_lo = 0
            freq_hi = 1
        slices = (slice(shape[0]),
                  slice(shape[1]),
                  slice(0, 1, None),
                  slice(0, shape[3], None),
                  slice(0, 1, None))

        data_matrix = input_data.get_data('power', slices)
        data_matrix = data_matrix.sum(axis=3)

        scale_range_start = max(1, int(0.25 * shape[1]))
        scale_range_end = max(1, int(0.75 * shape[1]))
        scale_min = data_matrix[:, scale_range_start:scale_range_end, :].min()
        scale_max = data_matrix[:, scale_range_start:scale_range_end, :].max()
        matrix_data = ABCDisplayer.dump_with_precision(data_matrix.flat)
        matrix_shape = json.dumps(data_matrix.squeeze().shape)
        params = dict(canvasName="Wavelet Spectrogram for: " + input_data.source.type,
                      xAxisName="Time (%s)" % str(input_data.source.sample_period_unit),
                      yAxisName="Frequency (%s)" % str("kHz"),
                      title=self._ui_name,
                      matrix_data=matrix_data,
                      matrix_shape=matrix_shape,
                      start_time=start_time,
                      end_time=end_time,
                      freq_lo=freq_lo,
                      freq_hi=freq_hi,
                      vmin=scale_min,
                      vmax=scale_max)

        return self.build_display_result("wavelet/wavelet_view", params,
                                         pages={"controlPage": "wavelet/controls"})
