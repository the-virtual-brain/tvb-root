# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""

from tvb.core.adapters.abcdisplayer import ABCMPLH5Displayer
from tvb.datatypes.spectral import WaveletCoefficients



class WaveletSpectrogramViewer(ABCMPLH5Displayer):
    """
    Plot the power of a WaveletCoefficients object using a MPLH5 canvas.
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
        return input_data[0] * input_data[1] * 8


    def plot(self, figure, **kwargs):
        input_data = kwargs['input_data']
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
            #TODO: This is a dummy, just showing first var, mode, and average over nodes
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

        axes = figure.gca()
        ## todo is squeeze ok here?
        img = axes.imshow(data_matrix.squeeze(), aspect="auto", origin="lower",
                          extent=(start_time, end_time, freq_lo, freq_hi),
                          vmin=scale_min, vmax=scale_max)
        figure.colorbar(img)
        axes.set_xlabel("Time (%s)" % str(input_data.source.sample_period_unit))
        axes.set_ylabel("Frequency (%s)" % str("kHz"))
        axes.set_title(input_data.source.type)


