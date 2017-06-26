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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
# Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
# Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
# The Virtual Brain: a simulator of primate brain network dynamics.
# Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.datatypes.spectral import WaveletCoefficients


class WaveletSpectrogramVisualizer(ABCDisplayer):
    """
        Plot the power of a WaveletCoefficients object using a MPLH5 canvas.
        """
    _ui_name = "New Spectrogram of Wavelet Power"
    _ui_subsection = "wavelet"

    def get_input_tree(self):
        """
        Accept as input result from Continuous wavelet transform analysis.
        """

        return [{'name': 'input_data', 'label': 'Wavelet transform Result',
                 'type': WaveletCoefficients, 'required': True,
                 'description': 'Wavelet spectrogram to display'}]


    def get_required_memory_size(self, **kwargs):
        return -1

    def launch(self, input_data, **kwarg):

        params = dict(title="Wavelet Spectrogram Visualizer")
        return self.build_display_result("wavelet/wavelet_view", params)