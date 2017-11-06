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
.. moduleauthor:: Dan Pop <dan.pop@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>

"""
import json
import numpy
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.datatypes.spectral import FourierSpectrum


class FourierSpectrumDisplay(ABCDisplayer):
    """
    This viewer takes as inputs a result form FFT analysis, and returns
    required parameters for a MatplotLib representation.
    """

    _ui_name = "Fourier Visualizer"
    _ui_subsection = "fourier"

    def get_input_tree(self):
        """ 
        Accept as input result from FFT Analysis.
        """
        return [{'name': 'input_data', 'label': 'Fourier Result',
                 'type': FourierSpectrum, 'required': True,
                 'description': 'Fourier Analysis to display'}]

    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        return numpy.prod(kwargs['input_data'].read_data_shape()) * 8

    def generate_preview(self, **kwargs):
        return self.launch(**kwargs)

    def launch(self, **kwargs):
        self.log.debug("Plot started...")
        input_data = kwargs['input_data']
        shape = list(input_data.read_data_shape())
        state_list = input_data.source.labels_dimensions.get(input_data.source.labels_ordering[1], [])
        mode_list = range(shape[3])
        available_scales = ["Linear", "Logarithmic"]

        params = dict(matrix_shape=json.dumps([shape[0], shape[2]]),
                      plotName=input_data.source.type,
                      url_base=ABCDisplayer.paths2url(input_data, "get_fourier_data", parameter=""),
                      xAxisName="Frequency [kHz]",
                      yAxisName="Power",
                      available_scales=available_scales,
                      state_list=state_list,
                      mode_list=mode_list,
                      normalize_list=["no", "yes"],
                      normalize="no",
                      state_variable=state_list[0],
                      mode=mode_list[0],
                      xscale=available_scales[0],
                      yscale=available_scales[0],
                      x_values=json.dumps(input_data.frequency[slice(shape[0])].tolist()),
                      xmin=input_data.freq_step,
                      xmax=input_data.max_freq)
        return self.build_display_result("fourier_spectrum/view", params)
