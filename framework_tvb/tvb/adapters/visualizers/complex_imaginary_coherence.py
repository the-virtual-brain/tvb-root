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
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
import json
import numpy
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.datatypes.spectral import ComplexCoherenceSpectrum


class ImaginaryCoherenceDisplay(ABCDisplayer):
    """
    This viewer takes as inputs a result from complex coherence analysis, 
    and returns required parameters for a MatplotLib representation.
    """

    _ui_name = "Complex (Imaginary) Coherence Visualizer"
    _ui_subsection = "complex_coherence"

    def get_input_tree(self):
        """ 
        Accept as input result from ComplexCoherence Analysis.
        """
        return [{'name': 'input_data',
                 'label': 'Complex Coherence Result',
                 'type': ComplexCoherenceSpectrum,
                 'required': True,
                 'description': 'Imaginary Coherence Analysis to display'}]

    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        return numpy.prod(kwargs['input_data'].read_data_shape()) * 8

    def generate_preview(self, input_data, **kwargs):
        return self.launch(input_data)

    def launch(self, input_data, **kwargs):
        """
        Draw interactive display.
        """
        self.log.debug("Plot started...")

        params = dict(plotName=input_data.source.type,
                      xAxisName="Frequency [kHz]",
                      yAxisName="CohSpec",
                      available_xScale=["Linear", "Logarithmic"],
                      available_spectrum=json.dumps(input_data.spectrum_types),
                      spectrum_list=input_data.spectrum_types,
                      xscale="Linear",
                      spectrum=input_data.spectrum_types[0],
                      url_base=ABCDisplayer.paths2url(input_data, "get_spectrum_data", parameter=""),
                      # TODO investigate the static xmin and xmax values
                      xmin=0.02,
                      xmax=0.8)
        return self.build_display_result("complex_coherence/view", params)
