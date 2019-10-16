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

from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer

from tvb.core.entities.model.datatypes.spectral import ComplexCoherenceSpectrumIndex
from tvb.core.neotraits.forms import DataTypeSelectField


class ImaginaryCoherenceDisplayForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(ImaginaryCoherenceDisplayForm, self).__init__(prefix, project_id)
        self.input_data = DataTypeSelectField(self.get_required_datatype(), self, 'input_data', required=True,
                                              label='Complex Coherence Result', conditions=self.get_filters(),
                                              doc='Imaginary Coherence Analysis to display')

    @staticmethod
    def get_required_datatype():
        return ComplexCoherenceSpectrumIndex

    @staticmethod
    def get_input_name():
        return '_input_data'

    @staticmethod
    def get_filters():
        return None


class ImaginaryCoherenceDisplay(ABCDisplayer):
    """
    This viewer takes as inputs a result from complex coherence analysis, 
    and returns required parameters for a MatplotLib representation.
    """

    _ui_name = "Complex (Imaginary) Coherence Visualizer"
    _ui_subsection = "complex_coherence"

    def get_form_class(self):
        return ImaginaryCoherenceDisplayForm

    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        input_data = kwargs['input_data']
        input_data_h5_class, input_data_h5_path = self._load_h5_of_gid(input_data.gid)
        with input_data_h5_class(input_data_h5_path) as input_data_h5:
            required_memory = numpy.prod(input_data_h5.read_data_shape()) * 8
        return required_memory

    def generate_preview(self, input_data, **kwargs):
        return self.launch(input_data)

    def launch(self, input_data, **kwargs):
        """
        Draw interactive display.
        """
        self.log.debug("Plot started...")

        input_data_h5_class, input_data_h5_path = self._load_h5_of_gid(input_data.gid)
        with input_data_h5_class(input_data_h5_path) as input_data_h5:
            source_gid = input_data_h5.source

        source_index = self.load_entity_by_gid(source_gid)

        params = dict(plotName=source_index.type,
                      xAxisName="Frequency [kHz]",
                      yAxisName="CohSpec",
                      available_xScale=["Linear", "Logarithmic"],
                      available_spectrum=json.dumps(input_data_h5_class.spectrum_types),
                      spectrum_list=input_data_h5_class.spectrum_types,
                      xscale="Linear",
                      spectrum=input_data_h5_class.spectrum_types[0],
                      url_base=self.build_h5_url(input_data.gid, 'get_spectrum_data', parameter=""),
                      # TODO investigate the static xmin and xmax values
                      xmin=0.02,
                      xmax=0.8)

        return self.build_display_result("complex_coherence/view", params)
