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
.. moduleauthor:: Dan Pop <dan.pop@codemart.ro>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
import json

import numpy
from tvb.adapters.datatypes.db.spectral import ComplexCoherenceSpectrumIndex
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer, URLGenerator
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.spectral import ComplexCoherenceSpectrum


class ImaginaryCoherenceDisplayModel(ViewModel):
    input_data = DataTypeGidAttr(
        linked_datatype=ComplexCoherenceSpectrum,
        label='Complex Coherence Result',
        doc='Imaginary Coherence Analysis to display'
    )


class ImaginaryCoherenceDisplayForm(ABCAdapterForm):

    def __init__(self):
        super(ImaginaryCoherenceDisplayForm, self).__init__()
        self.input_data = TraitDataTypeSelectField(ImaginaryCoherenceDisplayModel.input_data, 'input_data',
                                                   conditions=self.get_filters())

    @staticmethod
    def get_view_model():
        return ImaginaryCoherenceDisplayModel

    @staticmethod
    def get_required_datatype():
        return ComplexCoherenceSpectrumIndex

    @staticmethod
    def get_input_name():
        return 'input_data'

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

    def get_required_memory_size(self, view_model):
        # type: (ImaginaryCoherenceDisplayModel) -> numpy.ndarray
        """
        Return the required memory to run this algorithm.
        """
        with h5.h5_file_for_gid(view_model.input_data) as input_data_h5:
            required_memory = numpy.prod(input_data_h5.read_data_shape()) * 8

        return required_memory

    def launch(self, view_model):
        # type: (ImaginaryCoherenceDisplayModel) -> dict
        """
        Draw interactive display.
        """
        self.log.debug("Plot started...")

        with h5.h5_file_for_gid(view_model.input_data) as input_data_h5:
            source_gid = input_data_h5.source.load()
            source_index = self.load_entity_by_gid(source_gid)

            params = dict(plotName=source_index.type,
                          xAxisName="Frequency [kHz]",
                          yAxisName="CohSpec",
                          available_xScale=["Linear", "Logarithmic"],
                          available_spectrum=json.dumps(input_data_h5.spectrum_types),
                          spectrum_list=input_data_h5.spectrum_types,
                          xscale="Linear",
                          spectrum=input_data_h5.spectrum_types[0],
                          url_base=URLGenerator.build_h5_url(view_model.input_data, 'get_spectrum_data', parameter=""),
                          # TODO investigate the static xmin and xmax values
                          xmin=0.02,
                          xmax=0.8)

        return self.build_display_result("complex_coherence/view", params)
