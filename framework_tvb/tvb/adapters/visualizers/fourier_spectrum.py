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
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.entities.model.datatypes.spectral import FourierSpectrumIndex
from tvb.core.neotraits.forms import DataTypeSelectField
from tvb.core.neocom import h5
from tvb.datatypes.spectral import FourierSpectrum
from tvb.datatypes.time_series import TimeSeries


class FourierSpectrumForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(FourierSpectrumForm, self).__init__(prefix, project_id)
        self.input_data = DataTypeSelectField(self.get_required_datatype(), self, name='input_data', required=True,
                                              label='Fourier Result', doc='Fourier Analysis to display',
                                              conditions=self.get_filters())

    @staticmethod
    def get_input_name():
        return "_input_data"

    @staticmethod
    def get_required_datatype():
        return FourierSpectrumIndex

    @staticmethod
    def get_filters():
        return None

    def get_traited_datatype(self):
        return None


class FourierSpectrumDisplay(ABCDisplayer):
    """
    This viewer takes as inputs a result form FFT analysis, and returns
    required parameters for a MatplotLib representation.
    """

    _ui_name = "Fourier Visualizer"
    _ui_subsection = "fourier"

    def get_form_class(self):
        return FourierSpectrumForm

    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        return numpy.prod(kwargs['input_data'].read_data_shape()) * 8

    def generate_preview(self, **kwargs):
        return self.launch(**kwargs)

    def launch(self, input_data, **kwargs):
        self.log.debug("Plot started...")
        # these partial loads are dangerous for TS and FS instances, but efficient
        fourier_spectrum = FourierSpectrum()
        with h5.h5_file_for_index(input_data) as input_h5:
            shape = list(input_h5.array_data.shape)
            fourier_spectrum.segment_length = input_h5.segment_length.load()
            fourier_spectrum.windowing_function = input_h5.windowing_function.load()

        ts_index = self.load_entity_by_gid(input_data.source_gid)
        state_list = ts_index.get_labels_for_dimension(1)
        if len(state_list) == 0:
            state_list = list(range(shape[1]))
        fourier_spectrum.source = TimeSeries(sample_period=ts_index.sample_period)

        mode_list = list(range(shape[3]))
        available_scales = ["Linear", "Logarithmic"]

        params = dict(matrix_shape=json.dumps([shape[0], shape[2]]),
                      plotName=ts_index.title,
                      url_base=self.build_h5_url(input_data.gid, "get_fourier_data", parameter=""),
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
                      x_values=json.dumps(fourier_spectrum.frequency[slice(shape[0])].tolist()),
                      xmin=fourier_spectrum.freq_step,
                      xmax=fourier_spectrum.max_freq)
        return self.build_display_result("fourier_spectrum/view", params)
