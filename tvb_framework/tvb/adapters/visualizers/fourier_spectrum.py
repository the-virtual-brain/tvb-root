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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>

"""
import json
import numpy

from tvb.adapters.datatypes.db.spectral import FourierSpectrumIndex
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer, URLGenerator
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neocom import h5
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.spectral import FourierSpectrum
from tvb.datatypes.time_series import TimeSeries


class FourierSpectrumModel(ViewModel):
    input_data = DataTypeGidAttr(
        linked_datatype=FourierSpectrum,
        label='Fourier Result',
        doc='Fourier Analysis to display'
    )


class FourierSpectrumForm(ABCAdapterForm):

    def __init__(self):
        super(FourierSpectrumForm, self).__init__()
        self.input_data = TraitDataTypeSelectField(FourierSpectrumModel.input_data, name='input_data',
                                                   conditions=self.get_filters())

    @staticmethod
    def get_view_model():
        return FourierSpectrumModel

    @staticmethod
    def get_input_name():
        return "input_data"

    @staticmethod
    def get_required_datatype():
        return FourierSpectrumIndex

    @staticmethod
    def get_filters():
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

    def get_required_memory_size(self, view_model):
        # type: (FourierSpectrumModel) -> dict
        """
        Return the required memory to run this algorithm.
        """
        fs_input_index = self.load_entity_by_gid(view_model.input_data)
        return numpy.prod(fs_input_index.get_data_shape()) * 8

    def launch(self, view_model):
        # type: (FourierSpectrumModel) -> dict

        self.log.debug("Plot started...")
        # these partial loads are dangerous for TS and FS instances, but efficient
        fourier_spectrum = FourierSpectrum()
        with h5.h5_file_for_gid(view_model.input_data) as input_h5:
            shape = list(input_h5.array_data.shape)
            fourier_spectrum.segment_length = input_h5.segment_length.load()
            fourier_spectrum.windowing_function = input_h5.windowing_function.load()
            ts_index = self.load_entity_by_gid(input_h5.source.load())

        state_list = ts_index.get_labels_for_dimension(1)
        if len(state_list) == 0:
            state_list = list(range(shape[1]))
        fourier_spectrum.source = TimeSeries(sample_period=ts_index.sample_period)

        mode_list = list(range(shape[3]))
        available_scales = ["Linear", "Logarithmic"]

        params = dict(matrix_shape=json.dumps([shape[0], shape[2]]),
                      plotName=ts_index.title,
                      url_base=URLGenerator.build_h5_url(view_model.input_data, "get_fourier_data", parameter=""),
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
