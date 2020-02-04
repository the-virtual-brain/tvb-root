# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Adapter that uses the traits module to generate interfaces for
ContinuousWaveletTransform Analyzer.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""

import uuid
import numpy
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesH5
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.analyzers.wavelet import ContinuousWaveletTransform
from tvb.basic.neotraits.api import Range
from tvb.datatypes.time_series import TimeSeries
from tvb.core.adapters.abcadapter import ABCAsynchronous, ABCAdapterForm
from tvb.core.entities.filters.chain import FilterChain
from tvb.adapters.datatypes.h5.spectral_h5 import WaveletCoefficientsH5
from tvb.adapters.datatypes.db.spectral import WaveletCoefficientsIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.core.neotraits.forms import ScalarField, FormField, Form, SimpleFloatField, TraitDataTypeSelectField
from tvb.core.neotraits.db import from_ndarray
from tvb.core.neocom import h5


class WaveletAdapterModel(ViewModel, ContinuousWaveletTransform):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label="Time Series",
        required=True,
        doc="""The timeseries to which the wavelet is to be applied."""
    )


class RangeForm(Form):
    def __init__(self, prefix=''):
        super(RangeForm, self).__init__(prefix)
        self.lo = SimpleFloatField(self, name='lo', required=True, label='Lo', doc='start of range')
        # default=ContinuousWaveletTransform.frequencies.lo)
        self.step = SimpleFloatField(self, name='step', required=True, label='Step', doc='step of range')
        # default=ContinuousWaveletTransform.frequencies.step)
        self.hi = SimpleFloatField(self, name='hi', required=True, label='Hi', doc='end of range')
        # default=ContinuousWaveletTransform.frequencies.hi)


class ContinuousWaveletTransformAdapterForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(ContinuousWaveletTransformAdapterForm, self).__init__(prefix, project_id)
        self.time_series = TraitDataTypeSelectField(WaveletAdapterModel.time_series, self,
                                                    name=self.get_input_name(), conditions=self.get_filters(),
                                                    has_all_option=True)
        self.mother = ScalarField(ContinuousWaveletTransform.mother, self)
        self.sample_period = ScalarField(ContinuousWaveletTransform.sample_period, self)
        self.normalisation = ScalarField(ContinuousWaveletTransform.normalisation, self)
        self.q_ratio = ScalarField(ContinuousWaveletTransform.q_ratio, self)
        self.frequencies = FormField(RangeForm, self, name='frequencies',
                                     label=ContinuousWaveletTransform.frequencies.label,
                                     doc=ContinuousWaveletTransform.frequencies.doc)

    @staticmethod
    def get_view_model():
        return WaveletAdapterModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_input_name():
        return 'time_series'

    def get_traited_datatype(self):
        return ContinuousWaveletTransform()

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.data_ndim'], operations=["=="], values=[4])


class ContinuousWaveletTransformAdapter(ABCAsynchronous):
    """
    TVB adapter for calling the ContinuousWaveletTransform algorithm.
    """

    _ui_name = "Continuous Wavelet Transform"
    _ui_description = "Compute Wavelet Tranformation for a TimeSeries input DataType."
    _ui_subsection = "wavelet"

    def get_form_class(self):
        return ContinuousWaveletTransformAdapterForm

    def get_output(self):
        return [WaveletCoefficientsIndex]

    def configure(self, view_model):
        """
        Store the input shape to be later used to estimate memory usage. Also create the algorithm instance.
        """
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series.hex)

        input_shape = []
        for length in [self.input_time_series_index.data_length_1d,
                       self.input_time_series_index.data_length_2d,
                       self.input_time_series_index.data_length_3d,
                       self.input_time_series_index.data_length_4d]:
            if length is not None:
                input_shape.append(length)

        self.input_shape = tuple(input_shape)
        self.log.debug("Time series shape is %s" % str(self.input_shape))
        # -------------------- Fill Algorithm for Analysis -------------------##
        algorithm = ContinuousWaveletTransform()
        if view_model.mother is not None:
            algorithm.mother = view_model.mother

        if view_model.sample_period is not None:
            algorithm.sample_period = view_model.sample_period

        # TODO range form is not correctly populated, some work is still needed there
        # if (view_model.frequencies is not None):
        #     and 'lo' in frequencies_parameters
        #         and 'hi' in frequencies_parameters and frequencies_parameters['hi'] != frequencies_parameters['lo']):
        #     algorithm.frequencies = Range(**frequencies_parameters)

        if view_model.normalisation is not None:
            algorithm.normalisation = view_model.normalisation

        if view_model.q_ratio is not None:
            algorithm.q_ratio = view_model.q_ratio

        self.algorithm = algorithm

    def get_required_memory_size(self, view_model):
        """
        Return the required memory to run this algorithm.
        """
        used_shape = (self.input_shape[0],
                      self.input_shape[1],
                      1,
                      self.input_shape[3])
        input_size = numpy.prod(used_shape) * 8.0
        output_size = self.algorithm.result_size(used_shape, self.input_time_series_index.sample_period)
        return input_size + output_size

    def get_required_disk_size(self, view_model):
        """
        Returns the required disk size to be able to run the adapter.(in kB)
        """
        used_shape = (self.input_shape[0],
                      self.input_shape[1],
                      1,
                      self.input_shape[3])
        return self.array_size2kb(self.algorithm.result_size(used_shape, self.input_time_series_index.sample_period))

    def launch(self, view_model):
        """ 
        Launch algorithm and build results. 
        """
        # --------- Prepare a WaveletCoefficients object for result ----------##
        frequencies_array = numpy.array([])
        if self.algorithm.frequencies is not None:
            frequencies_array = self.algorithm.frequencies.to_array()

        time_series_h5 = h5.h5_file_for_index(self.input_time_series_index)
        assert isinstance(time_series_h5, TimeSeriesH5)

        wavelet_index = WaveletCoefficientsIndex()
        dest_path = h5.path_for(self.storage_path, WaveletCoefficientsH5, wavelet_index.gid)

        wavelet_h5 = WaveletCoefficientsH5(path=dest_path)
        wavelet_h5.gid.store(uuid.UUID(wavelet_index.gid))
        wavelet_h5.source.store(view_model.time)
        wavelet_h5.mother.store(self.algorithm.mother)
        wavelet_h5.q_ratio.store(self.algorithm.q_ratio)
        wavelet_h5.sample_period.store(self.algorithm.sample_period)
        wavelet_h5.frequencies.store(frequencies_array)
        wavelet_h5.normalisation.store(self.algorithm.normalisation)

        # ------------- NOTE: Assumes 4D, Simulator timeSeries. --------------##
        node_slice = [slice(self.input_shape[0]), slice(self.input_shape[1]), None, slice(self.input_shape[3])]

        # ---------- Iterate over slices and compose final result ------------##
        small_ts = TimeSeries()
        small_ts.sample_period = time_series_h5.sample_period.load()
        for node in range(self.input_shape[2]):
            node_slice[2] = slice(node, node + 1)
            small_ts.data = time_series_h5.read_data_slice(tuple(node_slice))
            self.algorithm.time_series = small_ts
            partial_wavelet = self.algorithm.evaluate()
            wavelet_h5.write_data_slice(partial_wavelet)

        wavelet_h5.close()
        time_series_h5.close()

        wavelet_index.source_gid = self.input_time_series_index.gid
        wavelet_index.mother = self.algorithm.mother
        wavelet_index.normalisation = self.algorithm.normalisation
        wavelet_index.q_ratio = self.algorithm.q_ratio
        wavelet_index.sample_period = self.algorithm.sample_period
        wavelet_index.number_of_scales = frequencies_array.shape[0]
        wavelet_index.frequencies_min, wavelet_index.frequencies_max, _ = from_ndarray(frequencies_array)

        return wavelet_index
