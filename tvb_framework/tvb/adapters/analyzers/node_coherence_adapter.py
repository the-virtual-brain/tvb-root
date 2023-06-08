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
Adapter that uses the traits module to generate interfaces for FFT Analyzer.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""

import uuid

import numpy
from tvb.adapters.datatypes.db.spectral import CoherenceSpectrumIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.h5.spectral_h5 import CoherenceSpectrumH5
from tvb.analyzers.node_coherence import calculate_cross_coherence
from tvb.basic.neotraits.api import Int
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField, IntField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.time_series import TimeSeries


class NodeCoherenceModel(ViewModel):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label="Time Series",
        required=True,
        doc="""The timeseries to which the Cross Coherence is to be applied."""
    )

    nfft = Int(
        label="Data-points per block",
        default=256,
        doc="""Should be a power of 2...""")


class NodeCoherenceForm(ABCAdapterForm):

    def __init__(self):
        super(NodeCoherenceForm, self).__init__()
        self.time_series = TraitDataTypeSelectField(NodeCoherenceModel.time_series, name=self.get_input_name(),
                                                    conditions=self.get_filters(), has_all_option=True)
        self.nfft = IntField(NodeCoherenceModel.nfft)

    @staticmethod
    def get_view_model():
        return NodeCoherenceModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.data_ndim'], operations=["=="], values=[4])

    @staticmethod
    def get_input_name():
        return "time_series"


class NodeCoherenceAdapter(ABCAdapter):
    """ TVB adapter for calling the NodeCoherence algorithm. """

    _ui_name = "Cross coherence of nodes"
    _ui_description = "Compute Node Coherence for a TimeSeries input DataType."
    _ui_subsection = "coherence"

    def get_form_class(self):
        return NodeCoherenceForm

    def get_output(self):
        return [CoherenceSpectrumIndex]

    def configure(self, view_model):
        # type: (NodeCoherenceModel) -> None
        """
        Store the input shape to be later used to estimate memory usage.
        """
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series)
        self.input_shape = (self.input_time_series_index.data_length_1d,
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)
        self.log.debug("Time series shape is %s" % str(self.input_shape))

    def get_required_memory_size(self, view_model):
        # type: (NodeCoherenceModel) -> int
        """
        Return the required memory to run this algorithm.
        """
        used_shape = (self.input_shape[0],
                      1,
                      self.input_shape[2],
                      self.input_shape[3])
        input_size = numpy.prod(used_shape) * 8.0
        output_size = self.result_size(used_shape, view_model.nfft)
        return input_size + output_size

    def get_required_disk_size(self, view_model):
        # type: (NodeCoherenceModel) -> int
        """
        Returns the required disk size to be able to run the adapter (in kB).
        """
        used_shape = (self.input_shape[0],
                      1,
                      self.input_shape[2],
                      self.input_shape[3])
        return self.array_size2kb(self.result_size(used_shape, view_model.nfft))

    def launch(self, view_model):
        # type: (NodeCoherenceModel) -> [CoherenceSpectrumIndex]
        """
        Launch algorithm and build results.
        :param view_model: the ViewModel keeping the algorithm inputs
        :return: the node coherence for the specified time series
        """
        # -------------------- Prepare result entities -----------------------##
        coherence_spectrum_index = CoherenceSpectrumIndex()
        dest_path = self.path_for(CoherenceSpectrumH5, coherence_spectrum_index.gid)
        coherence_h5 = CoherenceSpectrumH5(dest_path)

        # ------------- NOTE: Assumes 4D, Simulator timeSeries. --------------##
        time_series_h5 = h5.h5_file_for_index(self.input_time_series_index)
        input_shape = time_series_h5.data.shape
        node_slice = [slice(input_shape[0]), None, slice(input_shape[2]), slice(input_shape[3])]

        # ---------- Iterate over slices and compose final result ------------##
        small_ts = TimeSeries()
        small_ts.sample_period = time_series_h5.sample_period.load()
        small_ts.sample_period_unit = time_series_h5.sample_period_unit.load()
        partial_coh = None
        for var in range(input_shape[1]):
            node_slice[1] = slice(var, var + 1)
            small_ts.data = time_series_h5.read_data_slice(tuple(node_slice))
            partial_coh = calculate_cross_coherence(small_ts, view_model.nfft)
            coherence_h5.write_data_slice(partial_coh)

        time_series_h5.close()

        if partial_coh is None:
            self.log.warn(f"Empty TS {input_shape[1]} ?")
            return None

        partial_coh.source.gid = view_model.time_series
        partial_coh.gid = uuid.UUID(coherence_spectrum_index.gid)

        coherence_spectrum_index.fill_from_has_traits(partial_coh)
        self.fill_index_from_h5(coherence_spectrum_index, coherence_h5)

        coherence_h5.store(partial_coh, scalars_only=True)
        coherence_h5.frequency.store(partial_coh.frequency)
        coherence_h5.close()

        return coherence_spectrum_index

    def result_size(self, input_shape, nfft):
        """
        Returns the storage size in Bytes of the main result of NodeCoherence.
        """
        # TODO This depends on input array dtype!
        result_size = numpy.sum(list(map(numpy.prod, self.result_shape(input_shape, nfft)))) * 8.0  # Bytes

        return result_size

    @staticmethod
    def result_shape(input_shape, nfft):
        """Returns the shape of the main result of NodeCoherence."""
        freq_len = nfft / 2 + 1
        freq_shape = (freq_len,)
        result_shape = (freq_len, input_shape[2], input_shape[2], input_shape[1], input_shape[3])
        return [result_shape, freq_shape]
