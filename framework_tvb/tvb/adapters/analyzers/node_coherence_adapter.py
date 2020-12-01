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
Adapter that uses the traits module to generate interfaces for FFT Analyzer.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""

import json
import uuid

import numpy
from tvb.adapters.datatypes.db.spectral import CoherenceSpectrumIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.h5.spectral_h5 import CoherenceSpectrumH5
from tvb.analyzers.node_coherence import NodeCoherence
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField, IntField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.spectral import CoherenceSpectrum
from tvb.datatypes.time_series import TimeSeries


class NodeCoherenceModel(ViewModel, NodeCoherence):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label="Time Series",
        required=True,
        doc="""The timeseries to which the FFT is to be applied."""
    )


class NodeCoherenceForm(ABCAdapterForm):

    def __init__(self, project_id=None):
        super(NodeCoherenceForm, self).__init__(project_id)
        self.time_series = TraitDataTypeSelectField(NodeCoherenceModel.time_series, self.project_id,
                                                    name=self.get_input_name(), conditions=self.get_filters(),
                                                    has_all_option=True)
        self.nfft = IntField(NodeCoherenceModel.nfft, self.project_id)

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

    def get_traited_datatype(self):
        return NodeCoherence()


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
        Also create the algorithm instance.
        """
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series)
        self.input_shape = (self.input_time_series_index.data_length_1d,
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)
        self.log.debug("Time series shape is %s" % str(self.input_shape))
        # -------------------- Fill Algorithm for Analysis -------------------##
        self.algorithm = NodeCoherence()
        if view_model.nfft is not None:
            self.algorithm.nfft = view_model.nfft

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
        output_size = self.algorithm.result_size(used_shape)
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
        return self.array_size2kb(self.algorithm.result_size(used_shape))

    def launch(self, view_model):
        # type: (NodeCoherenceModel) -> [CoherenceSpectrumIndex]
        """
        Launch algorithm and build results. 
        """
        # --------- Prepare a CoherenceSpectrum object for result ------------##
        coherence_spectrum_index = CoherenceSpectrumIndex()
        time_series_h5 = h5.h5_file_for_index(self.input_time_series_index)

        dest_path = h5.path_for(self.storage_path, CoherenceSpectrumH5, coherence_spectrum_index.gid)
        coherence_h5 = CoherenceSpectrumH5(dest_path)
        coherence_h5.gid.store(uuid.UUID(coherence_spectrum_index.gid))
        coherence_h5.source.store(view_model.time_series)
        coherence_h5.nfft.store(self.algorithm.nfft)

        # ------------- NOTE: Assumes 4D, Simulator timeSeries. --------------##
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
            self.algorithm.time_series = small_ts
            partial_coh = self.algorithm.evaluate()
            coherence_h5.write_data_slice(partial_coh)
        coherence_h5.frequency.store(partial_coh.frequency)
        array_metadata = coherence_h5.array_data.get_cached_metadata()
        freq_metadata = coherence_h5.frequency.get_cached_metadata()
        coherence_h5.close()
        time_series_h5.close()

        coherence_spectrum_index.array_data_min = array_metadata.min
        coherence_spectrum_index.array_data_max = array_metadata.max
        coherence_spectrum_index.array_data_mean = array_metadata.mean
        coherence_spectrum_index.array_has_complex = array_metadata.has_complex
        coherence_spectrum_index.array_is_finite = array_metadata.is_finite
        coherence_spectrum_index.shape = json.dumps(coherence_h5.array_data.shape)
        coherence_spectrum_index.ndim = len(coherence_h5.array_data.shape)
        coherence_spectrum_index.fk_source_gid = self.input_time_series_index.gid
        coherence_spectrum_index.nfft = partial_coh.nfft
        coherence_spectrum_index.frequencies_min = freq_metadata.min
        coherence_spectrum_index.frequencies_max = freq_metadata.max
        coherence_spectrum_index.subtype = CoherenceSpectrum.__name__

        return coherence_spectrum_index
