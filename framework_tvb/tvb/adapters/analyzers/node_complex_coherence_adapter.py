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

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Paula Sanz Leon <paula@tvb.invalid>

"""
import uuid
import numpy
from tvb.analyzers.node_complex_coherence import NodeComplexCoherence
from tvb.core.adapters.abcadapter import ABCAsynchronous, ABCAdapterForm
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.time_series import TimeSeries
from tvb.core.entities.filters.chain import FilterChain
from tvb.adapters.datatypes.h5.spectral_h5 import ComplexCoherenceSpectrumH5
from tvb.adapters.datatypes.db.spectral import ComplexCoherenceSpectrumIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neocom import h5


class NodeComplexCoherenceModel(ViewModel, NodeComplexCoherence):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label="Time Series",
        required=True,
        doc="""The timeseries for which the CrossCoherence and ComplexCoherence is to be computed."""
    )


class NodeComplexCoherenceForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(NodeComplexCoherenceForm, self).__init__(prefix, project_id)
        self.time_series = TraitDataTypeSelectField(NodeComplexCoherenceModel.time_series, self,
                                                    name=self.get_input_name(),
                                                    conditions=self.get_filters())

    @staticmethod
    def get_view_model():
        return NodeComplexCoherenceModel

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
        return NodeComplexCoherence()


class NodeComplexCoherenceAdapter(ABCAsynchronous):
    """ TVB adapter for calling the NodeComplexCoherence algorithm. """

    _ui_name = "Complex Coherence of Nodes"
    _ui_description = "Compute the node complex (imaginary) coherence for a TimeSeries input DataType."
    _ui_subsection = "complexcoherence"

    def get_form_class(self):
        return NodeComplexCoherenceForm

    def get_output(self):
        return [ComplexCoherenceSpectrumIndex]

    def get_required_memory_size(self, view_model):
        # type: (NodeComplexCoherenceModel) -> int
        """
        Return the required memory to run this algorithm.
        """
        input_size = numpy.prod(self.input_shape) * 8.0
        output_size = self.algorithm.result_size(self.input_shape, self.algorithm.max_freq,
                                                 self.algorithm.epoch_length,
                                                 self.algorithm.segment_length,
                                                 self.algorithm.segment_shift,
                                                 self.input_time_series_index.sample_period,
                                                 self.algorithm.zeropad,
                                                 self.algorithm.average_segments)

        return input_size + output_size

    def get_required_disk_size(self, view_model):
        # type: (NodeComplexCoherenceModel) -> int
        """
        Returns the required disk size to be able to run the adapter (in kB).
        """
        result = self.algorithm.result_size(self.input_shape, self.algorithm.max_freq,
                                            self.algorithm.epoch_length,
                                            self.algorithm.segment_length,
                                            self.algorithm.segment_shift,
                                            self.input_time_series_index.sample_period,
                                            self.algorithm.zeropad,
                                            self.algorithm.average_segments)
        return self.array_size2kb(result)

    def configure(self, view_model):
        # type: (NodeComplexCoherenceModel) -> None
        """
        Do any configuration needed before launching and create an instance of the algorithm.
        """
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series.hex)
        self.input_shape = (self.input_time_series_index.data_length_1d,
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)
        self.log.debug("Time series shape is %s" % (str(self.input_shape)))
        # -------------------- Fill Algorithm for Analysis -------------------##
        self.algorithm = NodeComplexCoherence()
        self.memory_factor = 1

    def launch(self, view_model):
        # type: (NodeComplexCoherenceModel) -> [ComplexCoherenceSpectrumIndex]
        """
        Launch algorithm and build results.

        :returns: the `ComplexCoherenceSpectrum` built with the given time-series
        """
        # ------- Prepare a ComplexCoherenceSpectrum object for result -------##
        complex_coherence_spectrum_index = ComplexCoherenceSpectrumIndex()
        time_series_h5 = h5.h5_file_for_index(self.input_time_series_index)

        dest_path = h5.path_for(self.storage_path, ComplexCoherenceSpectrumH5, complex_coherence_spectrum_index.gid)
        spectra_h5 = ComplexCoherenceSpectrumH5(dest_path)
        spectra_h5.gid.store(uuid.UUID(complex_coherence_spectrum_index.gid))
        spectra_h5.source.store(time_series_h5.gid.load())

        # ------------------- NOTE: Assumes 4D TimeSeries. -------------------##
        input_shape = time_series_h5.data.shape
        node_slice = [slice(input_shape[0]), slice(input_shape[1]), slice(input_shape[2]), slice(input_shape[3])]

        # ---------- Iterate over slices and compose final result ------------##
        small_ts = TimeSeries()
        small_ts.sample_period = time_series_h5.sample_period.load()
        small_ts.data = time_series_h5.read_data_slice(tuple(node_slice))
        self.algorithm.time_series = small_ts

        partial_result = self.algorithm.evaluate()
        self.log.debug("got partial_result")
        self.log.debug("partial segment_length is %s" % (str(partial_result.segment_length)))
        self.log.debug("partial epoch_length is %s" % (str(partial_result.epoch_length)))
        self.log.debug("partial windowing_function is %s" % (str(partial_result.windowing_function)))
        # LOG.debug("partial frequency vector is %s" % (str(partial_result.frequency)))

        spectra_h5.write_data_slice(partial_result)
        spectra_h5.segment_length.store(partial_result.segment_length)
        spectra_h5.epoch_length.store(partial_result.epoch_length)
        spectra_h5.windowing_function.store(partial_result.windowing_function)
        # spectra.frequency = partial_result.frequency
        spectra_h5.close()
        time_series_h5.close()

        complex_coherence_spectrum_index.source_gid = self.input_time_series_index.gid
        complex_coherence_spectrum_index.epoch_length = partial_result.epoch_length
        complex_coherence_spectrum_index.segment_length = partial_result.segment_length
        complex_coherence_spectrum_index.windowing_function = partial_result.windowing_function
        complex_coherence_spectrum_index.frequency_step = partial_result.freq_step
        complex_coherence_spectrum_index.max_frequency = partial_result.max_freq

        return complex_coherence_spectrum_index
