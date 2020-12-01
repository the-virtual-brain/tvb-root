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

import numpy
from tvb.adapters.datatypes.db.spectral import ComplexCoherenceSpectrumIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.analyzers.node_complex_coherence import NodeComplexCoherence
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.time_series import TimeSeries


class NodeComplexCoherenceModel(ViewModel, NodeComplexCoherence):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label="Time Series",
        required=True,
        doc="""The timeseries for which the CrossCoherence and ComplexCoherence is to be computed."""
    )


class NodeComplexCoherenceForm(ABCAdapterForm):

    def __init__(self, project_id=None):
        super(NodeComplexCoherenceForm, self).__init__(project_id)
        self.time_series = TraitDataTypeSelectField(NodeComplexCoherenceModel.time_series, self.project_id,
                                                    name=self.get_input_name(), conditions=self.get_filters())

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


class NodeComplexCoherenceAdapter(ABCAdapter):
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
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series)
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
        """
        # TODO ---------- Iterate over slices and compose final result ------------##
        self.algorithm.time_series = h5.load_from_index(self.input_time_series_index)
        ht_result = self.algorithm.evaluate()
        self.log.debug("got ComplexCoherenceSpectrum result")
        self.log.debug("ComplexCoherenceSpectrum segment_length is %s" % (str(ht_result.segment_length)))
        self.log.debug("ComplexCoherenceSpectrum epoch_length is %s" % (str(ht_result.epoch_length)))
        self.log.debug("ComplexCoherenceSpectrum windowing_function is %s" % (str(ht_result.windowing_function)))
        # LOG.debug("ComplexCoherenceSpectrum frequency vector is %s" % (str(ht_result.frequency)))

        return h5.store_complete(ht_result, self.storage_path)
