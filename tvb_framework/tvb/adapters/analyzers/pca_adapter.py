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
from tvb.adapters.datatypes.db.mode_decompositions import PrincipalComponentsIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.h5.mode_decompositions_h5 import PrincipalComponentsH5
from tvb.analyzers.pca import compute_pca
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.time_series import TimeSeries


class PCAAdapterModel(ViewModel):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label="Time Series",
        required=True,
        doc="""The timeseries to which the PCA is to be applied. NOTE: The 
                TimeSeries must be longer(more time-points) than the number of nodes
                -- Mostly a problem for surface times-series, which, if sampled at
                1024Hz, would need to be greater than 16 seconds long."""
    )


class PCAAdapterForm(ABCAdapterForm):

    def __init__(self):
        super(PCAAdapterForm, self).__init__()
        self.time_series = TraitDataTypeSelectField(PCAAdapterModel.time_series, name=self.get_input_name(),
                                                    conditions=self.get_filters(), has_all_option=True)
    @staticmethod
    def get_view_model():
        return PCAAdapterModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.data_ndim'], operations=["=="], values=[4])

    @staticmethod
    def get_input_name():
        return "time_series"


class PCAAdapter(ABCAdapter):
    """ TVB adapter for calling the PCA algorithm. """

    _ui_name = "Principal Component Analysis"
    _ui_description = "PCA for a TimeSeries input DataType."
    _ui_subsection = "components"

    def get_form_class(self):
        return PCAAdapterForm

    def get_output(self):
        return [PrincipalComponentsIndex]

    def configure(self, view_model):
        # type: (PCAAdapterModel) -> None
        """
        Store the input shape to be later used to estimate memory usage
        """
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series)
        self.input_shape = (self.input_time_series_index.data_length_1d,
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)
        self.log.debug("Time series shape is %s" % str(self.input_shape))

    def get_required_memory_size(self, view_model):
        # type: (PCAAdapterModel) -> int
        """
        Return the required memory to run this algorithm.
        """
        used_shape = (self.input_shape[0], 1, self.input_shape[2], self.input_shape[3])
        input_size = numpy.prod(used_shape) * 8.0
        output_size = self.result_size(used_shape)
        return input_size + output_size

    def get_required_disk_size(self, view_model):
        # type: (PCAAdapterModel) -> int
        """
        Returns the required disk size to be able to run the adapter (in kB).
        """
        used_shape = (self.input_shape[0], 1, self.input_shape[2], self.input_shape[3])
        return self.array_size2kb(self.result_size(used_shape))

    def launch(self, view_model):
        # type: (PCAAdapterModel) -> [PrincipalComponentsIndex]
        """ 
        Launch algorithm and build results.
        :param view_model: the ViewModel keeping the algorithm inputs
        :return: the `PrincipalComponentsIndex` object built with the given timeseries as source
        """
        # --------------------- Prepare result entities ----------------------##
        principal_components_index = PrincipalComponentsIndex()
        dest_path = self.path_for(PrincipalComponentsH5, principal_components_index.gid)
        pca_h5 = PrincipalComponentsH5(path=dest_path)

        # ------------- NOTE: Assumes 4D, Simulator timeSeries. --------------##
        time_series_h5 = h5.h5_file_for_index(self.input_time_series_index)
        input_shape = time_series_h5.data.shape
        node_slice = [slice(input_shape[0]), None, slice(input_shape[2]), slice(input_shape[3])]

        # ---------- Iterate over slices and compose final result ------------##
        small_ts = TimeSeries()
        for var in range(input_shape[1]):
            node_slice[1] = slice(var, var + 1)
            small_ts.data = time_series_h5.read_data_slice(tuple(node_slice))
            self.time_series = small_ts.gid
            partial_pca = compute_pca(small_ts)
            pca_h5.write_data_slice(partial_pca)

        time_series_h5.close()

        partial_pca.source.gid = view_model.time_series
        partial_pca.gid = uuid.UUID(principal_components_index.gid)
        principal_components_index.fill_from_has_traits(partial_pca)

        pca_h5.store(partial_pca, scalars_only=True)
        pca_h5.close()

        return principal_components_index

    def result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the results of the PCA analysis.
        """
        result_size = numpy.sum(list(map(numpy.prod,
                                         self.result_shape(input_shape)))) * 8.0  # Bytes
        return result_size

    @staticmethod
    def result_shape(input_shape):
        """
        Returns the shape of the main result of the PCA analysis -- compnnent
        weights matrix and a vector of fractions.
        """
        weights_shape = (input_shape[2], input_shape[2], input_shape[1],
                         input_shape[3])
        fractions_shape = (input_shape[2], input_shape[1], input_shape[3])

        return [weights_shape, fractions_shape]
