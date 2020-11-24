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
import uuid

import numpy
from tvb.adapters.datatypes.db.mode_decompositions import PrincipalComponentsIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.h5.mode_decompositions_h5 import PrincipalComponentsH5
from tvb.analyzers.pca import PCA
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.time_series import TimeSeries


class PCAAdapterModel(ViewModel, PCA):
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

    def __init__(self, project_id=None):
        super(PCAAdapterForm, self).__init__(project_id)
        self.time_series = TraitDataTypeSelectField(PCAAdapterModel.time_series, self.project_id,
                                                    name=self.get_input_name(), conditions=self.get_filters(),
                                                    has_all_option=True)

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

    def get_traited_datatype(self):
        return PCA()


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
        Store the input shape to be later used to estimate memory usage. Also
        create the algorithm instance.
        """
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series)
        self.input_shape = (self.input_time_series_index.data_length_1d,
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)
        self.log.debug("Time series shape is %s" % str(self.input_shape))
        # -------------------- Fill Algorithm for Analysis -------------------##
        self.algorithm = PCA()

    def get_required_memory_size(self, view_model):
        # type: (PCAAdapterModel) -> int
        """
        Return the required memory to run this algorithm.
        """
        used_shape = (self.input_shape[0], 1, self.input_shape[2], self.input_shape[3])
        input_size = numpy.prod(used_shape) * 8.0
        output_size = self.algorithm.result_size(used_shape)
        return input_size + output_size

    def get_required_disk_size(self, view_model):
        # type: (PCAAdapterModel) -> int
        """
        Returns the required disk size to be able to run the adapter (in kB).
        """
        used_shape = (self.input_shape[0], 1, self.input_shape[2], self.input_shape[3])
        return self.array_size2kb(self.algorithm.result_size(used_shape))

    def launch(self, view_model):
        # type: (PCAAdapterModel) -> [PrincipalComponentsIndex]
        """ 
        Launch algorithm and build results.

        :returns: the `PrincipalComponents` object built with the given timeseries as source
        """
        # --------- Prepare a PrincipalComponents object for result ----------##
        principal_components_index = PrincipalComponentsIndex()
        principal_components_index.fk_source_gid = view_model.time_series.hex

        time_series_h5 = h5.h5_file_for_index(self.input_time_series_index)

        dest_path = h5.path_for(self.storage_path, PrincipalComponentsH5, principal_components_index.gid)
        pca_h5 = PrincipalComponentsH5(path=dest_path)
        pca_h5.source.store(time_series_h5.gid.load())
        pca_h5.gid.store(uuid.UUID(principal_components_index.gid))

        # ------------- NOTE: Assumes 4D, Simulator timeSeries. --------------##
        input_shape = time_series_h5.data.shape
        node_slice = [slice(input_shape[0]), None, slice(input_shape[2]), slice(input_shape[3])]

        # ---------- Iterate over slices and compose final result ------------##
        small_ts = TimeSeries()
        for var in range(input_shape[1]):
            node_slice[1] = slice(var, var + 1)
            small_ts.data = time_series_h5.read_data_slice(tuple(node_slice))
            self.algorithm.time_series = small_ts
            partial_pca = self.algorithm.evaluate()
            pca_h5.write_data_slice(partial_pca)
        pca_h5.close()
        time_series_h5.close()

        return principal_components_index
