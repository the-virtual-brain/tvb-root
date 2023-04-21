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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
.. moduleauthor:: Andrei Mihai <mihai.andrei@codemart.ro>
"""

import json
import numpy
from six import add_metaclass
from abc import ABCMeta
from tvb.adapters.visualizers.time_series import ABCSpaceDisplayer
from tvb.adapters.datatypes.db.spectral import DataTypeMatrix
from tvb.basic.neotraits.api import Attr
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.adapters.arguments_serialisation import parse_slice, slice_str
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.neotraits.forms import TraitDataTypeSelectField, StrField
from tvb.core.neocom import h5
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.core.utils import TVBJSONEncoder


@add_metaclass(ABCMeta)
class ABCMappedArraySVGVisualizer(ABCSpaceDisplayer):
    """
    To be inherited by visualizers for DataTypeMatrix subclasses
    """

    def get_required_memory_size(self, view_model):
        # type: (MatrixVisualizerModel) -> float
        """Return required memory."""
        dtm_index = self.load_entity_by_gid(view_model.datatype)
        input_size = dtm_index.parsed_shape
        return numpy.prod(input_size) * 8.0

    @staticmethod
    def compute_raw_matrix_params(matrix):
        """
        Serializes matrix data, shape and stride metadata to json
        """

        matrix = ABCDisplayer.handle_infinite_values(matrix)
        matrix_data = ABCDisplayer.dump_with_precision(matrix.flat)

        matrix_shape = json.dumps(matrix.shape)

        return dict(matrix_data=matrix_data,
                    matrix_shape=matrix_shape)

    def compute_2d_view(self, dtm_index, slice_s):
        # type: (DataTypeMatrix, str) -> (numpy.array, str, bool)
        """
        Create a 2d view of the matrix using the suggested slice
        If the given slice is invalid or fails to produce a 2d array the default is used
        which selects the first 2 dimensions.
        If the matrix is complex the real part is shown
        :param dtm_index: main input. It can have more then 2D
        :param slice_s: a string representation of a slice
        :return: (a 2d array,  the slice used to make it, is_default_returned)
        """
        default = (slice(None), slice(None)) + tuple(0 for _ in range(dtm_index.ndim - 2))  # [:,:,0,0,0,0 etc]
        slice_used = default

        try:
            if slice_s is not None and slice_s != "":
                slice_used = parse_slice(slice_s)
        except ValueError:  # if the slice could not be parsed
            self.log.warning("failed to parse the slice")

        try:
            with h5.h5_file_for_index(dtm_index) as h5_file:
                result_2d = h5_file.array_data[slice_used]
                result_2d = result_2d.astype(float)
            if result_2d.ndim > 2:  # the slice did not produce a 2d array, treat as error
                raise ValueError(str(dtm_index.shape))
        except (ValueError, IndexError, TypeError):  # if the slice failed to produce a 2d array
            self.log.warning("failed to produce a 2d array")
            return self.compute_2d_view(dtm_index, "")

        return result_2d, slice_str(slice_used), slice_used == default

    def compute_params(self, dtm_index, matrix2d, title_suffix, labels=None,
                       given_slice=None, slice_used=None, is_default_slice=True, has_infinite_values=False):
        # type: (DataTypeMatrix, numpy.array, str, list, str, str, bool) -> dict
        view_pars = self.compute_raw_matrix_params(matrix2d)
        view_pars.update(original_matrix_shape=dtm_index.shape,
                         show_slice_info=True,
                         given_slice=given_slice,
                         slice_used=slice_used,
                         is_default_slice=is_default_slice,
                         has_complex_numbers=dtm_index.array_has_complex,
                         has_infinite_values=has_infinite_values,
                         viewer_title=title_suffix,
                         title=dtm_index.display_name + " - " + title_suffix,
                         matrix_labels=json.dumps(labels, cls=TVBJSONEncoder))
        return view_pars

    def extract_source_labels(self, datatype_matrix):
        # type: (DataTypeMatrix) -> list
        if hasattr(datatype_matrix, "fk_connectivity_gid"):
            with h5.h5_file_for_gid(datatype_matrix.fk_connectivity_gid) as conn_h5:
                labels = list(conn_h5.region_labels.load())
            return labels

        if hasattr(datatype_matrix, "fk_source_gid"):
            with h5.h5_file_for_gid(datatype_matrix.fk_source_gid) as source_h5:
                labels = self.get_space_labels(source_h5)
            return labels

        return None


class MatrixVisualizerModel(ViewModel):
    datatype = DataTypeGidAttr(
        linked_datatype=DataTypeMatrix,
        label='Array data type'
    )

    slice = Attr(
        field_type=str,
        required=False,
        label='slice indices in numpy syntax'
    )


class MatrixVisualizerForm(ABCAdapterForm):

    def __init__(self):
        super(MatrixVisualizerForm, self).__init__()
        self.datatype = TraitDataTypeSelectField(MatrixVisualizerModel.datatype, name='datatype',
                                                 conditions=self.get_filters())
        self.slice = StrField(MatrixVisualizerModel.slice, name='slice')

    @staticmethod
    def get_view_model():
        return MatrixVisualizerModel

    @staticmethod
    def get_input_name():
        return 'datatype'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.ndim'], operations=[">="], values=[2])

    @staticmethod
    def get_required_datatype():
        return DataTypeMatrix


class MappedArrayVisualizer(ABCMappedArraySVGVisualizer):
    _ui_name = "Matrix Visualizer"
    _ui_subsection = "matrix"

    def get_form_class(self):
        return MatrixVisualizerForm

    def launch(self, view_model):
        # type: (MatrixVisualizerModel) -> dict
        dtm_gid = view_model.datatype
        dtm_index = self.load_entity_by_gid(dtm_gid)
        labels = self.extract_source_labels(dtm_index)
        matrix2d, slice_used, is_default_slice = self.compute_2d_view(dtm_index, view_model.slice)

        if matrix2d is None or labels is None or len(labels) != matrix2d.shape[0] or len(labels) != matrix2d.shape[1]:
            labels = None
        else:
            labels = [labels, labels]

        params = self.compute_params(dtm_index, matrix2d, "Matrix Plot", labels,
                                     view_model.slice, slice_used, is_default_slice, not dtm_index.array_is_finite)
        return self.build_display_result("matrix/svg_view", params)
