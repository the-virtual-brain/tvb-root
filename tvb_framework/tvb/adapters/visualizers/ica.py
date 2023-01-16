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
A matrix visualizer for the Independent Component Analysis.
It displays the mixing matrix of size n_features x n_components

.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
"""

from tvb.adapters.datatypes.db.mode_decompositions import IndependentComponentsIndex
from tvb.adapters.visualizers.matrix_viewer import ABCMappedArraySVGVisualizer
from tvb.basic.neotraits.api import Attr
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.arguments_serialisation import slice_str
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField, IntField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.mode_decompositions import IndependentComponents


class ICAModel(ViewModel):
    datatype = DataTypeGidAttr(
        linked_datatype=IndependentComponents,
        label='Independent component analysis:'
    )

    i_svar = Attr(
        field_type=int,
        default=0,
        label='Index of state variable (defaults to first state variable)'
    )

    i_mode = Attr(
        field_type=int,
        default=0,
        label='Index of mode (defaults to first mode)'
    )


class ICAForm(ABCAdapterForm):

    def __init__(self):
        super(ICAForm, self).__init__()
        self.datatype = TraitDataTypeSelectField(ICAModel.datatype, name='datatype', conditions=self.get_filters())
        self.i_svar = IntField(ICAModel.i_svar, name='i_svar')
        self.i_mode = IntField(ICAModel.i_mode, name='i_mode')

    @staticmethod
    def get_view_model():
        return ICAModel

    @staticmethod
    def get_required_datatype():
        return IndependentComponentsIndex

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_input_name():
        return 'datatype'


class ICA(ABCMappedArraySVGVisualizer):
    _ui_name = "Independent Components Analysis Visualizer"
    _ui_subsection = "ica"

    def get_form_class(self):
        return ICAForm

    def launch(self, view_model):
        # type: (ICAModel) -> dict
        """Construct data for visualization and launch it."""
        ica_gid = view_model.datatype
        ica_index = self.load_entity_by_gid(ica_gid)

        slice_given = slice_str((slice(None), slice(None), slice(view_model.i_svar), slice(view_model.i_mode)))
        if view_model.i_svar < 0 or view_model.i_svar >= ica_index.parsed_shape[2]:
            view_model.i_svar = 0
        if view_model.i_mode < 0 or view_model.i_mode >= ica_index.parsed_shape[3]:
            view_model.i_mode = 0
        slice_used = slice_str((slice(None), slice(None), slice(view_model.i_svar), slice(view_model.i_mode)))

        with h5.h5_file_for_index(ica_index) as h5_file:
            unmixing_matrix = h5_file.unmixing_matrix[..., view_model.i_svar, view_model.i_mode]
            prewhitening_matrix = h5_file.prewhitening_matrix[..., view_model.i_svar, view_model.i_mode]
        Cinv = unmixing_matrix.dot(prewhitening_matrix)

        title = 'ICA region contribution -- (Ellipsis, %d, 0)' % (view_model.i_svar)
        labels = self.extract_source_labels(ica_index)
        pars = self.compute_params(ica_index, Cinv, title, [labels, labels],
                                   slice_given, slice_used, slice_given != slice_used)
        return self.build_display_result("matrix/svg_view", pars)
