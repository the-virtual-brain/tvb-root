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
A visualizer for cross correlation.

.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

from tvb.adapters.datatypes.db.temporal_correlations import CrossCorrelationIndex
from tvb.adapters.visualizers.matrix_viewer import ABCMappedArraySVGVisualizer
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.temporal_correlations import CrossCorrelation


class CrossCorrelationVisualizerModel(ViewModel):
    datatype = DataTypeGidAttr(
        linked_datatype=CrossCorrelation,
        label='Cross correlation'
    )


class CrossCorrelationVisualizerForm(ABCAdapterForm):

    def __init__(self):
        super(CrossCorrelationVisualizerForm, self).__init__()
        self.datatype = TraitDataTypeSelectField(CrossCorrelationVisualizerModel.datatype, name='datatype')

    @staticmethod
    def get_view_model():
        return CrossCorrelationVisualizerModel

    @staticmethod
    def get_required_datatype():
        return CrossCorrelationIndex

    @staticmethod
    def get_input_name():
        return 'datatype'

    @staticmethod
    def get_filters():
        return None


class CrossCorrelationVisualizer(ABCMappedArraySVGVisualizer):
    _ui_name = "Cross Correlation Visualizer"
    _ui_subsection = "correlation"

    def get_form_class(self):
        return CrossCorrelationVisualizerForm

    def launch(self, view_model):
        # type: (CrossCorrelationVisualizerModel) -> dict
        """Construct data for visualization and launch it."""
        correlation_gid = view_model.datatype
        correlation_index = self.load_entity_by_gid(correlation_gid)
        labels = self.extract_source_labels(correlation_index)
        with h5.h5_file_for_index(correlation_index) as dt_h5:
            matrix = dt_h5.array_data[:]
            matrix = matrix.mean(axis=0)[:, :, 0, 0]
        pars = self.compute_params(correlation_index, matrix, 'Correlation matrix plot', labels=[labels, labels])
        pars['show_slice_info'] = False
        return self.build_display_result("matrix/svg_view", pars)
