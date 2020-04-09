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
A displayer for covariance.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

from tvb.adapters.visualizers.matrix_viewer import MappedArrayVisualizer
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.adapters.datatypes.db.graph import CovarianceIndex
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.graph import Covariance


class CovarianceVisualizerModel(ViewModel):
    datatype = DataTypeGidAttr(
        linked_datatype=Covariance,
        label='Covariance'
    )


class CovarianceVisualizerForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(CovarianceVisualizerForm, self).__init__(prefix, project_id)
        self.datatype = TraitDataTypeSelectField(CovarianceVisualizerModel.datatype, self, name='datatype')

    @staticmethod
    def get_view_model():
        return CovarianceVisualizerModel

    @staticmethod
    def get_required_datatype():
        return CovarianceIndex

    @staticmethod
    def get_input_name():
        return 'datatype'

    @staticmethod
    def get_filters():
        return None


class CovarianceVisualizer(MappedArrayVisualizer):
    _ui_name = "Covariance Visualizer"

    def get_form_class(self):
        return CovarianceVisualizerForm

    def launch(self, view_model):
        # type: (CovarianceVisualizerModel) -> dict
        """Construct data for visualization and launch it."""
        # get data from corr datatype
        time_series_index = self.load_entity_by_gid(view_model.datatype.hex)
        labels, matrix = self._extract_labels_and_data_matrix(time_series_index)
        pars = self.compute_params(matrix, 'Covariance matrix plot', labels=labels)
        return self.build_display_result("matrix/svg_view", pars)
