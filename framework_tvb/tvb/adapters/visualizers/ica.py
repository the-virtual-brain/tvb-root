# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
A matrix displayer for the Independent Component Analysis.
It displays the mixing matrix of siae n_features x n_components

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

from tvb.adapters.visualizers.matrix_viewer import MappedArraySVGVisualizerMixin
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.model.datatypes.mode_decompositions import IndependentComponentsIndex
from tvb.core.neotraits.forms import DataTypeSelectField, SimpleIntField

LOG = get_logger(__name__)


class ICAForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(ICAForm, self).__init__(prefix, project_id)
        self.datatype = DataTypeSelectField(self.get_required_datatype(), self, name='datatype', required=True,
                                            label='Independent component analysis:', conditions=self.get_filters())
        self.i_svar = SimpleIntField(self, name='i_svar', default=0,
                                     label='Index of state variable (defaults to first state variable)')
        self.i_mode = SimpleIntField(self, name='i_mode', default=0, label='Index of mode (defaults to first mode)')

    @staticmethod
    def get_required_datatype():
        return IndependentComponentsIndex

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_input_name():
        return '_datatype'


class ICA(MappedArraySVGVisualizerMixin):
    _ui_name = "Independent Components Analysis Visualizer"

    def get_form_class(self):
        return ICAForm

    def launch(self, datatype, i_svar=0, i_mode=0):
        """Construct data for visualization and launch it."""
        # get data from IndependentComponents datatype, convert to json
        # HACK: dump only a 2D array
        h5_class, h5_path = self._load_h5_of_gid(datatype.gid)
        with h5_class(h5_path) as h5_file:
            unmixing_matrix = h5_file.unmixing_matrix.load()
            prewhitening_matrix = h5_file.prewhitening_matrix.load()

        unmixing_matrix = unmixing_matrix[..., i_svar, i_mode]
        prewhitening_matrix = prewhitening_matrix[..., i_svar, i_mode]
        Cinv = unmixing_matrix.dot(prewhitening_matrix)
        pars = self.compute_params(Cinv, 'ICA region contribution', '(Ellipsis, %d, 0)' % (i_svar))
        return self.build_display_result("matrix/svg_view", pars)
