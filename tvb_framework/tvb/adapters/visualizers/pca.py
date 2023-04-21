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
A displayer for the principal component analysis.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""
import json
from tvb.adapters.datatypes.db.mode_decompositions import PrincipalComponentsIndex
from tvb.adapters.visualizers.time_series import ABCSpaceDisplayer
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import URLGenerator
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.core.utils import TVBJSONEncoder
from tvb.datatypes.mode_decompositions import PrincipalComponents


class PCAModel(ViewModel):
    pca = DataTypeGidAttr(
        linked_datatype=PrincipalComponents,
        label='Principal component analysis:'
    )


class PCAForm(ABCAdapterForm):

    def __init__(self):
        super(PCAForm, self).__init__()
        self.pca = TraitDataTypeSelectField(PCAModel.pca, name='pca', conditions=self.get_filters())

    @staticmethod
    def get_view_model():
        return PCAModel

    @staticmethod
    def get_input_name():
        return 'pca'

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_required_datatype():
        return PrincipalComponentsIndex


class PCA(ABCSpaceDisplayer):
    _ui_name = "Principal Components Analysis Visualizer"

    def get_form_class(self):
        return PCAForm

    def get_required_memory_size(self, view_model):
        # type: (PCAModel) -> int
        """Return required memory. Here, it's unknown/insignificant."""
        return -1

    def launch(self, view_model):
        # type: (PCAModel) -> dict
        """Construct data for visualization and launch it."""
        with h5.h5_file_for_gid(view_model.pca) as ts_h5:
            source_gid = ts_h5.source.load()

        with h5.h5_file_for_gid(source_gid) as source_h5:
            labels_data = self.get_space_labels(source_h5)

        fractions_update_url = URLGenerator.build_h5_url(view_model.pca, 'read_fractions_data')
        weights_update_url = URLGenerator.build_h5_url(view_model.pca, 'read_weights_data')
        return self.build_display_result("pca/view", dict(labels_data=json.dumps(labels_data, cls=TVBJSONEncoder),
                                                          fractions_update_url=fractions_update_url,
                                                          weights_update_url=weights_update_url))
