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
A displayer for the principal components analysis.

.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""
import json
from tvb.datatypes.mode_decompositions import PrincipalComponents
from tvb.core.adapters.abcdisplayer import ABCDisplayer



class PCA(ABCDisplayer):
    _ui_name = "Principal Components Analysis Visualizer"


    def get_input_tree(self):
        """Inform caller of the data we need"""

        return [{"name": "pca",
                 "type": PrincipalComponents,
                 "label": "Principal component analysis:",
                 "required": True
                 }]


    def get_required_memory_size(self, **kwargs):
        """Return required memory. Here, it's unknown/insignificant."""
        return -1


    def launch(self, pca):
        """Construct data for visualization and launch it."""
        ts_entity = self.load_entity_by_gid(pca.source.gid)
        labels_data = ts_entity.get_space_labels()
        fractions_update_url = self.paths2url(pca, 'read_fractions_data')
        weights_update_url = self.paths2url(pca, 'read_weights_data')
        return self.build_display_result("pca/view", dict(labels_data=json.dumps(labels_data),
                                                          fractions_update_url=fractions_update_url,
                                                          weights_update_url=weights_update_url))


    def generate_preview(self, pca, figure_size=None):
        return self.launch(pca)

