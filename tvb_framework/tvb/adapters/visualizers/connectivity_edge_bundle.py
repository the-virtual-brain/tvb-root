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
A Javascript displayer for connectivity, using hierarchical edge bundle diagrams from d3.js.

.. moduleauthor:: Vlad Farcas <vlad.farcas@codemart.ro>

"""

import json
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer, URLGenerator
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.core.utils import TVBJSONEncoder
from tvb.datatypes.connectivity import Connectivity


class ConnectivityEdgeBundleModel(ViewModel):
    connectivity = DataTypeGidAttr(
        linked_datatype=Connectivity,
        label="Connectivity to be displayed in a hierarchical edge bundle"
    )


class ConnectivityEdgeBundleForm(ABCAdapterForm):

    def __init__(self):
        super(ConnectivityEdgeBundleForm, self).__init__()
        self.connectivity = TraitDataTypeSelectField(ConnectivityEdgeBundleModel.connectivity, name="connectivity",
                                                     conditions=self.get_filters(), has_all_option=False)

    @staticmethod
    def get_required_datatype():
        return ConnectivityIndex

    @staticmethod
    def get_input_name():
        return 'connectivity'

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_view_model():
        return ConnectivityEdgeBundleModel


class ConnectivityEdgeBundle(ABCDisplayer):
    _ui_name = "Connectivity Edge Bundle View"
    _ui_subsection = "connectivity_edge"

    def get_form_class(self):
        return ConnectivityEdgeBundleForm

    def get_required_memory_size(self, **kwargs):
        """Return required memory."""
        return -1

    def launch(self, view_model):
        """Construct data for visualization and launch it."""

        connectivity = self.load_traited_by_gid(view_model.connectivity)

        pars = {"labels": json.dumps(connectivity.region_labels.tolist(), cls=TVBJSONEncoder),
                "url_base": URLGenerator.paths2url(view_model.connectivity,
                                                   attribute_name="weights", flatten="True")
                }

        return self.build_display_result("connectivity_edge_bundle/view", pars)
